import polars as pl
import orjson
import os
import json
import time
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine, text
from ossapi import Ossapi  # pip install ossapi
from dotenv import load_dotenv
from scipy.interpolate import PchipInterpolator

load_dotenv()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DB_URI = "mysql+mysqlconnector://root:root@127.0.0.1:3306/osu"
OUTPUT_DIR = "./training_data"
USER_CACHE_FILE = OUTPUT_DIR + "/user_metadata_cache.json"
BATCH_SIZE = 100000
RULESET_ID = 0  # 0 = osu!standard

# API CREDENTIALS
CLIENT_ID = int(os.environ.get("OSU_CLIENT_ID", 4004))
CLIENT_SECRET = os.environ.get("OSU_CLIENT_SECRET", "")

# ------------------------------------------------------------------
# MOD LOGIC
# ------------------------------------------------------------------
MOD_MULTIPLIERS = {
    "NF": 1.0, "EZ": 0.5, "HT": 0.3, "HD": 1.06, "HR": 1.10,
    "DT": 1.20, "NC": 1.20, "FL": 1.12, "SO": 0.9
}
ALLOWED_SIMPLE = {"HR", "HD", "FL", "NF", "EZ", "CL", "SO"}
ALLOWED_SPEEDS = {"DT": 1.5, "NC": 1.5, "HT": 0.75, "DC": 0.75}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------


def parse_mods_strict(json_bytes):
    try:
        data = orjson.loads(json_bytes)
    except:
        return None, 1.0

    mods_list = data.get('mods', [])
    if not mods_list:
        return "", 1.0

    out_mods = set()
    total_multiplier = 1.0

    for m in mods_list:
        acr = m['acronym']
        if acr in ALLOWED_SIMPLE:
            if acr == "CL":
                continue
            out_mods.add(acr)
            total_multiplier *= MOD_MULTIPLIERS.get(acr, 1.0)
        elif acr in ALLOWED_SPEEDS:
            settings = m.get('settings', {})
            speed = settings.get('speed_change', ALLOWED_SPEEDS[acr])
            if abs(speed - ALLOWED_SPEEDS[acr]) < 0.01:
                norm_acr = 'DT' if acr == 'NC' else (
                    'HT' if acr == 'DC' else acr)
                if norm_acr not in out_mods:
                    out_mods.add(norm_acr)
                    total_multiplier *= MOD_MULTIPLIERS.get(norm_acr, 1.0)
            else:
                return None, 1.0
        else:
            return None, 1.0

    return ",".join(sorted(list(out_mods))), total_multiplier


def fetch_map_metadata(engine, beatmap_ids):
    print(f"Fetching metadata for {len(beatmap_ids)} maps via SQL...")
    id_list = list(beatmap_ids)
    chunk_size = 5000
    meta_map = {}

    for i in tqdm(range(0, len(id_list), chunk_size), desc="Map Metadata"):
        chunk = id_list[i:i+chunk_size]
        if not chunk:
            continue
        ids_str = ",".join(map(str, chunk))
        query = text(f"""
            SELECT b.beatmap_id, s.artist, s.title, b.version 
            FROM osu_beatmaps b
            JOIN osu_beatmapsets s ON b.beatmapset_id = s.beatmapset_id
            WHERE b.beatmap_id IN ({ids_str})
        """)
        with engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                meta_map[row[0]] = {
                    "artist": row[1], "title": row[2], "version": row[3]
                }
    return meta_map


def fetch_user_metadata_robust(user_ids):
    print(f"\n--- Robust User Metadata Fetching ---")
    cache = {}
    if os.path.exists(USER_CACHE_FILE):
        print(f"Loading user cache from {USER_CACHE_FILE}...")
        try:
            with open(USER_CACHE_FILE, "r") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Cache corrupted: {e}")

    all_ids_set = set(user_ids)
    cached_ids_set = set(int(k) for k in cache.keys())
    missing_ids = list(all_ids_set - cached_ids_set)

    print(
        f"Total: {len(all_ids_set)} | Cached: {len(cached_ids_set)} | Missing: {len(missing_ids)}")

    if not missing_ids:
        return {int(k): v for k, v in cache.items()}

    try:
        api = Ossapi(CLIENT_ID, CLIENT_SECRET)
    except Exception as e:
        print(f"API Error (Check Credentials): {e}")
        return {int(k): v for k, v in cache.items()}

    chunk_size = 50
    save_interval = 10
    pbar = tqdm(total=len(missing_ids), desc="Fetching API")

    for i in range(0, len(missing_ids), chunk_size):
        chunk = missing_ids[i:i+chunk_size]
        retries = 0
        success = False
        while not success and retries < 5:
            try:
                users = api.users(chunk)
                for u in users:
                    cache[str(u.id)] = u.username
                success = True
                time.sleep(1.1)
            except Exception as e:
                wait_time = (2 ** retries) * 5
                print(f"\nAPI Error: {e}. Retry in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1
                try:
                    api = Ossapi(CLIENT_ID, CLIENT_SECRET)
                except:
                    pass

        pbar.update(len(chunk))
        if (i // chunk_size) % save_interval == 0:
            with open(USER_CACHE_FILE, "w") as f:
                json.dump(cache, f)

    pbar.close()
    with open(USER_CACHE_FILE, "w") as f:
        json.dump(cache, f)

    return {int(k): v for k, v in cache.items()}

# ------------------------------------------------------------------
# NEW: SPLINE NORMALIZATION
# ------------------------------------------------------------------


def fit_and_transform_spline(df_pandas):
    """
    Applied per map-group. Fits a PCHIP spline to the score distribution
    and replaces 'score_norm' with the percentile rank.
    """
    # 1. Get raw scores
    scores = df_pandas["score_norm"].values

    # 2. Compute Empirical CDF (Quantiles)
    # We want Unique scores to map to their Cumulative Frequency
    unique_scores, inverse_indices = np.unique(scores, return_inverse=True)

    # If map has too few data points, return simple linear rank or raw
    if len(unique_scores) < 3:
        # Fallback: simple rank normalized
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # Calculate percentile for each unique score
    # We use 'searchsorted' to find how many scores are <= current score
    # Ideally, we want the "end" of the step, so side='right'
    counts = np.searchsorted(scores, unique_scores, side='right')
    # Use explicit sorting if searchsorted assumes sorted input (it requires it)
    # Since we have raw unique_scores which are sorted, and we need counts in 'scores'
    # Easier: Just sort all scores first

    sorted_all = np.sort(scores)
    # Ranks: position in the sorted array / total N
    # For unique values, we take the LAST position (cumulative count)
    cumulative_counts = np.searchsorted(
        sorted_all, unique_scores, side='right')
    percentiles = cumulative_counts / len(scores)

    # 3. Add Anchors (0.0 -> 0.0 and 1.0 -> 1.0) for stability
    # This ensures a score of 0 is always 0% and max possible is 100%
    x_points = np.concatenate(([0.0], unique_scores, [1.0]))
    y_points = np.concatenate(([0.0], percentiles, [1.0]))

    # Handle duplicates in anchors (if unique_scores contained 0.0 or 1.0)
    x_points, unique_indices = np.unique(x_points, return_index=True)
    y_points = y_points[unique_indices]

    # 4. Fit Spline
    # PchipInterpolator guarantees monotonicity (no dips)
    try:
        spline = PchipInterpolator(x_points, y_points)
        transformed_scores = spline(scores)

        # Clip just in case of float errors
        return np.clip(transformed_scores, 0.0, 1.0).astype(np.float32)
    except Exception:
        # Fallback if fit fails
        return scores


def apply_spline_normalization(output_dir):
    print("\n--- 5. Applying PCHIP Spline Normalization ---")
    parquet_path = f"{output_dir}/train_final.parquet"

    if not os.path.exists(parquet_path):
        print("Error: train_final.parquet not found.")
        return

    print("Loading full dataset into memory (Polars)...")
    # Load all columns. We need map_idx to group, score_norm to transform.
    df = pl.read_parquet(parquet_path)

    print(f"Loaded {len(df)} rows. Grouping by Map for Spline fitting...")

    # We convert to pandas for the complex Apply, or iterate.
    # Iterating over 100k groups in python is slow, but PCHIP is complex.
    # Let's iterate on the unique map indices to save memory overhead of groupby objects

    # OPTIMIZATION: Work in chunks of maps to avoid exploding RAM if we did a full groupby apply
    # But for simplicity, we'll try a direct approach first.

    map_indices = df["map_idx"].unique().to_list()

    # We will build a new Score column
    # To do this efficiently, we sort the DF by map_idx first
    df = df.sort(["map_idx"])

    # Convert to pandas for mutation
    pdf = df.to_pandas()

    new_scores = np.zeros(len(pdf), dtype=np.float32)

    # We iterate over the slices manually
    # Since it's sorted, we can find start/end indices
    print(f"Processing splines for {len(map_indices)} maps...")

    # Extract map_idx column as numpy array for fast indexing
    map_idx_arr = pdf["map_idx"].values

    # Find change points (where map_idx changes)
    # This is much faster than df.groupby()
    change_indices = np.where(map_idx_arr[:-1] != map_idx_arr[1:])[0] + 1
    split_indices = np.concatenate(([0], change_indices, [len(pdf)]))

    for i in tqdm(range(len(split_indices) - 1), unit="map"):
        start = split_indices[i]
        end = split_indices[i+1]

        # Slice the scores
        map_slice = pdf.iloc[start:end]

        # Calculate Spline Scores
        transformed = fit_and_transform_spline(map_slice)

        # Write back to the numpy array
        new_scores[start:end] = transformed

    print("Updating DataFrame...")
    pdf["score_norm"] = new_scores

    print("Saving transformed parquet...")
    final_df = pl.from_pandas(pdf)
    final_df.write_parquet(parquet_path)  # Overwrite
    print("Done! Scores have been transformed to Spline Percentiles.")

# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------


def main():
    print("--- 1. Initialization ---")
    engine = create_engine(DB_URI, execution_options={"stream_results": True})

    with engine.connect() as conn:
        total_rows = conn.execute(
            text(
                f"SELECT COUNT(*) FROM scores WHERE ruleset_id = {RULESET_ID}")
        ).scalar()
        print(f"Total standard scores: {total_rows}")

    print("\n--- 2. Processing in Batches ---")
    batch_idx = 0
    pbar = tqdm(total=total_rows, unit="rows")
    last_id = 0

    while True:
        query = text(f"""
            SELECT id, user_id, beatmap_id, accuracy, total_score, data
            FROM scores 
            WHERE ruleset_id = {RULESET_ID} 
            AND id > {last_id}
            ORDER BY id ASC
            LIMIT {BATCH_SIZE}
        """)

        import pandas as pd
        chunk_pd = pd.read_sql(query, engine)

        if chunk_pd.empty:
            break

        last_id = chunk_pd['id'].iloc[-1]

        parsed = chunk_pd['data'].apply(
            lambda x: parse_mods_strict(
                x.encode() if isinstance(x, str) else x)
        )

        chunk_pd['mods_str'] = [x[0] if x else None for x in parsed]
        chunk_pd['multiplier'] = [x[1] if x else 1.0 for x in parsed]

        chunk_pd = chunk_pd.dropna(subset=['mods_str'])

        if chunk_pd.empty:
            pbar.update(BATCH_SIZE)
            continue

        chunk_pd['score_norm'] = (
            (chunk_pd['total_score'] / chunk_pd['multiplier']) / 1_000_000.0
        ).clip(0.0, 1.0)

        df = pl.from_pandas(chunk_pd).select([
            pl.col("user_id").cast(pl.Int32),
            pl.col("beatmap_id").cast(pl.Int32),
            pl.col("score_norm").cast(pl.Float32),
            pl.col("accuracy").cast(pl.Float32),
            pl.col("mods_str")
        ])

        df.write_parquet(f"{OUTPUT_DIR}/part_{batch_idx}.parquet")

        batch_idx += 1
        pbar.update(len(chunk_pd))
        del chunk_pd, df, parsed

    pbar.close()

    print("\n--- 3. Fetching Metadata ---")
    # Collect IDs from parquet parts
    user_set = set()
    map_mod_set = set()

    # Quick scan to get IDs
    print("Scanning parts for IDs...")
    for i in range(batch_idx):
        df_scan = pl.read_parquet(
            f"{OUTPUT_DIR}/part_{i}.parquet", columns=["user_id", "beatmap_id", "mods_str"])
        user_set.update(df_scan["user_id"].to_list())
        pairs = zip(df_scan["beatmap_id"].to_list(),
                    df_scan["mods_str"].to_list())
        map_mod_set.update(pairs)

    username_map = fetch_user_metadata_robust(user_set)

    # Extract unique map IDs for SQL query
    unique_map_ids = set([m[0] for m in map_mod_set])
    map_meta_map = fetch_map_metadata(engine, unique_map_ids)

    print("\n--- 4. Consolidating and Mapping ---")
    sorted_users = sorted(list(user_set))
    user_id_to_idx = {uid: i for i, uid in enumerate(sorted_users)}

    user_json_map = {}
    for uid in sorted_users:
        user_json_map[str(uid)] = {
            "idx": user_id_to_idx[uid],
            "name": username_map.get(uid, "Unknown")
        }

    sorted_map_mods = sorted(list(map_mod_set), key=lambda x: (x[0], x[1]))
    map_mod_to_idx = {pair: i for i, pair in enumerate(sorted_map_mods)}

    map_json_map = {}
    for (bid, mods), idx in map_mod_to_idx.items():
        meta = map_meta_map.get(
            bid, {"artist": "?", "title": "?", "version": "?"})
        key = f"{bid}|{mods}"
        map_json_map[key] = {
            "idx": idx, "artist": meta['artist'],
            "title": meta['title'], "version": meta['version']
        }

    import zstandard
    print("Saving compressed mappings...")
    cctx = zstandard.ZstdCompressor()
    with open(f"{OUTPUT_DIR}/mappings_users.json.zst", "wb") as f:
        f.write(cctx.compress(json.dumps(user_json_map).encode()))

    with open(f"{OUTPUT_DIR}/mappings_maps.json.zst", "wb") as f:
        f.write(cctx.compress(json.dumps(map_json_map).encode()))

    print("Rewriting final parquet...")
    u_map_df = pl.DataFrame({
        "user_id": sorted_users,
        "user_idx": range(len(sorted_users))
    }, schema={"user_id": pl.Int32, "user_idx": pl.Int32}).lazy()

    map_ids = [x[0] for x in sorted_map_mods]
    mod_strs = [x[1] for x in sorted_map_mods]
    map_idxs = list(range(len(sorted_map_mods)))

    m_map_df = pl.DataFrame({
        "beatmap_id": map_ids, "mods_str": mod_strs, "map_idx": map_idxs
    }, schema={"beatmap_id": pl.Int32, "mods_str": pl.String, "map_idx": pl.Int32}).lazy()

    lazy_df = pl.scan_parquet(f"{OUTPUT_DIR}/part_*.parquet")

    final_lazy = (
        lazy_df
        .join(u_map_df, on="user_id", how="inner")
        .join(m_map_df, on=["beatmap_id", "mods_str"], how="inner")
        .select([
            pl.col("user_idx"), pl.col("map_idx"),
            pl.col("score_norm"), pl.col("accuracy"),
            pl.col("mods_str").alias("mods")
        ])
    )

    final_lazy.sink_parquet(f"{OUTPUT_DIR}/train_final.parquet")

    # Cleanup parts
    for i in range(batch_idx):
        try:
            os.remove(f"{OUTPUT_DIR}/part_{i}.parquet")
        except:
            pass

    # --- STEP 5: SPLINE NORMALIZATION ---
    apply_spline_normalization(OUTPUT_DIR)

    print("\n--- Processing Complete ---")


if __name__ == "__main__":
    main()
