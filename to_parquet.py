import polars as pl
import orjson
import os
import json
import time
from tqdm import tqdm
from sqlalchemy import create_engine, text
from ossapi import Ossapi  # pip install ossapi
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DB_URI = "mysql+mysqlconnector://root:root@127.0.0.1:3306/osu"
OUTPUT_DIR = "./training_data"
USER_CACHE_FILE = OUTPUT_DIR + "/user_metadata_cache.json"  # <--- NEW CACHE FILE
BATCH_SIZE = 100000
RULESET_ID = 0  # 0 = osu!standard

# API CREDENTIALS (REQUIRED FOR USERNAMES)
CLIENT_ID = 4004        # <--- REPLACE THIS
CLIENT_SECRET = os.environ["OSU_CLIENT_SECRET"]  # <--- REPLACE THIS

# ------------------------------------------------------------------
# MOD LOGIC
# ------------------------------------------------------------------
MOD_MULTIPLIERS = {
    "NF": 1.0,
    "EZ": 0.5,
    "HT": 0.3,
    "HD": 1.06,
    "HR": 1.10,
    "DT": 1.20,
    "NC": 1.20,
    "FL": 1.12,
    "SO": 0.9
}

ALLOWED_SIMPLE = {"HR", "HD", "FL", "NF", "EZ", "CL", "SO"}
ALLOWED_SPEEDS = {"DT": 1.5, "NC": 1.5, "HT": 0.75, "DC": 0.75}

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
                    "artist": row[1],
                    "title": row[2],
                    "version": row[3]
                }
    return meta_map


def fetch_user_metadata_robust(user_ids):
    """
    Fetches usernames with Caching, Rate Limiting (60/min), and Retry Logic.
    """
    print(f"\n--- Robust User Metadata Fetching ---")

    # 1. Load Cache
    cache = {}
    if os.path.exists(USER_CACHE_FILE):
        print(f"Loading user cache from {USER_CACHE_FILE}...")
        try:
            with open(USER_CACHE_FILE, "r") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} cached users.")
        except Exception as e:
            print(f"Cache corrupted or empty: {e}. Starting fresh.")

    # 2. Filter missing users
    # Ensure IDs are integers for comparison set
    all_ids_set = set(user_ids)
    cached_ids_set = set(int(k) for k in cache.keys())
    missing_ids = list(all_ids_set - cached_ids_set)

    print(f"Total Users: {len(all_ids_set)}")
    print(f"Already Cached: {len(cached_ids_set)}")
    print(f"To Fetch: {len(missing_ids)}")

    if not missing_ids:
        return {int(k): v for k, v in cache.items()}

    # 3. Setup API
    try:
        api = Ossapi(CLIENT_ID, CLIENT_SECRET)
    except Exception as e:
        print(f"API Error: {e}")
        return {int(k): v for k, v in cache.items()}

    # 4. Process in Chunks
    chunk_size = 50  # Ossapi limit per request

    # We save every N chunks to disk
    save_interval = 10

    pbar = tqdm(total=len(missing_ids), desc="Fetching API")

    for i in range(0, len(missing_ids), chunk_size):
        chunk = missing_ids[i:i+chunk_size]

        # RETRY LOOP
        retries = 0
        success = False
        while not success and retries < 5:
            try:
                users = api.users(chunk)
                for u in users:
                    cache[str(u.id)] = u.username

                success = True

                # --- CRITICAL RATE LIMITING ---
                # 60 req/min = 1 req/sec. We sleep 1.1s to be safe.
                time.sleep(1.1)
                # ------------------------------

            except Exception as e:
                # Exponential backoff: 5s, 10s, 20s...
                wait_time = (2 ** retries) * 5
                print(f"\nAPI Error on chunk {i}: {e}")
                print(f"Sleeping {wait_time}s before retry...")
                time.sleep(wait_time)
                retries += 1

                # Re-initialize API in case token expired during long sleep
                try:
                    api = Ossapi(CLIENT_ID, CLIENT_SECRET)
                except:
                    pass

        if not success:
            print(f"\nSkipping chunk starting at index {i} after 5 retries.")

        pbar.update(len(chunk))

        # Periodic Save
        if (i // chunk_size) % save_interval == 0:
            with open(USER_CACHE_FILE, "w") as f:
                json.dump(cache, f)

    pbar.close()

    # Final Save
    with open(USER_CACHE_FILE, "w") as f:
        json.dump(cache, f)

    return {int(k): v for k, v in cache.items()}


def main():
    print("--- 1. Initialization ---")
    engine = create_engine(DB_URI, execution_options={"stream_results": True})

    with engine.connect() as conn:
        print("Counting rows...")
        total_rows = conn.execute(
            text(
                f"SELECT COUNT(*) FROM scores WHERE ruleset_id = {RULESET_ID}")
        ).scalar()
        print(f"Total standard scores: {total_rows}")

        user_set = set()
        map_mod_set = set()
        unique_map_ids = set()

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

        user_set.update(df["user_id"].to_list())
        unique_map_ids.update(df["beatmap_id"].to_list())

        current_pairs = zip(df["beatmap_id"].to_list(),
                            df["mods_str"].to_list())
        map_mod_set.update(current_pairs)

        df.write_parquet(f"{OUTPUT_DIR}/part_{batch_idx}.parquet")

        batch_idx += 1
        pbar.update(len(chunk_pd))
        del chunk_pd, df, parsed

    pbar.close()

    print("\n--- 3. Fetching Metadata ---")

    # --- UPDATED FUNCTION CALL ---
    username_map = fetch_user_metadata_robust(user_set)
    # -----------------------------

    map_meta_map = fetch_map_metadata(engine, unique_map_ids)

    print("\n--- 4. Consolidating and Mapping ---")
    print("Mapping Users...")
    sorted_users = sorted(list(user_set))

    user_id_to_idx = {uid: i for i, uid in enumerate(sorted_users)}

    user_json_map = {}
    for uid in sorted_users:
        user_json_map[str(uid)] = {
            "idx": user_id_to_idx[uid],
            "name": username_map.get(uid, "Unknown")
        }

    print("Mapping Map+Mod combinations...")
    sorted_map_mods = sorted(list(map_mod_set), key=lambda x: (x[0], x[1]))
    map_mod_to_idx = {pair: i for i, pair in enumerate(sorted_map_mods)}

    map_json_map = {}
    for (bid, mods), idx in map_mod_to_idx.items():
        meta = map_meta_map.get(
            bid, {"artist": "?", "title": "?", "version": "?"})
        key = f"{bid}|{mods}"
        map_json_map[key] = {
            "idx": idx,
            "artist": meta['artist'],
            "title": meta['title'],
            "version": meta['version']
        }

    with open(f"{OUTPUT_DIR}/mappings_users.json", "w") as f:
        json.dump(user_json_map, f, indent=2)

    with open(f"{OUTPUT_DIR}/mappings_maps.json", "w") as f:
        json.dump(map_json_map, f, indent=2)

    print(
        f"Saved mappings for {len(sorted_users)} users and {len(sorted_map_mods)} items.")

    print("Rewriting final parquet...")
    u_map_df = pl.DataFrame({
        "user_id": sorted_users,
        "user_idx": range(len(sorted_users))
    }, schema={"user_id": pl.Int32, "user_idx": pl.Int32}).lazy()

    map_ids = [x[0] for x in sorted_map_mods]
    mod_strs = [x[1] for x in sorted_map_mods]
    map_idxs = list(range(len(sorted_map_mods)))

    m_map_df = pl.DataFrame({
        "beatmap_id": map_ids,
        "mods_str": mod_strs,
        "map_idx": map_idxs
    }, schema={"beatmap_id": pl.Int32, "mods_str": pl.String, "map_idx": pl.Int32}).lazy()

    lazy_df = pl.scan_parquet(f"{OUTPUT_DIR}/part_*.parquet")

    final_lazy = (
        lazy_df
        .join(u_map_df, on="user_id", how="inner")
        .join(m_map_df, on=["beatmap_id", "mods_str"], how="inner")
        .select([
            pl.col("user_idx"),
            pl.col("map_idx"),
            pl.col("score_norm"),
            pl.col("accuracy"),
            pl.col("mods_str").alias("mods")
        ])
    )

    final_lazy.sink_parquet(f"{OUTPUT_DIR}/train_final.parquet")

    print("Cleaning up temporary files...")
    for i in range(batch_idx):
        try:
            os.remove(f"{OUTPUT_DIR}/part_{i}.parquet")
        except:
            pass

    print("Done! train_final.parquet created.")


if __name__ == "__main__":
    main()
