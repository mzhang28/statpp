import polars as pl
import orjson
import os
from tqdm import tqdm
from sqlalchemy import create_engine, text

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# SQLAlchemy format: mysql+mysqlconnector://user:pass@host/db
DB_URI = "mysql+mysqlconnector://root:root@127.0.0.1:3306/osu"
OUTPUT_DIR = "./training_data"
BATCH_SIZE = 100000
RULESET_ID = 0  # 0 = osu!standard

# Allowed mods (Simple flags)
ALLOWED_SIMPLE = {"HR", "HD", "FL", "NF", "EZ", "CL"}
# Allowed variable mods and their required speeds
ALLOWED_SPEEDS = {
    "DT": 1.5, "NC": 1.5,
    "HT": 0.75, "DC": 0.75
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_mods_strict(json_bytes):
    """
    Parses the 'data' JSON blob.
    Returns a sorted, comma-separated string of mods (e.g., "DT,HD").
    Returns None if the score contains ANY disallowed mod or invalid setting.
    """
    try:
        # orjson accepts bytes directly, which is faster
        data = orjson.loads(json_bytes)
    except:
        return None

    mods_list = data.get('mods', [])

    # If no mods, return empty string (NoMod)
    if not mods_list:
        return ""

    out_mods = set()

    for m in mods_list:
        acr = m['acronym']

        if acr in ALLOWED_SIMPLE:
            out_mods.add(acr)

        elif acr in ALLOWED_SPEEDS:
            # Check for specific speed settings
            settings = m.get('settings', {})

            # Lazer stores speed_change; if missing, it implies default for that mod context?
            # Safest to assume default is the standard mod value if not present.
            speed = settings.get('speed_change', ALLOWED_SPEEDS[acr])

            # Floating point comparison with small epsilon
            if abs(speed - ALLOWED_SPEEDS[acr]) < 0.01:
                # Normalize NC -> DT and DC -> HT for the model
                if acr == 'NC':
                    out_mods.add('DT')
                elif acr == 'DC':
                    out_mods.add('HT')
                else:
                    out_mods.add(acr)
            else:
                # Found a custom speed (e.g. DT 1.2x) -> Discard entire score
                return None
        else:
            # Found an unranked mod (RX, DA, etc.) -> Discard entire score
            return None

    # Return sorted list
    return ",".join(sorted(list(out_mods)))


def main():
    print("--- 1. Initialization ---")
    engine = create_engine(DB_URI, execution_options={"stream_results": True})

    # Get total count for tqdm
    with engine.connect() as conn:
        print("Counting rows...")
        total_rows = conn.execute(
            text(
                f"SELECT COUNT(*) FROM scores WHERE ruleset_id = {RULESET_ID}")
        ).scalar()
        print(f"Total standard scores to process: {total_rows}")

        # Initialize dense mapping dictionaries
        # We need to build these incrementally or pre-fetch.
        # To save RAM, we will build sets first, then map later.
        user_set = set()
        map_set = set()

    print("\n--- 2. Processing in Batches ---")

    # We will write temporary parquet files per batch to keep RAM low
    batch_idx = 0
    pbar = tqdm(total=total_rows, unit="rows")

    # SQL to fetch data efficiently using seek method (WHERE id > last_id)
    # Using offset is O(N^2) slow; Seeking is O(N)
    last_id = 0

    while True:
        query = text(f"""
            SELECT 
                id, 
                user_id, 
                beatmap_id, 
                accuracy, 
                total_score, 
                data
            FROM scores 
            WHERE ruleset_id = {RULESET_ID} 
            AND id > {last_id}
            ORDER BY id ASC
            LIMIT {BATCH_SIZE}
        """)

        # Read batch using Polars (via pandas to handle the SQL connection nicely)
        # We use pandas read_sql because it handles the Alchemy engine well
        import pandas as pd
        chunk_pd = pd.read_sql(query, engine)

        if chunk_pd.empty:
            break

        # Update cursor
        last_id = chunk_pd['id'].iloc[-1]

        # --- PROCESSING ---

        # 1. Parse Mods (The CPU heavy part)
        # Apply the function row-by-row
        chunk_pd['mods_str'] = chunk_pd['data'].apply(
            lambda x: parse_mods_strict(
                x.encode() if isinstance(x, str) else x)
        )

        # 2. Drop rows where mods were invalid (None)
        chunk_pd = chunk_pd.dropna(subset=['mods_str'])

        if chunk_pd.empty:
            pbar.update(BATCH_SIZE)
            continue

        # 3. Convert to Polars for fast math
        df = pl.from_pandas(chunk_pd)

        # 4. Normalize Score
        # Lazer total_score is typically standardized to 1,000,000 for standard ruleset
        # If it's classic scoring, it might be vastly different.
        # For this baseline, we assume standard lazer scoring logic.
        df = df.with_columns([
            (pl.col("total_score") / 1_000_000.0).clip(0.0,
                                                       1.0).cast(pl.Float32).alias("score_norm"),
            pl.col("accuracy").cast(pl.Float32),
            pl.col("user_id").cast(pl.Int32),
            pl.col("beatmap_id").cast(pl.Int32)
        ])

        # Collect IDs for mapping later
        # (Note: In a massive scale, you might stream this to a side file,
        # but Python sets handle 10k users/50k maps easily in RAM)
        user_set.update(df["user_id"].to_list())
        map_set.update(df["beatmap_id"].to_list())

        # Select columns to save
        save_df = df.select([
            "user_id", "beatmap_id", "score_norm", "accuracy", "mods_str"
        ])

        # Write batch to parquet
        save_df.write_parquet(f"{OUTPUT_DIR}/part_{batch_idx}.parquet")

        batch_idx += 1
        # Update by actual processed rows (pre-filter)
        pbar.update(len(chunk_pd))

        # Explicit garbage collection hint
        del chunk_pd, df, save_df

    pbar.close()

    print("\n--- 3. Consolidating and Mapping ---")

    # 1. Create Mappings
    print("Generating dense ID mappings...")
    sorted_users = sorted(list(user_set))
    sorted_maps = sorted(list(map_set))

    user_map = {uid: i for i, uid in enumerate(sorted_users)}
    beatmap_map = {bid: i for i, bid in enumerate(sorted_maps)}

    # Save Mappings
    with open(f"{OUTPUT_DIR}/mappings_users.json", "w") as f:
        # Save as json string to avoid int64 keys issue in standard json lib
        import json
        json.dump(user_map, f)
    with open(f"{OUTPUT_DIR}/mappings_maps.json", "w") as f:
        import json
        json.dump(beatmap_map, f)

    print(f"Total Unique Users: {len(user_map)}")
    print(f"Total Unique Maps: {len(beatmap_map)}")

    # 2. Rewrite Parquet files with Dense IDs
    # We read the parts back, map them, and write the final big file (or kept chunked)
    print("Rewriting files with dense IDs...")

    # Depending on RAM, we might want to keep them separate.
    # For 50M rows, a single file is fine (~2-3GB).

    schema = {
        "user_idx": pl.Int32,
        "map_idx": pl.Int32,
        "score_norm": pl.Float32,
        "accuracy": pl.Float32,
        "mods": pl.String  # We keep the string representation for now
    }

    # Create a lazy scan of all parts
    lazy_df = pl.scan_parquet(f"{OUTPUT_DIR}/part_*.parquet")

    # We need to map. Polars join is best for this.
    # Convert dicts to DataFrames for joining
    u_map_df = pl.DataFrame({"user_id": sorted_users, "user_idx": range(
        len(sorted_users))}, schema={"user_id": pl.Int32, "user_idx": pl.Int32}).lazy()
    m_map_df = pl.DataFrame({"beatmap_id": sorted_maps, "map_idx": range(
        len(sorted_maps))}, schema={"beatmap_id": pl.Int32, "map_idx": pl.Int32}).lazy()

    final_lazy = (
        lazy_df
        .join(u_map_df, on="user_id", how="inner")
        .join(m_map_df, on="beatmap_id", how="inner")
        .select([
            pl.col("user_idx"),
            pl.col("map_idx"),
            pl.col("score_norm"),
            pl.col("accuracy"),
            pl.col("mods_str").alias("mods")
        ])
    )

    # Execute and write single file
    # If this OOMs, you can stick to processing per-file.
    # But 50M rows with these columns is <2GB RAM.
    final_lazy.sink_parquet(f"{OUTPUT_DIR}/train_final.parquet")

    # Cleanup temp files
    print("Cleaning up temp files...")
    for i in range(batch_idx):
        try:
            os.remove(f"{OUTPUT_DIR}/part_{i}.parquet")
        except:
            pass

    print("Done! Output at train_final.parquet")


if __name__ == "__main__":
    main()
