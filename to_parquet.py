import polars as pl
import orjson
import os
import json
from tqdm import tqdm
from sqlalchemy import create_engine, text

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DB_URI = "mysql+mysqlconnector://root:root@127.0.0.1:3306/osu"
OUTPUT_DIR = "./training_data"
BATCH_SIZE = 100000
RULESET_ID = 0  # 0 = osu!standard

# Allowed flags
# CL is allowed in input, but will be stripped during parsing
ALLOWED_SIMPLE = {"HR", "HD", "FL", "NF", "EZ", "CL"}
ALLOWED_SPEEDS = {
    "DT": 1.5, "NC": 1.5,
    "HT": 0.75, "DC": 0.75
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_mods_strict(json_bytes):
    """
    Parses 'data'. 
    - Normalizes NC->DT, DC->HT.
    - Removes CL completely.
    - Returns comma-separated string (e.g., "DT,HD").
    - Returns None if invalid.
    """
    try:
        data = orjson.loads(json_bytes)
    except:
        return None

    mods_list = data.get('mods', [])

    # If strictly empty list
    if not mods_list:
        return ""

    out_mods = set()

    for m in mods_list:
        acr = m['acronym']

        if acr in ALLOWED_SIMPLE:
            # IGNORE CL (Classic)
            if acr == "CL":
                continue
            out_mods.add(acr)

        elif acr in ALLOWED_SPEEDS:
            settings = m.get('settings', {})
            speed = settings.get('speed_change', ALLOWED_SPEEDS[acr])

            if abs(speed - ALLOWED_SPEEDS[acr]) < 0.01:
                # Normalize NC -> DT and DC -> HT
                if acr == 'NC':
                    out_mods.add('DT')
                elif acr == 'DC':
                    out_mods.add('HT')
                else:
                    out_mods.add(acr)
            else:
                return None  # Custom speed -> Invalid
        else:
            return None  # Unranked mod -> Invalid

    return ",".join(sorted(list(out_mods)))


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

        # user_set tracks unique User IDs
        user_set = set()
        # map_mod_set tracks unique (BeatmapID, ModsString) tuples
        map_mod_set = set()

    print("\n--- 2. Processing in Batches ---")

    batch_idx = 0
    pbar = tqdm(total=total_rows, unit="rows")
    last_id = 0

    while True:
        query = text(f"""
            SELECT 
                id, user_id, beatmap_id, accuracy, total_score, data
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

        # 1. Parse Mods
        chunk_pd['mods_str'] = chunk_pd['data'].apply(
            lambda x: parse_mods_strict(
                x.encode() if isinstance(x, str) else x)
        )

        # 2. Drop Invalid
        chunk_pd = chunk_pd.dropna(subset=['mods_str'])

        if chunk_pd.empty:
            pbar.update(BATCH_SIZE)
            continue

        # 3. Convert to Polars
        df = pl.from_pandas(chunk_pd)

        # 4. Normalize Data
        df = df.with_columns([
            (pl.col("total_score") / 1_000_000.0).clip(0.0,
                                                       1.0).cast(pl.Float32).alias("score_norm"),
            pl.col("accuracy").cast(pl.Float32),
            pl.col("user_id").cast(pl.Int32),
            pl.col("beatmap_id").cast(pl.Int32)
        ])

        # 5. Update Unique Sets
        # Users
        user_set.update(df["user_id"].to_list())

        # Maps + Mods (We need both to define a unique "Item")
        # We zip them into tuples: (1001, "HD,DT")
        current_pairs = zip(df["beatmap_id"].to_list(),
                            df["mods_str"].to_list())
        map_mod_set.update(current_pairs)

        # 6. Save Temp
        save_df = df.select(
            ["user_id", "beatmap_id", "score_norm", "accuracy", "mods_str"])
        save_df.write_parquet(f"{OUTPUT_DIR}/part_{batch_idx}.parquet")

        batch_idx += 1
        pbar.update(len(chunk_pd))
        del chunk_pd, df, save_df

    pbar.close()

    print("\n--- 3. Consolidating and Mapping ---")

    # 1. Generate User Mappings
    print("Mapping Users...")
    sorted_users = sorted(list(user_set))
    user_map = {uid: i for i, uid in enumerate(sorted_users)}

    # 2. Generate Map+Mod Mappings
    print("Mapping Map+Mod combinations...")
    # Sort by ID then Mod string for consistency
    sorted_map_mods = sorted(list(map_mod_set), key=lambda x: (x[0], x[1]))

    # map_mod_map: (beatmap_id, mods_str) -> dense_idx
    map_mod_map = {pair: i for i, pair in enumerate(sorted_map_mods)}

    print(f"Total Unique Users: {len(user_map)}")
    print(f"Total Unique Map+Mod Combos: {len(map_mod_map)}")

    # 3. Save Mappings to JSON
    # For Users: Simple ID -> Index
    with open(f"{OUTPUT_DIR}/mappings_users.json", "w") as f:
        json.dump(user_map, f)

    # For Maps: We need a string key for JSON.
    # Format: "BEATMAP_ID|MODS" -> Index (e.g., "55432|HD,DT": 42)
    json_friendly_map = {f"{bid}|{m}": idx for (
        bid, m), idx in map_mod_map.items()}

    with open(f"{OUTPUT_DIR}/mappings_maps.json", "w") as f:
        json.dump(json_friendly_map, f)

    # 4. Rewrite Parquet
    print("Rewriting final parquet...")

    # Create User DataFrame for Join
    u_map_df = pl.DataFrame({
        "user_id": sorted_users,
        "user_idx": range(len(sorted_users))
    }, schema={"user_id": pl.Int32, "user_idx": pl.Int32}).lazy()

    # Create Map+Mod DataFrame for Join
    # Unzip the pairs list
    map_ids = [x[0] for x in sorted_map_mods]
    mod_strs = [x[1] for x in sorted_map_mods]
    map_idxs = list(range(len(sorted_map_mods)))

    m_map_df = pl.DataFrame({
        "beatmap_id": map_ids,
        "mods_str": mod_strs,
        "map_idx": map_idxs
    }, schema={"beatmap_id": pl.Int32, "mods_str": pl.String, "map_idx": pl.Int32}).lazy()

    # Process
    lazy_df = pl.scan_parquet(f"{OUTPUT_DIR}/part_*.parquet")

    final_lazy = (
        lazy_df
        # Join User ID
        .join(u_map_df, on="user_id", how="inner")
        # Join on BOTH Beatmap ID and Mod String
        .join(m_map_df, on=["beatmap_id", "mods_str"], how="inner")
        .select([
            pl.col("user_idx"),
            pl.col("map_idx"),  # This now represents specific Map+Mod combo
            pl.col("score_norm"),
            pl.col("accuracy"),
            pl.col("mods_str").alias("mods")
        ])
    )

    final_lazy.sink_parquet(f"{OUTPUT_DIR}/train_final.parquet")

    # Cleanup
    print("Cleaning up...")
    for i in range(batch_idx):
        try:
            os.remove(f"{OUTPUT_DIR}/part_{i}.parquet")
        except:
            pass

    print("Done! train_final.parquet created.")


if __name__ == "__main__":
    main()
