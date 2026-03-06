# extract from the db and write to a parquet file

import os
from multiprocessing import Pool, cpu_count

import orjson
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from tqdm import tqdm

load_dotenv()

DB_URI = "mysql+mysqlconnector://root:root@127.0.0.1:3306/osu"
OUTPUT_DIR = "./training_data"
BATCH_SIZE = 100000

MOD_MULTS = {
    "NF": 1.0,
    "EZ": 0.5,
    "HT": 0.3,
    "HD": 1.06,
    "HR": 1.10,
    "DT": 1.20,
    "NC": 1.20,
    "FL": 1.12,
    "SO": 0.9,
}
SIMPLE_MODS = {"HR", "HD", "FL", "NF", "EZ", "CL", "SO"}
SPEED_MODS = {"DT": 1.5, "NC": 1.5, "HT": 0.75, "DC": 0.75}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_score(raw):
    data = orjson.loads(raw if isinstance(raw, bytes) else raw.encode())

    mods = set()
    mult = 1.0
    for m in data.get("mods", []):
        acr = m["acronym"]
        if acr in SIMPLE_MODS:
            if acr != "CL":
                mods.add(acr)
                mult *= MOD_MULTS.get(acr, 1.0)
        elif acr in SPEED_MODS:
            speed = m.get("settings", {}).get("speed_change", SPEED_MODS[acr])
            if abs(speed - SPEED_MODS[acr]) < 0.01:
                norm = "DT" if acr == "NC" else ("HT" if acr == "DC" else acr)
                if norm not in mods:
                    mods.add(norm)
                    mult *= MOD_MULTS.get(norm, 1.0)
            else:
                return None, 1.0, 0
        else:
            return None, 1.0, 0

    mods_str = ",".join(sorted(mods)) if mods else ""
    misses = data.get("statistics", {}).get("miss", 0)
    return mods_str, mult, misses


def main():
    engine = create_engine(DB_URI, execution_options={"stream_results": True})

    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM scores WHERE ruleset_id = 0")
        ).scalar()
    print(f"{total} scores")

    last_id = 0
    batch_idx = 0
    pool = Pool(cpu_count())
    pbar = tqdm(total=total, unit="rows")

    while True:
        q = text(f"""
            SELECT id, user_id, beatmap_id, accuracy, total_score, data
            FROM scores WHERE ruleset_id = 0 AND id > {last_id}
            ORDER BY id ASC LIMIT {BATCH_SIZE}
        """)
        chunk = pd.read_sql(q, engine)
        if chunk.empty:
            break
        last_id = chunk["id"].iloc[-1]

        parsed = pool.map(parse_score, chunk["data"])
        chunk["mods"] = [p[0] for p in parsed]
        chunk["multiplier"] = [p[1] for p in parsed]
        chunk["miss_count"] = [p[2] for p in parsed]
        chunk = chunk.dropna(subset=["mods"])

        if not chunk.empty:
            chunk["score_norm"] = (
                chunk["total_score"] / chunk["multiplier"] / 1_000_000
            ).clip(0.0, 1.0)
            pl.from_pandas(chunk).select(
                [
                    pl.col("user_id").cast(pl.Int32),
                    pl.col("beatmap_id").cast(pl.Int32),
                    pl.col("score_norm").cast(pl.Float32),
                    pl.col("accuracy").cast(pl.Float32),
                    pl.col("miss_count").cast(pl.Int32),
                    pl.col("mods"),
                ]
            ).write_parquet(f"{OUTPUT_DIR}/part_{batch_idx}.parquet")
            batch_idx += 1

        pbar.update(len(chunk))

    pbar.close()
    pool.close()
    pool.join()

    print("consolidating")
    df = pl.scan_parquet(f"{OUTPUT_DIR}/part_*.parquet")
    df.sink_parquet(f"{OUTPUT_DIR}/scores.parquet")

    for i in range(batch_idx):
        os.remove(f"{OUTPUT_DIR}/part_{i}.parquet")

    print("done")


if __name__ == "__main__":
    main()
