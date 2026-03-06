import polars as pl

INPUT = "./training_data/scores.parquet"
OUTPUT = "./training_data/scores_indexed.parquet"

print("loading")
df = pl.read_parquet(INPUT)

users = df["user_id"].unique().sort()
maps = df.select(["beatmap_id", "mods"]).unique().sort(["beatmap_id", "mods"])

user_map = pl.DataFrame({"user_id": users, "user_idx": range(
    len(users))}).cast({"user_idx": pl.Int32})
map_map = maps.with_columns(
    pl.Series("map_idx", range(len(maps)), dtype=pl.Int32))

print(f"{len(user_map)} users, {len(map_map)} maps")

df = df.join(user_map, on="user_id").join(map_map, on=["beatmap_id", "mods"])
df.write_parquet(OUTPUT)

print(f"wrote {len(df)} rows to {OUTPUT}")
