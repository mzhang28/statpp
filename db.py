import mysql.connector

conn = mysql.connector.connect(
    host="localhost", user="osu", password="osu", database="osu"
)

def fetch_all_scores(limit=100000):
    cur = conn.cursor()
    cur.execute(f"select user_id, beatmap_id, accuracy, total_score, json_extract(data, '$.mods') from scores order by beatmap_id desc limit {limit}")
    data = cur.fetchall()
    return data

with open("mods.txt") as f:
    all_mods = list(map(lambda s: s.strip(), f.readlines()))
all_mods_map = enumerate(all_mods)
all_mods_map = map(lambda s: (s[1], s[0]), all_mods_map)
all_mods_map = dict(list(all_mods_map))
all_mods_rev = {v: k for k, v in all_mods_map.items()}
IGNORED_MODS = set(["NF", "CL"])
def mods_to_bits(mods):
    n = 0
    for mod in mods:
        if mod["acronym"] in IGNORED_MODS: continue
        n |= 1 << all_mods_map[mod["acronym"]]
    return n

def bits_to_mods(bits):
    mods = []
    for i in range(len(all_mods)):
        if bits & (1 << i):
            mods.append(all_mods_rev[i])
    return mods
