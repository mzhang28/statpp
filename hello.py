import torch
import json
from collections import defaultdict
import mysql.connector

conn = mysql.connector.connect(
    host="localhost", user="osu", password="osu", database="osu"
)
cur = conn.cursor()
cur.execute("select user_id, beatmap_id, accuracy, total_score, json_extract(data, '$.mods') from scores order by beatmap_id desc limit 100000")
data = cur.fetchall()

IGNORED_MODS = set(["NF", "CL"])
existing_mods = dict()
ctr = 1
def mods_to_bits(mods):
    global ctr
    n = 0
    for mod in mods:
        if mod["acronym"] in IGNORED_MODS: continue
        if mod["acronym"] not in existing_mods:
            existing_mods[mod["acronym"]] = ctr
            ctr += 1
        n |= 1 << existing_mods[mod["acronym"]]
    return n
def bits_to_mods(bits):
    mods = []
    for i in range(ctr):
        if bits & (1 << i):
            mods.append(list(existing_mods.keys())[i])
    return mods

def transform1(row):
    user_id, beatmap_id, accuracy, total_score, mods = row
    mod_bits = mods_to_bits(json.loads(mods))
    return user_id, beatmap_id, mod_bits, total_score / 1_000_000.0
data = list(map(transform1, data))

user_counts = defaultdict(int)
mapmod_counts = defaultdict(int)
for user_id, beatmap_id, mod_bits, total_score in data:
    user_counts[user_id] += 1
    mapmod_counts[(beatmap_id, mod_bits)] += 1

MIN_COUNT = 10
REG = 1e-2
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def filter2(row):
    user_id, beatmap_id, mod_bits, total_score = row
    return user_counts[user_id] >= MIN_COUNT and mapmod_counts[beatmap_id, mod_bits] >= MIN_COUNT
data = list(filter(filter2, data))

print(len(data))
users = sorted({user_id for user_id, _, _, _ in data})
maps_mods = sorted({(beatmap_id, mod_bits) for _, beatmap_id, mod_bits, _ in data})
uid = {u: i for i, u in enumerate(users)}
mmid = {mm: i for i, mm in enumerate(maps_mods)}
n_users, n_maps = len(users), len(maps_mods)
print(n_users, n_maps)

rows, cols, targets = [], [], []
for u, beatmap_id, mod_bits, s in data:
    mm = (beatmap_id, mod_bits)
    rows.append(len(targets))
    cols.append(uid[u])
    targets.append(s)
    rows.append(len(targets))
    cols.append(n_users + mmid[mm])
    targets.append(-s)

X = torch.zeros(len(rows), n_users + n_maps, dtype=torch.float32, device=device)
X[torch.arange(len(rows)), cols] = 1.0
y = torch.tensor(targets, dtype=torch.float32, device=device)

XtX = X.T @ X
Xty = X.T @ y
theta = torch.linalg.solve(XtX + REG * torch.eye(XtX.shape[0], device=device), Xty)

user_skill = theta[:n_users]
mapmod_difficulty = theta[n_users:]

sorted_users = sorted(zip(users, user_skill.tolist()), key=lambda x: -x[1])
print("Top users by skill:")
for u, skill in sorted_users[:5]:
    print(f"User {u} skill: {skill:.3f}")

# Sort maps descending by difficulty
sorted_maps = sorted(zip(maps_mods, mapmod_difficulty.tolist()), key=lambda x: -x[1])
print("\nTop maps by difficulty:")
for (map_id, mods), diff in sorted_maps[:5]:
    print(f"Map {map_id} with mods {bits_to_mods(mods)} difficulty: {diff:.3f}")
