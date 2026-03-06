import numpy as np
import pandas as pd

OUTPUT = "./training_data"

data = np.load(f"{OUTPUT}/pmf_hddt.npz", allow_pickle=True)
pu, qi = data["pu"], data["qi"]
bu, bi, mu = data["bu"], data["bi"], float(data["mu"])
uids, mids = data["uids"], data["mids"]
n_users = len(uids)

# synthetic map: sum of v_j weighted by ||v_j||
norms = np.linalg.norm(qi, axis=1)
v_bar = (qi * norms[:, None]).sum(axis=0)

# endogenous skill = u_i . v_bar
skill = pu @ v_bar

# also compute bias-inclusive version
skill_with_bias = skill + bu

ranking = np.argsort(-skill)
ranking_wb = np.argsort(-skill_with_bias)

df = pd.read_parquet(f"{OUTPUT}/scores_indexed.parquet")
df = df[df["mods"] == "DT,HD"]
user_lookup = df[["user_idx", "user_id"]
                 ].drop_duplicates().set_index("user_idx")
uid_to_local = {uid: i for i, uid in enumerate(uids)}


def uid_str(li):
    return str(int(user_lookup.loc[uids[li], "user_id"])) if uids[li] in user_lookup.index else "?"


print("top 20 by endogenous skill (factor only):")
for r, li in enumerate(ranking[:20]):
    print(
        f"  #{r+1} user_id={uid_str(li)} skill={skill[li]:.4f} bias={bu[li]:.4f}")

print("\ntop 20 by endogenous skill (factor + bias):")
for r, li in enumerate(ranking_wb[:20]):
    print(
        f"  #{r+1} user_id={uid_str(li)} skill={skill_with_bias[li]:.4f} (factor={skill[li]:.4f} bias={bu[li]:.4f})")

KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}
print("\nknown players:")
for uid, name in KNOWN.items():
    matches = user_lookup[user_lookup["user_id"] == uid]
    if matches.empty or matches.index[0] not in uid_to_local:
        print(f"  {name}: not in pool")
        continue
    li = uid_to_local[matches.index[0]]
    r_f = int((skill > skill[li]).sum()) + 1
    r_fb = int((skill_with_bias > skill_with_bias[li]).sum()) + 1
    print(
        f"  {name}: factor #{r_f} | factor+bias #{r_fb} | skill={skill[li]:.4f} bias={bu[li]:.4f}")

# correlation between the two rankings
print(f"\ncorr(factor, bias) = {np.corrcoef(skill, bu)[0, 1]:.4f}")
print(
    f"synthetic map direction (v_bar normalized): {v_bar / np.linalg.norm(v_bar)}")
print(f"synthetic map norm: {np.linalg.norm(v_bar):.4f}")
