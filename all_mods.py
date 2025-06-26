import mysql.connector
import json
conn = mysql.connector.connect(
    host="localhost", user="osu", password="osu", database="osu"
)

cur = conn.cursor()
cur.execute("select json_extract(data, '$.mods') from scores")
all_mods = set()
with open("mods.txt", "w") as f:
    while True:
        rows = cur.fetchmany(size=1000)
        for row in rows:
            mods = json.loads(row[0])
            for mod in mods:
                name = mod["acronym"]
                if name not in all_mods:
                    print(name, file=f)
                    f.flush()
                    all_mods.add(name)
