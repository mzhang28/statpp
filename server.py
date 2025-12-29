from fastapi import FastAPI
from pony.orm import db_session, select
from db import db, Beatmap, BeatmapMod

db.generate_mapping()

app = FastAPI()

@app.get("/stats/beatmap_difficulty_distribution")
def beatmap_difficulty_distribution():
    with db_session:
        all = select((t.id, t.difficulty, t.beatmap.artist, t.beatmap.title, t.beatmap.diffname, t.mod) for t in BeatmapMod if t.difficulty != 5)[:]
    return list(all)
