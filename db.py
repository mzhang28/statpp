import mysql.connector
from pony.orm import Database, Optional, PrimaryKey, Required, Set, composite_key

conn = mysql.connector.connect(host="localhost", user="osu", password="osu", database="osu")

db = Database()
db.bind(provider='mysql', user='root', password='root',
    host='localhost', database='statpp')

class User(db.Entity):
    id = PrimaryKey(int)
    username = Optional(str)
    total_pp = Required(float)
    scores = Set('Score')

class Beatmap(db.Entity):
    id = PrimaryKey(int)
    title = Optional(str)
    artist = Optional(str)
    diffname = Optional(str)
    mod_combos = Set('BeatmapMod')

class BeatmapMod(db.Entity):
    id = PrimaryKey(int, auto=True)
    beatmap = Required(Beatmap, index=True)
    mod = Required(str, index=True)
    difficulty = Required(float)
    scores = Set('Score')
    success_rate = Required(float, default=0.0)
    composite_key(beatmap, mod)

class Score(db.Entity):
    id = PrimaryKey(int)
    user = Required(User, index=True)
    beatmap_mod = Required(BeatmapMod, index=True)
    score = Required(int)
    score_pp = Required(float, index=True)
