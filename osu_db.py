from pony.orm import Database, PrimaryKey, Required, db_session

db = Database()


class Score(db.Entity):
    _table_ = "scores"
    id = PrimaryKey(int, auto=True)
    user_id = Required(int)
    beatmap_id = Required(int)
    ruleset_id = Required(int)
    accuracy = Required(float)
    total_score = Required(int)
    data = Required(bytes)


class Beatmap(db.Entity):
    _table_ = "osu_beatmaps"
    beatmap_id = PrimaryKey(int)
    beatmapset_id = Required(int)
    version = Required(str)


class Beatmapset(db.Entity):
    _table_ = "osu_beatmapsets"
    beatmapset_id = PrimaryKey(int)
    artist = Required(str)
    title = Required(str)


db.bind(provider="mysql", host="127.0.0.1", user="ro", passwd="ro", db="osu")
db.generate_mapping(create_tables=False)


@db_session
def main():
    print(db.get("select count(*) from scores;"))


if __name__ == "__main__":
    main()
