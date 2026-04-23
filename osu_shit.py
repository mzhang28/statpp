import rosu_pp_py as rosu
import libosu

MOD_MULTIPLIERS = {
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


def parse_osu_mods(
    data,
):  # data is the json object from the osu beatmap table data column
    print("hello")


def get_beatmap_info(path: str):
    with open(path, "r") as f:
        bb = f.read()
    b = rosu.Beatmap(content=bb)
    p = rosu.Performance()
    pp = p.calculate(b)
    map = libosu.parse_beatmap(bb)
    print(dict(
        mode=b.mode,
        ar=b.ar,
        cs=b.cs,
        hp=b.hp,
        od=b.od,
        bpm=b.bpm,
        n_circles=b.n_circles,
        n_objects=b.n_objects,
        n_sliders=b.n_sliders,
        n_spinners=b.n_spinners,
        slider_tick_rate=b.slider_tick_rate,
        slider_multiplier=b.slider_multiplier,
        artist=map.artist,
        artist_unicode=map.artist_unicode,
        title=map.title,
        title_unicode=map.title_unicode,
        difficulty_name=map.difficulty_name,
        # pp_aim=pp.pp_aim,
        # pp_speed=pp.pp_speed,
        # pp_flashlight=pp.pp_flashlight,
        # pp_accuracy=pp.pp_accuracy,
        # effective_miss_count=pp.effective_miss_count,
        difficulty=pp.difficulty.reading,
    ))


if __name__ == "__main__":
    # example
    get_beatmap_info("source-data/2026_01_01_osu_files/4030010.osu")
