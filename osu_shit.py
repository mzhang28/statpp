import math
import rosu_pp_py as rosu
import libosu
from functools import lru_cache

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


@lru_cache(maxsize=10000)
def get_beatmap_info(path: str, mods_str: str = ""):
    """
    Extracts comprehensive beatmap information including raw, counts, skill, and derived attributes.
    """
    with open(path, "r") as f:
        content = f.read()

    b = rosu.Beatmap(content=content)
    perf = rosu.Performance(mods=mods_str)
    res = perf.calculate(b)
    diff = res.difficulty

    map_obj = libosu.parse_beatmap(content)

    # Manual hit_length calculation since it is not provided by libosu/rosu directly in a clean way
    # hit_length is usually (end_time of last object) - (start_time of first object)
    # converting to seconds for density calculation
    hit_length = 0
    if map_obj.hit_objects:
        first_start = map_obj.hit_objects[0].start_time
        last_start = map_obj.hit_objects[-1].start_time
        # Since we don't have easy end_time for sliders/spinners, last_start is a proxy
        # for start of last object.
        hit_length = (last_start - first_start) / 1000.0

    # Skill attributes
    # Standard: stars, aim, speed, flashlight
    # Extra: reading, slider_factor, speed_note_count
    skill_attrs = {
        "stars": getattr(diff, "stars", 0.0),
        "aim": getattr(diff, "aim", 0.0),
        "speed": getattr(diff, "speed", 0.0),
        "flashlight": getattr(diff, "flashlight", 0.0),
        "reading": getattr(diff, "reading", 0.0),
        "slider_factor": getattr(diff, "slider_factor", 0.0),
        "speed_note_count": getattr(diff, "speed_note_count", 0.0),
    }

    # Derived
    n_objects = b.n_objects
    density = n_objects / hit_length if hit_length > 0 else 0.0

    # Note: the user asked for length = n_objects / density_proxy or pull actual hit_length
    # I will provide hit_length as the primary proxy for length.

    derived = {
        "hit_length": hit_length,
        "density": density,
        "circle_ratio": b.n_circles / n_objects if n_objects > 0 else 0.0,
        "slider_ratio": b.n_sliders / n_objects if n_objects > 0 else 0.0,
        "aim_minus_speed": (skill_attrs["aim"] or 0.0) - (skill_attrs["speed"] or 0.0),
        "log_bpm": math.log(b.bpm) if b.bpm > 0 else 0.0,
        "log_n_objects": math.log(n_objects) if n_objects > 0 else 0.0,
    }

    # Mod one-hots
    mod_list = [m.strip() for m in mods_str.split(",") if m.strip()]
    one_hots = {
        "HD": 1 if "HD" in mod_list else 0,
        "DT": 1 if "DT" in mod_list or "NC" in mod_list else 0,
        "HR": 1 if "HR" in mod_list else 0,
        "EZ": 1 if "EZ" in mod_list else 0,
        "FL": 1 if "FL" in mod_list else 0,
        "NF": 1 if "NF" in mod_list else 0,
    }

    return {
        # Raw
        "ar": b.ar,
        "cs": b.cs,
        "hp": b.hp,
        "od": b.od,
        "bpm": b.bpm,
        "slider_multiplier": b.slider_multiplier,
        "slider_tick_rate": b.slider_tick_rate,
        # Counts
        "n_circles": b.n_circles,
        "n_sliders": b.n_sliders,
        "n_spinners": b.n_spinners,
        "n_objects": b.n_objects,
        # Skill
        **skill_attrs,
        # Derived
        **derived,
        # One-hots
        **one_hots,
        # Metadata
        "artist": map_obj.artist,
        "title": map_obj.title,
        "version": map_obj.difficulty_name,
    }


if __name__ == "__main__":
    # example
    info = get_beatmap_info(
        "source-data/2026_01_01_osu_files/4030010.osu", "HDDT")
    import json
    print(json.dumps(info, indent=2))
