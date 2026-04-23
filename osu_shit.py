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
    Extracts comprehensive beatmap information.
    Separates raw (base) attributes from mod-adjusted attributes.
    Uses rosu-pp's BeatmapAttributesBuilder for official mod adjustments.
    """
    with open(path, "r") as f:
        content = f.read()

    b = rosu.Beatmap(content=content)
    perf = rosu.Performance(mods=mods_str)
    res = perf.calculate(b)
    diff = res.difficulty

    # Official mod-adjusted attributes from rosu-pp
    attr_builder = rosu.BeatmapAttributesBuilder()
    attr_builder.set_map(b)
    attr_builder.set_mods(mods_str)
    attrs = attr_builder.build()

    map_obj = libosu.parse_beatmap(content)

    # 1. Raw (Base) Attributes
    raw = {
        "ar": b.ar,
        "cs": b.cs,
        "hp": b.hp,
        "od": b.od,
        "bpm": b.bpm,
        "slider_multiplier": b.slider_multiplier,
        "slider_tick_rate": b.slider_tick_rate,
    }

    # 2. Mod-adjusted Attributes
    adjusted = {
        "ar": attrs.ar,
        "cs": attrs.cs,
        "hp": attrs.hp,
        "od": attrs.od,
        "bpm": b.bpm * attrs.clock_rate,
    }

    # 3. Counts
    n_objects = b.n_objects
    counts = {
        "n_circles": b.n_circles,
        "n_sliders": b.n_sliders,
        "n_spinners": b.n_spinners,
        "n_objects": n_objects,
    }

    # 4. Skill attributes
    skill = {
        "stars": getattr(diff, "stars", 0.0),
        "aim": getattr(diff, "aim", 0.0),
        "speed": getattr(diff, "speed", 0.0),
        "flashlight": getattr(diff, "flashlight", 0.0),
        "reading": getattr(diff, "reading", 0.0),
        "slider_factor": getattr(diff, "slider_factor", 0.0),
        "speed_note_count": getattr(diff, "speed_note_count", 0.0),
        "stamina": getattr(diff, "stamina", 0.0),
        "rhythm": getattr(diff, "rhythm", 0.0),
        "color": getattr(diff, "color", 0.0),
    }
    skill = {k: (v if v is not None else 0.0) for k, v in skill.items()}

    # 5. Derived
    hit_length_base = 0
    if map_obj.hit_objects:
        first_start = map_obj.hit_objects[0].start_time
        last_start = map_obj.hit_objects[-1].start_time
        hit_length_base = (last_start - first_start) / 1000.0

    hit_length_adj = hit_length_base / attrs.clock_rate
    density = n_objects / hit_length_adj if hit_length_adj > 0 else 0.0

    derived = {
        "hit_length": hit_length_adj,
        "density": density,
        "circle_ratio": b.n_circles / n_objects if n_objects > 0 else 0.0,
        "slider_ratio": b.n_sliders / n_objects if n_objects > 0 else 0.0,
        "aim_minus_speed": skill["aim"] - skill["speed"],
        "log_bpm": math.log(adjusted["bpm"]) if adjusted["bpm"] > 0 else 0.0,
        "log_n_objects": math.log(n_objects) if n_objects > 0 else 0.0,
    }

    return {
        "raw": raw,
        "adjusted": adjusted,
        "counts": counts,
        "skill": skill,
        "derived": derived,
        "metadata": {
            "artist": map_obj.artist,
            "title": map_obj.title,
            "version": map_obj.difficulty_name,
        }
    }


if __name__ == "__main__":
    # example
    info = get_beatmap_info(
        "source-data/2026_01_01_osu_files/4030010.osu", "HDDT")
    import json
    print(json.dumps(info, indent=2))
