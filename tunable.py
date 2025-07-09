from math import exp, log

def expected_performance(player_rating, beatmap_difficulty):
    """
    Purpose:
    Given a player rating [0-1] and beatmap difficulty [0-1]
    (these scales should match, i.e if player == beatmap, then this is JUST the right map at their skill)
    Returns a value (0-1] as their expected output. This will be compared with their actual score (normalized to [0-1]).
    """
    # I think basically if the difficulty is less than ur player rating, ur expected to get a really high score (i'll just say 1M for now)
    # Otherwise, it should gradually drop off but still pretty fast
    # check out this fucked function i cooked up
    P = player_rating
    D = beatmap_difficulty
    return 1.0 / (1.0 + exp(-1.0 * (10.0 / (1.0 + 1e-8 - P)) * (P - D) - 5.0))

def compute_beatmap_individual_score_influence(accum, bm, score):
    """
    Purpose: Given a beatmap/mod combo and a score, determine some metrics that will be used by collect_beatmap_score_influences below.
    """
    # ok the general format of this should be
    # 1. Find the expected score
    # 2. Find the actual score
    # 3. Make sure they're on the same scale
    # 4. Try to compute some sort of "adjustment factor", based on if it's too easy or hard
    # Notes tho:
    # - if a player is better than the map, their expected score should be 1 and their actual score might also be 1. we should go for a NEGATIVE factor
    # - if the map is better than the player, but they did WELL, then player should go up and map should go SLIGHTLY down
    # - if the map is better than the player, but they did poor, the player should go down and the map should go SLIGHTLY up
    # takeaway here is that the map should change a LOT less than the player

    # Normalize player rating and beatmap difficulty to [0,1]
    player_rating = score.user.total_pp / 300  # or scale as needed, max_pp=300
    beatmap_diff_norm = bm.difficulty / 8       # or scale max_diff=8

    expected = expected_performance(player_rating, beatmap_diff_norm)
    actual = min(max(score.score / 1_000_000, 0), 1)  # clamp actual score [0,1]

    # Reward more on low-success-rate maps: boost adjustment by inverse of success_rate
    # We will get success_rate later in collect, so here just accumulate logs.

    # Take logs to combine multiplicatively and stabilize differences
    # Clamp expected and actual away from zero to avoid log(0)
    eps = 1e-8
    log_expected = log(max(expected, eps))
    log_actual = log(max(actual, eps))

    player_weight = 1.0 / (1 + exp(-8.0 * (player_rating - 0.7)))
    success_rate_alpha = 20.0
    beatmap_weight = 1.0 / (1.0 + success_rate_alpha * bm.success_rate)
    combined_weight = player_weight * beatmap_weight

    accum['count'] += combined_weight
    accum['sum_log_expected'] += combined_weight * log_expected
    accum['sum_log_actual'] += combined_weight * log_actual

def collect_beatmap_score_influences(accum, bm):
    """
    Purpose: Given a beatmap and a score, aggregate the previous influences into a final adjustment factor
    """

    if accum['count'] == 0:
        return 1.0  # no data => no change

    avg_log_expected = accum['sum_log_expected'] / accum['count']
    avg_log_actual = accum['sum_log_actual'] / accum['count']

    # Compute adjustment log (clamped)
    adjustment_log = avg_log_actual - avg_log_expected
    adjustment_log = max(min(adjustment_log, 0.02), -0.2)
    adjustment_ratio = exp(adjustment_log)

    # Bonus multiplier for low success_rate maps
    # The lower the success rate, the more the map should be pushed harder
    # Use bm.success_rate assumed in [0,1], add a small floor to avoid div by zero
    success_rate = max(bm.success_rate, 0.01)

    # Inverse success_rate weighting: maps with low success_rate get boosted adjustment
    adjustment_ratio *= (1.0 + (1.0 + 1e-8 - success_rate)) * 0.5  # scale factor to limit impact

    return adjustment_ratio

def score_to_pp(actual_score_norm, player_rating, beatmap_difficulty):
    """
    Purpose: Given a player's actual score and the map's difficulty, compute their resulting pp
    """
    # Prevent negative or NaN effects
    actual_score_norm = max(min(actual_score_norm, 1.0), 0.0)
    player_rating = max(min(player_rating, 1.0), 0.0)
    beatmap_difficulty = max(min(beatmap_difficulty, 1.0), 0.0)

    # Base pp from map difficulty and score power curve
    base_pp = beatmap_difficulty * actual_score_norm

    # Difficulty gap adjustment
    diff_gap = player_rating - beatmap_difficulty

    if diff_gap > 0: # Map is harder than player → reward more
        multiplier = 1.0 + 2.0 * diff_gap
    else: # Map is easier → penalize
        multiplier = max(0.3, 1.0 + 5.0 * diff_gap)

    pp = base_pp * multiplier

    # Clamp final pp to [0, beatmap_difficulty]
    pp = max(0.0, pp)

    return pp
