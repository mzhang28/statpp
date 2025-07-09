# statpp

Overview:

- Build an unsupervised performance system for osu! that jointly learns player skill (pp) and beatmap difficulty.
  - The algorithm essentially computes map changes from player scores as well as the skill level of those players, and then the maps then influence the player's rating.
  - THIS ALGORITHM SHOULD CONVERGE.
- Ensure difficulty ratings reflect actual challenge:
  - If a player scores well on a hard map (diff > rating), then the player should gain score. The map should also slightly lose difficulty.
  - If a lot of people do well on a map, its rating should go down.
  - If a lot of people do poorly on a map, but top players do better on it, then the top players should push it up (the people who are bad don't count towards it)

Implementation:

- All ratings, difficulties, scores should be normalized to [0-1]
