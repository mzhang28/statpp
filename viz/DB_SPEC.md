# Statpp Database Specification

This document defines the SQLite database schema expected by the `statpp-viewer` utility.

## Tables

### 1. `meta`
| Column | Type | Description |
| :--- | :--- | :--- |
| `key` | TEXT | Metadata key (e.g., `'dimensions'`) |
| `value` | TEXT | Metadata value |

---

### 2. `players`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER | Primary Key |
| `username` | TEXT | Player display name |
| `metadata` | JSON | Extended data (must contain `ratings` array) |

---

### 3. `beatmaps`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER | Primary Key |
| `artist` | TEXT | Song artist |
| `title` | TEXT | Song title |
| `version` | TEXT | Difficulty name |
| `metadata` | JSON | Extended data (must contain `difficulties` array) |

---

### 4. `scores`
| Column | Type | Description |
| :--- | :--- | :--- |
| `player_id` | INTEGER | FK to `players.id` |
| `beatmap_id` | INTEGER | FK to `beatmaps.id` |
| `mod` | TEXT | Mod string |
| `metadata` | JSON | Extended data (must contain `scores` array and `accuracy`) |

## Indexing
The viewer dynamically creates indexes on `json_extract(metadata, '$.ratings[i]')`, etc., for all $n$ dimensions.
