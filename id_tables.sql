-- Table for mapping generic osu! user IDs to dense [0..N] indices
CREATE TABLE IF NOT EXISTS player_embeddings (
    dense_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    osu_user_id INT(10) UNSIGNED NOT NULL, -- Matching your statpp_scores type
    UNIQUE KEY (osu_user_id)               -- CRITICAL: This enables INSERT IGNORE to work
);

-- Table for mapping (Beatmap + Mods) to dense [0..M] indices
CREATE TABLE IF NOT EXISTS beatmap_embeddings (
    dense_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    beatmap_id MEDIUMINT(9) NOT NULL,      -- Matching your statpp_scores type
    mods VARCHAR(255) NOT NULL,            -- Matching your statpp_scores type
    UNIQUE KEY (beatmap_id, mods)          -- CRITICAL: Unique constraint on the COMBO
);
