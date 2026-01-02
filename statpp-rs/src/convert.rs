use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use sqlx::{MySql, Pool, Row, mysql::MySqlPoolOptions};
use std::collections::{HashMap, HashSet};
use std::env;

// --- JSON Structs for parsing Score Data ---
#[derive(Debug, Deserialize)]
struct ModItem {
    acronym: String,
}

#[derive(Debug, Deserialize)]
struct ScoreMetadata {
    #[serde(default)]
    mods: Vec<ModItem>,
}

struct IndexDef {
    name: String,
    columns: Vec<String>,
    unique: bool,
}

const BATCH_SIZE: i64 = 10_000;

pub async fn run() -> Result<()> {
    dotenvy::dotenv().ok();

    // Assumption: Both schemas are on the same DB server, accessible via one URL.
    // If not, create two pools. We assume tables are qualified or we switch DBs.
    // Ideally, set DATABASE_URL=mysql://user:pass@localhost:3306/osu
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    let pool = MySqlPoolOptions::new()
        .max_connections(20)
        .connect(&database_url)
        .await?;

    println!("Connected to database.");

    // 0. Truncate Tables
    truncate_tables(&pool).await?;

    // Disable keys for performance
    toggle_keys(&pool, false).await?;

    // 1. Migrate Users
    // Returns a map of osu_user_id -> target_table_id
    let user_map = migrate_users(&pool).await?;

    // 2. Migrate Beatmaps
    migrate_beatmaps(&pool).await?;

    // 3. Migrate Scores
    // Remove indexes for performance
    let indexes = drop_indexes(&pool).await?;
    migrate_scores(&pool, &user_map).await?;
    // Restore indexes
    restore_indexes(&pool, indexes).await?;

    // Enable keys back
    toggle_keys(&pool, true).await?;

    Ok(())
}

async fn truncate_tables(pool: &Pool<MySql>) -> Result<()> {
    println!("Truncating tables...");
    // Acquire a single connection to ensure SET FOREIGN_KEY_CHECKS applies to the TRUNCATEs
    let mut conn = pool.acquire().await?;

    sqlx::query("SET FOREIGN_KEY_CHECKS = 0")
        .execute(&mut *conn)
        .await?;

    sqlx::query("TRUNCATE TABLE statpp.score")
        .execute(&mut *conn)
        .await?;
    sqlx::query("TRUNCATE TABLE statpp.beatmapmod")
        .execute(&mut *conn)
        .await?;
    sqlx::query("TRUNCATE TABLE statpp.beatmap")
        .execute(&mut *conn)
        .await?;
    sqlx::query("TRUNCATE TABLE statpp.user")
        .execute(&mut *conn)
        .await?;

    sqlx::query("SET FOREIGN_KEY_CHECKS = 1")
        .execute(&mut *conn)
        .await?;

    println!("Tables truncated.");
    Ok(())
}

async fn toggle_keys(pool: &Pool<MySql>, enable: bool) -> Result<()> {
    let state = if enable { "ENABLE" } else { "DISABLE" };
    println!("{} keys for all tables...", state);

    let tables = [
        "statpp.user",
        "statpp.beatmap",
        "statpp.beatmapmod",
        "statpp.score",
    ];

    for table in tables {
        let query = format!("ALTER TABLE {} {} KEYS", table, state);
        sqlx::query(&query).execute(pool).await?;
    }

    Ok(())
}

async fn drop_indexes(pool: &Pool<MySql>) -> Result<Vec<IndexDef>> {
    println!("Fetching and dropping indexes on statpp.score...");

    // Fetch index definitions
    let rows = sqlx::query(
        r#"
        SELECT INDEX_NAME, COLUMN_NAME, NON_UNIQUE, SEQ_IN_INDEX
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = 'statpp' AND TABLE_NAME = 'score'
        ORDER BY INDEX_NAME, SEQ_IN_INDEX
        "#,
    )
    .fetch_all(pool)
    .await?;

    let mut indices: HashMap<String, IndexDef> = HashMap::new();

    for row in rows {
        let name: String = row.get("INDEX_NAME");
        // Don't drop Primary Key
        if name == "PRIMARY" {
            continue;
        }

        let column: String = row.get("COLUMN_NAME");
        let non_unique: i64 = row.get("NON_UNIQUE");

        let entry = indices.entry(name.clone()).or_insert(IndexDef {
            name,
            columns: Vec::new(),
            unique: non_unique == 0,
        });
        entry.columns.push(column);
    }

    let mut result: Vec<IndexDef> = indices.into_values().collect();
    // Sort for deterministic output
    result.sort_by(|a, b| a.name.cmp(&b.name));

    // Drop them
    for idx in &result {
        let query = format!("DROP INDEX `{}` ON statpp.score", idx.name);
        sqlx::query(&query).execute(pool).await?;
        println!("Dropped index: {}", idx.name);
    }

    Ok(result)
}

async fn restore_indexes(pool: &Pool<MySql>, indices: Vec<IndexDef>) -> Result<()> {
    println!("Restoring indexes on statpp.score...");
    for idx in indices {
        let cols = idx
            .columns
            .iter()
            .map(|c| format!("`{}`", c))
            .collect::<Vec<_>>()
            .join(", ");
        let unique_str = if idx.unique { "UNIQUE" } else { "" };
        let query = format!(
            "CREATE {} INDEX `{}` ON statpp.score ({})",
            unique_str, idx.name, cols
        );

        println!("Restoring index: {} on ({})", idx.name, cols);
        sqlx::query(&query).execute(pool).await?;
    }
    Ok(())
}

async fn migrate_users(pool: &Pool<MySql>) -> Result<HashMap<u32, i32>, sqlx::Error> {
    println!("Migrating users...");

    // Note: The provided first schema does NOT have a table with 'username'.
    // The Python code references 'sample_users'. We use that here.
    // If that table doesn't exist, this query needs to be adjusted to your actual user source.
    let rows = sqlx::query!("SELECT user_id FROM osu.osu_user_stats")
        .fetch_all(pool)
        .await?;

    if !rows.is_empty() {
        let pb = ProgressBar::new(rows.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} {bar:40} {pos}/{len}")
                .unwrap(),
        );
        pb.set_message("Users (Initial)");

        // We insert in chunks to prevent packet size errors
        for chunk in rows.chunks(5000) {
            let mut query_builder =
                sqlx::QueryBuilder::new("INSERT INTO statpp.user (osu_id, username, total_pp) ");

            query_builder.push_values(chunk, |mut b, user| {
                b.push_bind(user.user_id)
                    .push_bind("<unknown>") // or user.username if available
                    .push_bind(0.0);
            });

            query_builder.push(" ON DUPLICATE KEY UPDATE username = VALUES(username)");

            query_builder.build().execute(pool).await?;
            pb.inc(chunk.len() as u64);
        }
        pb.finish();
    } else {
        println!("No users found in sample_users. Proceeding to score-based user discovery.");
    }

    // Ensure all users referenced in scores exist
    println!("Ensuring all users from scores exist...");
    // Insert any users found in scores but missing from user table
    // statpp.user needs (osu_id, username, total_pp)
    sqlx::query(
        r#"
        INSERT IGNORE INTO statpp.user (osu_id, username, total_pp)
        SELECT DISTINCT user_id, '<unknown>', 0.0
        FROM osu.scores
        "#,
    )
    .execute(pool)
    .await?;

    // Build the ID map for Foreign Key lookups later
    println!("Building user ID map...");
    let mapping_rows = sqlx::query!("SELECT id, osu_id FROM statpp.user")
        .fetch_all(pool)
        .await?;
    let map: HashMap<u32, i32> = mapping_rows
        .into_iter()
        .map(|r| (r.osu_id as u32, r.id))
        .collect();

    println!("User migration done. Mapped {} users.", map.len());
    Ok(map)
}

async fn migrate_beatmaps(pool: &Pool<MySql>) -> Result<(), sqlx::Error> {
    println!("Migrating beatmaps...");

    let rows = sqlx::query!(
        r#"
        SELECT CAST (b.beatmap_id AS unsigned) AS beatmap_id, s.artist, s.title, b.version
        FROM osu.osu_beatmaps b
        JOIN osu.osu_beatmapsets s ON b.beatmapset_id = s.beatmapset_id
        "#
    )
    .fetch_all(pool)
    .await?;

    let pb = ProgressBar::new(rows.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Beatmaps");

    // Insert in batches
    let mut chunks = rows.chunks(5000);
    while let Some(chunk) = chunks.next() {
        let mut query_builder = sqlx::QueryBuilder::new(
            "INSERT INTO statpp.beatmap (id, artist, title, diffname, osu_difficulty) ",
        );
        query_builder.push_values(chunk, |mut b, item| {
            b.push_bind(item.beatmap_id)
                .push_bind(&item.artist)
                .push_bind(&item.title)
                .push_bind(&item.version)
                .push_bind(0.0); // osu_difficulty required by schema, defaulting to 0.0
        });

        // Handle duplicates (update or ignore)
        query_builder.push(" ON DUPLICATE KEY UPDATE title = VALUES(title)");
        query_builder.build().execute(pool).await?;
        pb.inc(chunk.len() as u64);
    }

    pb.finish();
    println!("Beatmap migration done.");
    Ok(())
}

async fn migrate_scores(pool: &Pool<MySql>, user_map: &HashMap<u32, i32>) -> Result<()> {
    println!("Migrating scores...");

    // Cache for (beatmap_id, mod_string) -> beatmapmod.id
    let mut beatmap_mod_cache: HashMap<(u32, String), i32> = HashMap::new();

    // Pre-populate cache if table isn't empty
    let existing_mods = sqlx::query!("SELECT id, beatmap, `mod` FROM statpp.beatmapmod")
        .fetch_all(pool)
        .await?;
    for r in existing_mods {
        beatmap_mod_cache.insert((r.beatmap as u32, r.r#mod), r.id);
    }

    // Get max ID to resume or start
    let count_row = sqlx::query!("SELECT count(*) as c FROM osu.scores")
        .fetch_one(pool)
        .await?;
    let total_scores = count_row.c;

    println!("Total scores to process: {}", total_scores);

    let pb = ProgressBar::new(total_scores as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len} {eta}")
            .unwrap(),
    );
    pb.set_message("Scores");

    let mut last_seen_id = 0;

    loop {
        // 1. Fetch Batch
        let scores = sqlx::query!("SELECT id, user_id, CAST (beatmap_id AS unsigned) AS beatmap_id, total_score, pp, data FROM osu.scores WHERE id > ? ORDER BY id LIMIT ?", last_seen_id, BATCH_SIZE)
        .fetch_all(pool)
        .await?;

        if scores.is_empty() {
            break;
        }

        // 2. Process Metadata & Prepare Mods
        // We need to identify which beatmap_mods are missing from the DB/Cache for this batch
        let mut batch_data = Vec::with_capacity(scores.len());
        let mut mods_to_insert: HashSet<(u32, String)> = HashSet::new();

        for score in &scores {
            last_seen_id = score.id; // Update cursor

            // Parse JSON
            let mod_str = match serde_json::from_slice::<ScoreMetadata>(&score.data) {
                Ok(meta) => {
                    let mut list: Vec<String> = meta
                        .mods
                        .into_iter()
                        .map(|m| m.acronym)
                        .filter(|a| a != "NF") // Filter out NoFail
                        .collect();
                    list.sort();
                    list.join("|")
                }
                Err(_) => String::new(), // Fallback empty mods
            };

            // Check if we have this mapping
            if !beatmap_mod_cache.contains_key(&(score.beatmap_id, mod_str.clone())) {
                mods_to_insert.insert((score.beatmap_id, mod_str.clone()));
            }

            batch_data.push((score, mod_str));
        }

        // 3. Insert Missing Mods
        if !mods_to_insert.is_empty() {
            let mut mod_qb = sqlx::QueryBuilder::new(
                "INSERT IGNORE INTO statpp.beatmapmod (beatmap, `mod`, difficulty, success_rate) ",
            );
            mod_qb.push_values(mods_to_insert.iter(), |mut b, (bid, mstr)| {
                b.push_bind(bid)
                    .push_bind(mstr)
                    .push_bind(5.0) // Default per Python logic
                    .push_bind(0.0); // success_rate default
            });
            mod_qb.build().execute(pool).await?;

            // Update Cache: Fetch IDs of what we just inserted (or what existed)
            // It's slightly inefficient to query all matching, but easier than tracking insert IDs individually in MySQL
            let beatmap_ids: Vec<u32> = mods_to_insert.iter().map(|(b, _)| *b).collect();
            // We can't bind a slice easily in a simple query for a tuple IN clause,
            // so we just query specific beatmaps involved in this batch.
            let query = format!(
                "SELECT id, beatmap, `mod` FROM statpp.beatmapmod WHERE beatmap IN ({})",
                beatmap_ids
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            // This could return a lot of rows if a map has many mods, but constrained to the current batch's maps
            let new_mod_rows = sqlx::query(&query).fetch_all(pool).await?;
            for r in new_mod_rows {
                // The column name is `mod`, so we use r#mod or get("mod")
                let m_str: String = r.get("mod");
                let b_id: i32 = r.get("beatmap");
                let id: i32 = r.get("id");
                beatmap_mod_cache.insert((b_id as u32, m_str), id);
            }
        }

        // 4. Insert Scores
        // We filter out scores where we can't find a User ID (user doesn't exist in target)
        // or can't find beatmap mod (shouldn't happen due to step 3).
        let mut scores_to_insert = Vec::new();

        for (score, mod_str) in batch_data {
            let target_user_id = match user_map.get(&score.user_id) {
                Some(uid) => *uid,
                None => unreachable!("somehow missed uid {}", score.user_id), // Skip score if user not in target DB
            };

            let beatmap_mod_id = match beatmap_mod_cache.get(&(score.beatmap_id, mod_str)) {
                Some(bmid) => *bmid,
                None => unreachable!("somehow misseed bmid {}", score.beatmap_id), // Should not happen
            };

            let pp_val = score.pp.unwrap_or(0.0);

            scores_to_insert.push((
                score.id,
                target_user_id,
                beatmap_mod_id,
                score.total_score,
                pp_val,
            ));
        }

        if !scores_to_insert.is_empty() {
            let mut score_qb = sqlx::QueryBuilder::new(
                "INSERT IGNORE INTO statpp.score (osu_score_id, osu_user, beatmap_mod, score, score_pp) ",
            );
            score_qb.push_values(scores_to_insert, |mut b, (sid, uid, bmid, sc, pp)| {
                b.push_bind(sid)
                    .push_bind(uid)
                    .push_bind(bmid)
                    .push_bind(sc)
                    .push_bind(pp); // score_pp from source
            });
            score_qb.build().execute(pool).await?;
        }

        pb.inc(scores.len() as u64);
    }

    pb.finish();
    println!("Score migration complete.");
    Ok(())
}
