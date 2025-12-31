use anyhow::Result;
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use sqlx::{MySql, Pool, Row, mysql::MySqlPoolOptions};
use std::collections::HashMap;
use std::env;
use std::sync::Arc;

const CONCURRENT_BATCHES: usize = 50;
const USER_BATCH_SIZE: u32 = 100;

pub async fn run() -> Result<()> {
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    let pool = MySqlPoolOptions::new()
        .max_connections(100) // Ensure we have enough connections for concurrency
        .connect(&database_url)
        .await?;

    let pool = Arc::new(pool);
    println!("Connected to database.");

    // 1. Get User ID Range
    println!("Fetching user ID range...");
    let (min_id, max_id): (i32, i32) = sqlx::query_as("SELECT MIN(id), MAX(id) FROM statpp.user")
        .fetch_one(&*pool)
        .await
        .unwrap_or((0, 0)); // Handle case with no users gracefully if needed

    if max_id == 0 {
        println!("No users found.");
        return Ok(());
    }

    let total_users = (max_id - min_id + 1) as u64;
    println!(
        "Processing users with IDs {} to {} (approx {} users)...",
        min_id, max_id, total_users
    );

    let pb = ProgressBar::new(total_users);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len} {eta}")
            .unwrap(),
    );
    pb.set_message("Calculating Skills");

    // Create ranges
    let mut ranges = Vec::new();
    let mut current = min_id;
    while current <= max_id {
        let end = std::cmp::min(current + USER_BATCH_SIZE as i32 - 1, max_id);
        ranges.push((current, end));
        current += USER_BATCH_SIZE as i32;
    }

    let pb_shared = Arc::new(pb);

    // 2. Process concurrently
    stream::iter(ranges)
        .map(|(start, end)| {
            let pool = pool.clone();
            let pb = pb_shared.clone();
            async move {
                process_batch(&pool, start, end, pb).await?;
                // pb.inc((end - start + 1) as u64);
                Ok::<(), anyhow::Error>(())
            }
        })
        .buffer_unordered(CONCURRENT_BATCHES)
        .collect::<Vec<_>>()
        .await;

    pb_shared.finish();
    println!("Skill calculation complete.");

    Ok(())
}

async fn process_batch(
    pool: &Pool<MySql>,
    start_id: i32,
    end_id: i32,
    pb_shared: Arc<ProgressBar>,
) -> Result<()> {
    // 1. Fetch scores for this range of users
    // We select `osu_user` (which corresponds to user.id) and `score_pp`
    let rows = sqlx::query!(
        "SELECT osu_user, score_pp FROM statpp.score WHERE osu_user BETWEEN ? AND ?",
        start_id,
        end_id
    )
    .fetch_all(pool)
    .await?;

    pb_shared.println(format!("Fetched all rows ({})", rows.len()));

    if rows.is_empty() {
        return Ok(());
    }

    // 2. Group by user
    let mut user_scores: HashMap<i32, Vec<f64>> = HashMap::new();
    for row in rows {
        user_scores
            .entry(row.osu_user)
            .or_default()
            .push(row.score_pp);
    }

    // 3. Calculate weighted PP
    let mut updates = Vec::with_capacity(user_scores.len());
    for (user_id, mut pps) in user_scores {
        // Sort descending
        pps.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Weighted sum: pp * 0.95^index
        let total_pp: f64 = pps
            .iter()
            .enumerate()
            .map(|(i, pp)| pp * 0.95f64.powi(i as i32))
            .sum();

        updates.push((user_id, total_pp));
        pb_shared.inc(1);
    }

    if updates.is_empty() {
        return Ok(());
    }

    // 4. Bulk Update
    // UPDATE user SET total_pp = CASE id WHEN ? THEN ? ... END WHERE id IN (...)
    let mut query = String::from("UPDATE statpp.user SET total_pp = CASE id ");
    let mut params = Vec::new();
    let mut ids = Vec::new();

    for (uid, pp) in &updates {
        query.push_str("WHEN ? THEN ? ");
        params.push(*uid);
        // We push the float as strict value, or bind it.
        // Since we are building the query string for the CASE, we use bind placeholders.
    }

    query.push_str("END WHERE id IN (");
    for (i, (uid, _)) in updates.iter().enumerate() {
        if i > 0 {
            query.push_str(", ");
        }
        query.push('?');
        ids.push(*uid);
    }
    query.push(')');

    let mut qb = sqlx::query(&query);

    // Bind CASE WHEN parameters
    for (uid, pp) in &updates {
        qb = qb.bind(uid).bind(pp);
    }
    // Bind WHERE IN parameters
    for uid in ids {
        qb = qb.bind(uid);
    }

    qb.execute(pool).await?;

    Ok(())
}
