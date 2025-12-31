use anyhow::Result;
use futures::TryStreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use sqlx::{mysql::MySqlPoolOptions, Pool, MySql};
use std::env;
use std::sync::Arc;
use tokio::sync::mpsc;

const UPDATE_BATCH_SIZE: usize = 2000;
const CHANNEL_BUFFER_SIZE: usize = 10000;

pub async fn run() -> Result<()> {
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    let pool = MySqlPoolOptions::new()
        .max_connections(20)
        .connect(&database_url)
        .await?;

    let pool = Arc::new(pool);
    println!("Connected to database.");

    // Count total scores for progress bar
    println!("Counting scores...");
    let count_result: Option<i64> = sqlx::query_scalar("SELECT COUNT(*) FROM statpp.score")
        .fetch_optional(&*pool)
        .await?;
    let total_scores = count_result.unwrap_or(0);
    println!("Total scores to process: {}", total_scores);

    let pb = ProgressBar::new(total_scores as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}")?
            .progress_chars("#>-"),
    );
    pb.set_message("Calculating PP...");

    let (tx, mut rx) = mpsc::channel::<(i32, f64)>(CHANNEL_BUFFER_SIZE);
    let pool_clone = pool.clone();
    let pb_clone = pb.clone();

    // Spawn Writer Task
    let writer_handle = tokio::spawn(async move {
        let mut batch = Vec::with_capacity(UPDATE_BATCH_SIZE);

        while let Some(item) = rx.recv().await {
            batch.push(item);
            if batch.len() >= UPDATE_BATCH_SIZE {
                if let Err(e) = flush_batch(&pool_clone, &batch).await {
                    eprintln!("Error flushing batch: {}", e);
                    return Err(e);
                }
                pb_clone.set_message(format!("Saved {} users", batch.len()));
                batch.clear();
            }
        }

        if !batch.is_empty() {
             flush_batch(&pool_clone, &batch).await?;
        }

        Ok::<(), anyhow::Error>(())
    });

    // Stream Reader
    // We sort by osu_user to process user-by-user
    let mut stream = sqlx::query!("SELECT osu_user, score_pp FROM statpp.score ORDER BY osu_user")
        .fetch(&*pool);

    let mut current_user_id: Option<i32> = None;
    let mut current_scores: Vec<f64> = Vec::new();

    while let Some(row) = stream.try_next().await? {
        let user_id = row.osu_user;
        let pp = row.score_pp; 

        if let Some(curr) = current_user_id {
            if curr != user_id {
                process_user(curr, &mut current_scores, &tx).await?;
                current_user_id = Some(user_id);
            }
        } else {
            current_user_id = Some(user_id);
        }

        current_scores.push(pp);
        pb.inc(1);
    }

    // Process last user
    if let Some(curr) = current_user_id {
        process_user(curr, &mut current_scores, &tx).await?;
    }

    drop(tx); // Close channel

    // Wait for writer
    writer_handle.await??;

    pb.finish_with_message("Skill calculation complete.");

    Ok(())
}

async fn process_user(
    user_id: i32, 
    scores: &mut Vec<f64>, 
    tx: &mpsc::Sender<(i32, f64)>
) -> Result<()> {
    if scores.is_empty() {
        return Ok(());
    }

    // Sort descending
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Weighted sum: pp * 0.95^index
    let total_pp: f64 = scores
        .iter()
        .enumerate()
        .map(|(i, pp)| pp * 0.95f64.powi(i as i32))
        .sum();

    // Send to writer
    tx.send((user_id, total_pp)).await?;

    scores.clear();
    Ok(())
}

async fn flush_batch(pool: &Pool<MySql>, batch: &[(i32, f64)]) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }

    let mut query = String::from("UPDATE statpp.user SET total_pp = CASE id ");
    let mut ids = Vec::with_capacity(batch.len());

    for _ in 0..batch.len() {
        query.push_str("WHEN ? THEN ? ");
    }

    query.push_str("END WHERE id IN (");
    for (i, (uid, _)) in batch.iter().enumerate() {
        if i > 0 {
            query.push_str(", ");
        }
        query.push('?');
        ids.push(*uid);
    }
    query.push(')');

    let mut qb = sqlx::query(&query);

    for (uid, pp) in batch {
        qb = qb.bind(uid).bind(pp);
    }
    for uid in ids {
        qb = qb.bind(uid);
    }

    qb.execute(pool).await?;

    Ok(())
}