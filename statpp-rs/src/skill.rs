use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use sea_orm::{ConnectionTrait, Database, Statement, FromQueryResult};
use std::env;

#[derive(Debug, FromQueryResult)]
struct MinMax {
    min_id: Option<i32>,
    max_id: Option<i32>,
}

pub async fn run() -> Result<()> {
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let db = Database::connect(&database_url).await?;

    println!("Connected to database.");

    // 1. Get User ID Range
    println!("Fetching user ID range...");
    let result = MinMax::find_by_statement(Statement::from_string(
        db.get_database_backend(),
        "SELECT MIN(id) as min_id, MAX(id) as max_id FROM user".to_string(),
    ))
    .one(&db)
    .await?
    .ok_or_else(|| anyhow::anyhow!("No users found"))?;

    let min_id = result.min_id.unwrap_or(0);
    let max_id = result.max_id.unwrap_or(0);
    let total_range = max_id - min_id + 1;

    if total_range <= 0 {
        println!("No users to process.");
        return Ok(());
    }

    println!("Processing users with IDs {} to {}...", min_id, max_id);

    // 2. Process in Batches
    let pb = ProgressBar::new(total_range as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len} {eta}")
            .unwrap(),
    );
    pb.set_message("Calculating Skills");

    const BATCH_SIZE: i32 = 10000;
    let mut current_min = min_id;

    while current_min <= max_id {
        let current_max = std::cmp::min(current_min + BATCH_SIZE - 1, max_id);
        
        // We use a raw SQL query to perform the calculation entirely in the database.
        // This avoids fetching millions of scores into the application memory.
        // We calculate the weighted PP: sum(score_pp * 0.95^rank)
        let sql = format!(
            r#"
            UPDATE user u
            JOIN (
                SELECT 
                    osu_user, 
                    SUM(score_pp * POW(0.95, rn)) as total_pp
                FROM (
                    SELECT 
                        osu_user, 
                        score_pp, 
                        ROW_NUMBER() OVER (PARTITION BY osu_user ORDER BY score_pp DESC) - 1 as rn
                    FROM score
                    WHERE osu_user BETWEEN {} AND {}
                ) ranked
                GROUP BY osu_user
            ) stats ON u.id = stats.osu_user
            SET u.total_pp = stats.total_pp;
            "#,
            current_min, current_max
        );

        db.execute_unprepared(&sql).await?;
        
        pb.inc((current_max - current_min + 1) as u64);
        current_min += BATCH_SIZE;
    }

    pb.finish();
    println!("Skill calculation complete.");

    Ok(())
}
