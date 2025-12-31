use anyhow::Result;
use burn::backend::Wgpu;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Int, Tensor, backend::Backend},
};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use sea_orm::{Database, DatabaseConnection, DbBackend, FromQueryResult, Statement};
use std::env;

// --- Data Structures ---
// ... (ScoreItem and RawScoreRow omitted for brevity in thought, but I'll include them in the actual call if needed, or just replace the specific section)

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScoreItem {
    pub user_id: usize, // 0-based dense index
    pub map_id: usize,  // 0-based dense index
    pub score: f32,
}

// Custom struct to map the specific raw SQL output
#[derive(Debug, FromQueryResult)]
struct RawScoreRow {
    user_dense_id: i32,
    map_dense_id: i32,
    score_pp: f64,
}

// --- Efficient In-Memory Dataset ---

pub struct InMemoryDataset {
    items: Vec<ScoreItem>,
}

impl InMemoryDataset {
    // This function performs the heavy lifting
    pub async fn load(db: &DatabaseConnection) -> Result<Self> {
        println!("Starting bulk data load... this may take a moment.");

        // SQL MAGIC:
        // 1. We JOIN `score` and `user` to map sparse `osu_user` -> dense `user.id`.
        // 2. We select only the 3 columns we need to minimize traffic.
        // 3. We use `beatmap_mod` directly as it corresponds to `beatmapmod.id` (dense).
        let sql = r#"
            SELECT
                u.id as user_dense_id,
                s.beatmap_mod as map_dense_id,
                s.score_pp
            FROM score s
            INNER JOIN user u ON s.osu_user = u.osu_id
        "#;

        // Use streaming to handle 54M rows without exploding memory during the fetch
        let stmt = Statement::from_sql_and_values(DbBackend::MySql, sql, []);
        let mut stream = RawScoreRow::find_by_statement(stmt).stream(db).await?;

        let mut items = Vec::with_capacity(55_000_000); // Pre-allocate to prevent re-sizing

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {pos} items loaded ({per_sec})")?,
        );

        while let Some(row) = stream.next().await {
            let row = row?;

            // CONVERSION:
            // DB IDs are 1-based. Embedding indices must be 0-based.
            // Subtract 1 safely.
            items.push(ScoreItem {
                user_id: (row.user_dense_id as usize).saturating_sub(1),
                map_id: (row.map_dense_id as usize).saturating_sub(1),
                score: row.score_pp as f32,
            });

            pb.inc(1);
        }

        pb.finish_with_message("Load complete");
        println!("Load complete. Total items: {}", items.len());
        Ok(Self { items })
    }
}

impl Dataset<ScoreItem> for InMemoryDataset {
    fn get(&self, index: usize) -> Option<ScoreItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

// --- Batcher (Standard Burn Boilerplate) ---

#[derive(Clone, Debug)]
pub struct ScoreBatch<B: Backend> {
    pub user_ids: Tensor<B, 1, Int>,
    pub map_ids: Tensor<B, 1, Int>,
    pub targets: Tensor<B, 1>,
}

#[derive(Clone)]
pub struct ScoreBatcher<B: Backend> {
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> ScoreBatcher<B> {
    pub fn new() -> Self {
        Self {
            _b: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, ScoreItem, ScoreBatch<B>> for ScoreBatcher<B> {
    fn batch(&self, items: Vec<ScoreItem>, device: &B::Device) -> ScoreBatch<B> {
        let user_ids: Vec<i32> = items.iter().map(|item| item.user_id as i32).collect();
        let map_ids: Vec<i32> = items.iter().map(|item| item.map_id as i32).collect();
        let targets: Vec<f32> = items.iter().map(|item| item.score).collect();

        let user_ids = Tensor::<B, 1, Int>::from_data(user_ids.as_slice(), device);
        let map_ids = Tensor::<B, 1, Int>::from_data(map_ids.as_slice(), device);
        let targets = Tensor::<B, 1>::from_data(targets.as_slice(), device);

        ScoreBatch {
            user_ids,
            map_ids,
            targets,
        }
    }
}

// --- Main Execution ---

pub async fn run() -> Result<()> {
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    println!("Connecting to database...");
    // Ensure you have `sqlx` features enabled in Cargo.toml for SeaORM to use raw SQL efficiently
    let db = Database::connect(&database_url).await?;

    // 1. Bulk Load (RAM Optimization)
    let dataset = InMemoryDataset::load(&db).await?;

    // 2. Setup Burn
    let batcher = ScoreBatcher::<Wgpu>::new();
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(2048) // Larger batch size is safe now
        .shuffle(42)
        .num_workers(4)
        .build(dataset);

    println!("Starting iteration...");

    // Test loop
    for (i, batch) in dataloader.iter().enumerate() {
        if i >= 5 {
            break;
        }
        println!(
            "Batch {}: Users shape {:?}, Targets shape {:?}",
            i,
            batch.user_ids.shape(),
            batch.targets.shape()
        );
    }

    Ok(())
}
