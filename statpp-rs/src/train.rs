use anyhow::Result;
use burn::backend::Wgpu;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Int, Tensor, backend::Backend},
};
use sea_orm::{Database, DatabaseConnection, EntityTrait, QueryOrder};
use std::env;
use std::sync::Arc;
use tokio::runtime::Runtime;

// Access the entity module from the crate root
use crate::entity::score;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ScoreItem {
    pub user_id: usize,
    pub map_id: usize,
    pub score: f32,
}

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

// --- DB Dataset ---

struct DbDataset {
    len: usize,
    db: DatabaseConnection,
    rt: Arc<Runtime>,
}

impl DbDataset {
    pub fn new(db: DatabaseConnection, len: usize) -> Self {
        // Create a dedicated runtime for DB operations within the dataset
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        Self { len, db, rt }
    }
}

impl Dataset<ScoreItem> for DbDataset {
    fn get(&self, index: usize) -> Option<ScoreItem> {
        if index >= self.len {
            return None;
        }
        // Assuming IDs are 1-based contiguous
        let id = (index + 1) as i32;

        // Perform blocking DB fetch
        let result = self
            .rt
            .block_on(async { score::Entity::find_by_id(id).one(&self.db).await });

        match result {
            Ok(Some(model)) => Some(ScoreItem {
                // Adjusting 1-based IDs to 0-based for embeddings
                // Ensure no underflow if ID is 0 or negative (though schema implies +ve)
                user_id: (model.osu_user.max(1) - 1) as usize,
                map_id: (model.beatmap_mod.max(1) - 1) as usize,
                score: model.score_pp as f32,
            }),
            _ => None, // Error or Not Found (Gap in ID sequence)
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

pub async fn run() -> Result<()> {
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    println!("Connecting to database...");
    let db = Database::connect(&database_url).await?;
    println!("Connected.");

    println!("Fetching Max Score ID...");
    // Fetch max ID to determine dataset length
    // We order by ID desc and take 1
    let max_score = score::Entity::find()
        .order_by_desc(score::Column::Id)
        .one(&db)
        .await?;

    let max_id = match max_score {
        Some(s) => s.id,
        None => {
            println!("No scores found in database.");
            return Ok(());
        }
    };

    println!("Max Score ID: {}", max_id);

    // Pass the connection and length to the dataset
    // Note: DatabaseConnection is Clone (Arc internally)
    let dataset = DbDataset::new(db.clone(), max_id as usize);
    let batcher = ScoreBatcher::<Wgpu>::new();

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(32)
        .shuffle(42)
        .num_workers(4) // Use workers to parallelize DB fetches
        .build(dataset);

    println!("DataLoader created. Verifying first few batches...");

    // Iterate a few times
    let mut count = 0;
    for batch in dataloader.iter() {
        count += 1;
        if count <= 3 {
            println!(
                "Batch {}: UserIDs [{:?}...], Targets [{:?}...]",
                count,
                batch.user_ids.to_data(),
                batch.targets
            );
        } else {
            break;
        }
    }
    println!("Verification complete.");

    Ok(())
}
