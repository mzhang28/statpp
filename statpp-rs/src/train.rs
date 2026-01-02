use crate::entity::{beatmapmod_vector, user_vector};
use anyhow::Result;
use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    optim::{GradientsParams, Optimizer, SgdConfig},
    record::{CompactRecorder, Recorder},
    tensor::{
        Int, Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use indicatif::{ProgressBar, ProgressStyle};
use sea_orm::{
    ActiveModelTrait, ConnectionTrait, Database, DatabaseConnection, DbBackend, EntityTrait,
    FromQueryResult, Set, Statement,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;

// --- Constants ---

const LATENT_DIM: usize = 32;
const PLAYER_LR_SCALE: f64 = 1.0;
const MAP_LR_SCALE: f64 = 0.001;
const BASE_LR: f64 = 2.0; // SGD needs higher LR than Adam usually
const NUM_EPOCHS: usize = 10;
const BATCH_SIZE: usize = 4096;

// --- Data Structures ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreItem {
    pub user_idx: usize,
    pub map_idx: usize,
    pub score: f32,
    pub original_user_id: i32,
    pub original_map_id: i32,
}

#[derive(Debug, FromQueryResult)]
struct RawScoreRow {
    user_id: i32,
    map_id: i32,
    score: i32,
}

// --- Dataset ---

#[derive(Clone)]
pub struct StatPPDataset {
    items: Vec<ScoreItem>,
    pub num_users: usize,
    pub num_maps: usize,
    pub user_id_map: HashMap<i32, usize>,
    pub map_id_map: HashMap<i32, usize>,
}

impl StatPPDataset {
    pub async fn load(db: &DatabaseConnection) -> Result<Self> {
        println!("Loading training data from database...");
        let sql = r#"
            SELECT
                u.id as user_id,
                s.beatmap_mod as map_id,
                s.score
            FROM score s
            INNER JOIN user u ON s.osu_user = u.osu_id
        "#;

        let stmt = Statement::from_sql_and_values(DbBackend::MySql, sql, []);
        let rows = RawScoreRow::find_by_statement(stmt).all(db).await?;

        let mut items = Vec::with_capacity(rows.len());
        let mut user_id_map = HashMap::new();
        let mut map_id_map = HashMap::new();
        let mut next_user_idx = 0;
        let mut next_map_idx = 0;

        println!("Processing {} rows...", rows.len());
        let pb = ProgressBar::new(rows.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar().template("{spinner} {bar:40} {pos}/{len} ({eta})")?,
        );

        for row in rows {
            let u_idx = *user_id_map.entry(row.user_id).or_insert_with(|| {
                let i = next_user_idx;
                next_user_idx += 1;
                i
            });

            let m_idx = *map_id_map.entry(row.map_id).or_insert_with(|| {
                let i = next_map_idx;
                next_map_idx += 1;
                i
            });

            let norm_score = (row.score as f32 / 1_000_000.0).clamp(0.0, 1.0);

            items.push(ScoreItem {
                user_idx: u_idx,
                map_idx: m_idx,
                score: norm_score,
                original_user_id: row.user_id,
                original_map_id: row.map_id,
            });
            pb.inc(1);
        }
        pb.finish_with_message("Data loaded and indexed");

        println!(
            "Dataset stats: {} Scores, {} Unique Users, {} Unique Maps",
            items.len(),
            next_user_idx,
            next_map_idx
        );

        Ok(Self {
            items,
            num_users: next_user_idx,
            num_maps: next_map_idx,
            user_id_map,
            map_id_map,
        })
    }
}

impl Dataset<ScoreItem> for StatPPDataset {
    fn get(&self, index: usize) -> Option<ScoreItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

// --- Batcher ---

#[derive(Clone, Debug)]
pub struct ScoreBatch<B: Backend> {
    pub user_indices: Tensor<B, 1, Int>,
    pub map_indices: Tensor<B, 1, Int>,
    pub targets: Tensor<B, 1>,
}

#[derive(Clone)]
pub struct ScoreBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ScoreBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, ScoreItem, ScoreBatch<B>> for ScoreBatcher<B> {
    fn batch(&self, items: Vec<ScoreItem>, device: &B::Device) -> ScoreBatch<B> {
        let users: Vec<i32> = items.iter().map(|i| i.user_idx as i32).collect();
        let maps: Vec<i32> = items.iter().map(|i| i.map_idx as i32).collect();
        let targets: Vec<f32> = items.iter().map(|i| i.score).collect();

        let user_indices = Tensor::from_ints(users.as_slice(), &self.device);
        let map_indices = Tensor::from_ints(maps.as_slice(), &self.device);
        let targets = Tensor::from_floats(targets.as_slice(), &self.device);

        ScoreBatch {
            user_indices,
            map_indices,
            targets,
        }
    }
}

// --- Model ---

#[derive(Module, Debug)]
pub struct StatPPModel<B: Backend> {
    pub player_embed: Embedding<B>,
    pub map_embed: Embedding<B>,
    pub map_bias: Embedding<B>,
}

impl<B: Backend> StatPPModel<B> {
    pub fn new(num_users: usize, num_maps: usize, latent_dim: usize, device: &B::Device) -> Self {
        let player_embed = EmbeddingConfig::new(num_users, latent_dim).init(device);
        let map_embed = EmbeddingConfig::new(num_maps, latent_dim).init(device);
        let map_bias = EmbeddingConfig::new(num_maps, 1).init(device);

        Self {
            player_embed,
            map_embed,
            map_bias,
        }
    }

    pub fn forward_train(&self, batch: ScoreBatch<B>) -> Tensor<B, 1> {
        // 1. Map Branch (Map learns from everything)
        // Treat Player as Constant (detach)
        let p_vec_detached = self
            .player_embed
            .forward(batch.user_indices.clone().unsqueeze_dim(1))
            .detach()
            .squeeze::<2>();

        let m_vec = self
            .map_embed
            .forward(batch.map_indices.clone().unsqueeze_dim(1))
            .squeeze::<2>();

        let m_bias = self
            .map_bias
            .forward(batch.map_indices.clone().unsqueeze_dim(1))
            .flatten::<2>(1, 2)
            .squeeze::<1>();

        let dot_map = (p_vec_detached * m_vec).sum_dim(1).squeeze::<1>();
        let pred_map = burn::tensor::activation::sigmoid(dot_map - m_bias);

        // Standard MSE for Maps
        let loss_map_raw = (batch.targets.clone() - pred_map).powf_scalar(2.0).mean();

        // 2. Player Branch (Player learns only from success - Teflon)
        // Treat Map as Constant (detach)
        let p_vec = self
            .player_embed
            .forward(batch.user_indices.clone().unsqueeze_dim(1))
            .squeeze::<2>();

        let m_vec_detached = self
            .map_embed
            .forward(batch.map_indices.clone().unsqueeze_dim(1))
            .detach()
            .squeeze::<2>();

        let m_bias_detached = self
            .map_bias
            .forward(batch.map_indices.clone().unsqueeze_dim(1))
            .detach()
            .flatten::<2>(1, 2)
            .squeeze::<1>();

        let dot_player = (p_vec * m_vec_detached).sum_dim(1).squeeze::<1>();
        let pred_player = burn::tensor::activation::sigmoid(dot_player - m_bias_detached);

        // Calculate Error
        let error = batch.targets.clone() - pred_player;

        // Teflon Mask: If error > 0 (Underprediction / "I played better than expected"), learn.
        // If error < 0 (Overprediction / "I had a bad day"), ignore.
        let mask = error.clone().greater_elem(0.0).float();

        // Masked MSE for Players
        let loss_player_raw = (error.powf_scalar(2.0) * mask).mean();

        // 3. Combine with Inertia Scaling
        // We scale the LOSS, which scales the GRADIENTS.
        // SGD will respect this scaling.
        (loss_player_raw * PLAYER_LR_SCALE) + (loss_map_raw * MAP_LR_SCALE)
    }
}

// --- Ranking System (Supply & Demand) ---

#[derive(Serialize)]
struct PlayerRank {
    user_id: i32,
    rank_score: f64,
}

async fn calculate_and_save_rankings<B: Backend>(
    db: &DatabaseConnection,
    model: &StatPPModel<B>,
    dataset: &StatPPDataset,
) -> Result<()> {
    println!("Calculating Supply & Demand Rankings...");

    // 1. Extract Player Vectors
    let record = model.clone().into_record();
    let player_weights = record
        .player_embed
        .weight
        .to_data()
        .to_vec::<f32>()
        .unwrap();

    // 2. Compute Skill Scarcity (W_d)
    let mut dim_sums = vec![0.0; LATENT_DIM];
    let num_users = dataset.num_users as f64;

    for user_vec in player_weights.chunks(LATENT_DIM) {
        for (d, &val) in user_vec.iter().enumerate() {
            dim_sums[d] += val as f64;
        }
    }

    let mut scarcity_weights = vec![0.0; LATENT_DIM];
    for (d, &sum) in dim_sums.iter().enumerate() {
        let mean = sum / num_users;
        scarcity_weights[d] = 1.0 / (mean.abs() + 1e-6);
    }

    // 3. Compute Player Ranks
    let mut rankings = Vec::with_capacity(dataset.num_users);
    let index_to_user_id: HashMap<usize, i32> =
        dataset.user_id_map.iter().map(|(k, v)| (*v, *k)).collect();

    for (idx, user_vec_slice) in player_weights.chunks(LATENT_DIM).enumerate() {
        if let Some(&original_id) = index_to_user_id.get(&idx) {
            let mut wss = 0.0;
            for (d, &val) in user_vec_slice.iter().enumerate() {
                wss += ((val as f64) * scarcity_weights[d]).powi(2);
            }
            let rank_score = wss.sqrt();

            rankings.push(PlayerRank {
                user_id: original_id,
                rank_score,
            });
        }
    }

    rankings.sort_by(|a, b| b.rank_score.partial_cmp(&a.rank_score).unwrap());

    println!("Top 10 Players:");
    for (i, r) in rankings.iter().take(10).enumerate() {
        println!("{}. User {} - Score {:.4}", i + 1, r.user_id, r.rank_score);
    }

    save_vectors_to_db(db, &player_weights, &index_to_user_id, model, dataset).await?;

    Ok(())
}

async fn save_vectors_to_db<B: Backend>(
    db: &DatabaseConnection,
    player_weights: &[f32],
    index_to_user_id: &HashMap<usize, i32>,
    model: &StatPPModel<B>,
    dataset: &StatPPDataset,
) -> Result<()> {
    println!("Saving vectors to database (Deleting old data)...");

    // Clear old data
    user_vector::Entity::delete_many().exec(db).await?;
    beatmapmod_vector::Entity::delete_many().exec(db).await?;

    println!("Preparing user vectors...");
    let mut user_active_models = Vec::with_capacity(index_to_user_id.len());
    for (idx, user_vec_slice) in player_weights.chunks(LATENT_DIM).enumerate() {
        if let Some(&user_id) = index_to_user_id.get(&idx) {
            let vec_u8 = user_vec_slice
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>();

            user_active_models.push(user_vector::ActiveModel {
                user_id: Set(user_id),
                vec: Set(Some(vec_u8)),
                ..Default::default()
            });
        }
    }

    if !user_active_models.is_empty() {
        // Insert in chunks to avoid hitting packet limits
        for chunk in user_active_models.chunks(1000) {
            user_vector::Entity::insert_many(chunk.to_vec())
                .exec(db)
                .await?;
        }
        println!("Saved {} user vectors.", user_active_models.len());
    }

    println!("Preparing map vectors...");
    // Extract map weights
    // We need to create a new record to get weights out of Embedding module safely in Burn
    let record = model.clone().into_record();
    let map_weights = record.map_embed.weight.to_data().to_vec::<f32>().unwrap();

    let index_to_map_id: HashMap<usize, i32> =
        dataset.map_id_map.iter().map(|(k, v)| (*v, *k)).collect();

    let mut map_active_models = Vec::with_capacity(index_to_map_id.len());
    for (idx, map_vec_slice) in map_weights.chunks(LATENT_DIM).enumerate() {
        if let Some(&map_id) = index_to_map_id.get(&idx) {
            let vec_u8 = map_vec_slice
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>();

            map_active_models.push(beatmapmod_vector::ActiveModel {
                beatmapmod_id: Set(map_id),
                vec: Set(Some(vec_u8)),
                ..Default::default()
            });
        }
    }

    if !map_active_models.is_empty() {
        for chunk in map_active_models.chunks(1000) {
            beatmapmod_vector::Entity::insert_many(chunk.to_vec())
                .exec(db)
                .await?;
        }
        println!("Saved {} map vectors.", map_active_models.len());
    }

    Ok(())
}

// --- Main Execution ---

pub async fn run() -> Result<()> {
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let db = Database::connect(&database_url).await?;

    // setup_schema(&db).await?;

    let dataset = StatPPDataset::load(&db).await?;
    let num_users = dataset.num_users;
    let num_maps = dataset.num_maps;
    // Clone dataset for post-processing usage before moving it into DataLoader
    let ranking_dataset = dataset.clone();

    let device = WgpuDevice::default();
    let batcher = ScoreBatcher::<Autodiff<Wgpu>>::new(device.clone());

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(BATCH_SIZE)
        .shuffle(42)
        .num_workers(1)
        .build(dataset);

    let mut model = StatPPModel::new(num_users, num_maps, LATENT_DIM, &device);

    // Using SGD to ensure 'Force = Mass * Acceleration' (Inertia) metaphor works via loss scaling.
    let mut optimizer = SgdConfig::new().with_momentum(None).init();

    println!("Starting manual training loop...");
    println!("Device: {:?}", device);
    println!("Epochs: {}", NUM_EPOCHS);

    for epoch in 1..=NUM_EPOCHS {
        let mut iterator = dataloader.iter();
        let mut batch_idx = 0;
        let mut total_loss = 0.0;

        let pb = ProgressBar::new(dataloader.num_items() as u64 / BATCH_SIZE as u64);
        pb.set_style(
            ProgressStyle::default_bar().template("{spinner} {msg} {bar:40} {pos}/{len}")?,
        );
        pb.set_message(format!("Epoch {}", epoch));

        while let Some(batch) = iterator.next() {
            let weighted_loss = model.forward_train(batch);

            // Extract loss value for logging (approximate, before backward consumes it)
            // Note: extracting scalar from WGPU tensor is async/slow.
            // We'll just do it occasionally or not at all for speed in this CLI.
            // total_loss += ...;

            let grads = weighted_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(BASE_LR, model, grads);

            batch_idx += 1;
            pb.inc(1);
        }
        pb.finish_with_message(format!("Epoch {} Complete", epoch));
    }

    calculate_and_save_rankings(&db, &model, &ranking_dataset).await?;

    Ok(())
}

// async fn setup_schema(db: &DatabaseConnection) -> Result<()> {
//     let sql_player = r#"
//         CREATE TABLE IF NOT EXISTS player_vectors (
//             user_id INT PRIMARY KEY,
//             vector JSON,
//             rank_score DOUBLE
//         )
//     "#;
//     db.execute(Statement::from_string(
//         DbBackend::MySql,
//         sql_player.to_string(),
//     ))
//     .await?;

//     let sql_map = r#"
//         CREATE TABLE IF NOT EXISTS map_vectors (
//             map_id INT PRIMARY KEY,
//             vector JSON,
//             bias FLOAT
//         )
//     "#;
//     db.execute(Statement::from_string(
//         DbBackend::MySql,
//         sql_map.to_string(),
//     ))
//     .await?;
//     Ok(())
// }
