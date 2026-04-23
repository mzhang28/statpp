use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode, Uri},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use clap::Parser;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use rust_embed::RustEmbed;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the SQLite database
    #[arg(short, long, default_value = "statpp.db")]
    database: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 3000)]
    port: u16,

    /// Interface to bind to
    #[arg(short, long, default_value = "0.0.0.0")]
    host: String,
}

#[derive(RustEmbed)]
#[folder = "../frontend/dist/"]
struct Assets;

#[derive(Clone)]
struct AppState {
    db_path: String,
}

#[derive(Serialize, Deserialize)]
struct Player {
    id: i64,
    metadata: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct Beatmap {
    id: i64,
    metadata: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct Score {
    player_id: i64,
    beatmap_id: i64,
    mod_str: String,
    metadata: serde_json::Value,
}

#[derive(Deserialize)]
struct TopQuery {
    dim: usize,
    limit: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("Connecting to database: {}", args.database);
    
    let conn = Connection::open(&args.database)?;
    
    let n: usize = conn.query_row(
        "SELECT value FROM meta WHERE key = 'dimensions'",
        [],
        |row| {
            let val: String = row.get(0)?;
            val.parse::<usize>().map_err(|_| rusqlite::Error::InvalidQuery)
        },
    )?;
    
    println!("Found {} dimensions. Creating/Verifying indexes...", n);
    
    for i in 0..n {
        conn.execute(
            &format!("CREATE INDEX IF NOT EXISTS idx_players_dim_{} ON players (json_extract(metadata, '$.ratings[{}]') DESC)", i, i),
            [],
        )?;
        conn.execute(
            &format!("CREATE INDEX IF NOT EXISTS idx_beatmaps_dim_{} ON beatmaps (json_extract(metadata, '$.difficulties[{}]') DESC)", i, i),
            [],
        )?;
        conn.execute(
            &format!("CREATE INDEX IF NOT EXISTS idx_scores_dim_{} ON scores (beatmap_id, json_extract(metadata, '$.scores[{}]') DESC)", i, i),
            [],
        )?;
    }
    
    let state = Arc::new(AppState { db_path: args.database.clone() });
    
    let app = Router::new()
        .route("/api/dimensions", get(get_dimensions))
        .route("/api/players/top", get(get_top_players))
        .route("/api/beatmaps/hardest", get(get_hardest_beatmaps))
        .route("/api/beatmaps/:id", get(get_beatmap))
        .route("/api/beatmaps/:id/scores", get(get_beatmap_scores))
        .fallback(static_handler)
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("Statpp Viewer running on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

async fn static_handler(uri: Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/');

    if path.is_empty() || path == "index.html" {
        return index_html().await;
    }

    match Assets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            Response::builder()
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(axum::body::Body::from(content.data))
                .unwrap()
        }
        None => {
            if path.contains('.') {
                return (StatusCode::NOT_FOUND, "Not Found").into_response();
            }
            index_html().await
        }
    }
}

async fn index_html() -> Response {
    match Assets::get("index.html") {
        Some(content) => Html(content.data).into_response(),
        None => (StatusCode::NOT_FOUND, "Index not found").into_response(),
    }
}

async fn get_dimensions(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let conn = Connection::open(&state.db_path).unwrap();
    let n: Result<usize, _> = conn.query_row(
        "SELECT value FROM meta WHERE key = 'dimensions'",
        [],
        |row| {
            let val: String = row.get(0)?;
            val.parse::<usize>().map_err(|_| rusqlite::Error::InvalidQuery)
        },
    );

    match n {
        Ok(n) => (StatusCode::OK, Json(serde_json::json!({ "dimensions": n }))).into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Failed to read dimensions").into_response(),
    }
}

async fn get_top_players(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TopQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let conn = Connection::open(&state.db_path).unwrap();
    
    let mut stmt = conn.prepare(&format!(
        "SELECT id, metadata FROM players ORDER BY json_extract(metadata, '$.ratings[{}]') DESC LIMIT ?",
        query.dim
    )).unwrap();
    
    let players: Vec<Player> = stmt.query_map([limit], |row| {
        Ok(Player {
            id: row.get(0)?,
            metadata: serde_json::from_str::<serde_json::Value>(&row.get::<_, String>(1)?).unwrap(),
        })
    }).unwrap().map(|r| r.unwrap()).collect();

    Json(players)
}

async fn get_hardest_beatmaps(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TopQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let conn = Connection::open(&state.db_path).unwrap();
    
    let mut stmt = conn.prepare(&format!(
        "SELECT id, metadata FROM beatmaps ORDER BY json_extract(metadata, '$.difficulties[{}]') DESC LIMIT ?",
        query.dim
    )).unwrap();
    
    let beatmaps: Vec<Beatmap> = stmt.query_map([limit], |row| {
        Ok(Beatmap {
            id: row.get(0)?,
            metadata: serde_json::from_str::<serde_json::Value>(&row.get::<_, String>(1)?).unwrap(),
        })
    }).unwrap().map(|r| r.unwrap()).collect();

    Json(beatmaps)
}

async fn get_beatmap(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    let conn = Connection::open(&state.db_path).unwrap();
    let res: Result<Beatmap, _> = conn.query_row(
        "SELECT id, metadata FROM beatmaps WHERE id = ?",
        [id],
        |row| {
            Ok(Beatmap {
                id: row.get(0)?,
                metadata: serde_json::from_str::<serde_json::Value>(&row.get::<_, String>(1)?).unwrap(),
            })
        },
    );

    match res {
        Ok(b) => (StatusCode::OK, Json(b)).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Beatmap not found").into_response(),
    }
}

async fn get_beatmap_scores(
    State(state): State<Arc<AppState>>,
    Path(beatmap_id): Path<i64>,
    Query(query): Query<TopQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let conn = Connection::open(&state.db_path).unwrap();
    
    let mut stmt = conn.prepare(&format!(
        "SELECT player_id, beatmap_id, mod, metadata FROM scores WHERE beatmap_id = ? ORDER BY json_extract(metadata, '$.scores[{}]') DESC LIMIT ?",
        query.dim
    )).unwrap();
    
    let scores: Vec<Score> = stmt.query_map(params![beatmap_id, limit], |row| {
        Ok(Score {
            player_id: row.get(0)?,
            beatmap_id: row.get(1)?,
            mod_str: row.get(2)?,
            metadata: serde_json::from_str::<serde_json::Value>(&row.get::<_, String>(3)?).unwrap(),
        })
    }).unwrap().map(|r| r.unwrap()).collect();

    Json(scores)
}
