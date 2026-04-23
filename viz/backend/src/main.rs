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
    #[arg(short, long, default_value = "statpp.db")]
    database: String,
    #[arg(short, long, default_value_t = 3000)]
    port: u16,
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
    username: String,
    metadata: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    rank: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct Beatmap {
    id: i64,
    artist: String,
    title: String,
    version: String,
    metadata: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct Score {
    player_id: i64,
    player_username: Option<String>,
    beatmap_id: i64,
    beatmap_title: Option<String>,
    beatmap_artist: Option<String>,
    beatmap_version: Option<String>,
    mod_str: String,
    metadata: serde_json::Value,
}

#[derive(Deserialize)]
struct TopQuery {
    dim: usize,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    dim: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let conn = Connection::open(&args.database)?;
    let n: usize = conn.query_row("SELECT value FROM meta WHERE key = 'dimensions'", [], |row| {
        let val: String = row.get(0)?;
        val.parse::<usize>().map_err(|_| rusqlite::Error::InvalidQuery)
    })?;
    
    for i in 0..n {
        conn.execute(&format!("CREATE INDEX IF NOT EXISTS idx_players_dim_{} ON players (json_extract(metadata, '$.ratings[{}]') DESC)", i, i), [])?;
        conn.execute(&format!("CREATE INDEX IF NOT EXISTS idx_beatmaps_dim_{} ON beatmaps (json_extract(metadata, '$.difficulties[{}]') DESC)", i, i), [])?;
        conn.execute(&format!("CREATE INDEX IF NOT EXISTS idx_scores_dim_{} ON scores (beatmap_id, json_extract(metadata, '$.scores[{}]') DESC)", i, i), [])?;
    }

    // Initialize Full Text Search for players
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS players_fts USING fts5(username, content='players', content_rowid='id')", [])?;
    conn.execute("INSERT INTO players_fts(players_fts) VALUES('rebuild')", [])?;
    
    let state = Arc::new(AppState { db_path: args.database.clone() });
    let app = Router::new()
        .route("/api/meta", get(get_meta))
        .route("/api/dimensions", get(get_dimensions))
        .route("/api/players/search", get(search_players))
        .route("/api/players/top", get(get_top_players))
        .route("/api/players/:id", get(get_player))
        .route("/api/players/:id/scores", get(get_player_scores))
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
    if path.is_empty() || path == "index.html" { return index_html().await; }
    match Assets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            Response::builder().header(header::CONTENT_TYPE, mime.as_ref())
                .body(axum::body::Body::from(content.data)).unwrap()
        }
        None => if path.contains('.') { (StatusCode::NOT_FOUND, "Not Found").into_response() } else { index_html().await }
    }
}

async fn index_html() -> Response {
    match Assets::get("index.html") {
        Some(content) => Html(content.data).into_response(),
        None => (StatusCode::NOT_FOUND, "Index not found").into_response(),
    }
}

async fn get_meta(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let conn = Connection::open(&state.db_path).unwrap();
    let mut stmt = conn.prepare("SELECT key, value FROM meta").unwrap();
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }).unwrap();

    let mut meta = serde_json::Map::new();
    for row in rows {
        let (k, v) = row.unwrap();
        meta.insert(k, serde_json::Value::String(v));
    }
    Json(meta)
}

async fn get_dimensions(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let conn = Connection::open(&state.db_path).unwrap();
    let n: Result<usize, _> = conn.query_row("SELECT value FROM meta WHERE key = 'dimensions'", [], |row| {
        let val: String = row.get(0)?;
        val.parse::<usize>().map_err(|_| rusqlite::Error::InvalidQuery)
    });
    match n {
        Ok(n) => (StatusCode::OK, Json(serde_json::json!({ "dimensions": n }))).into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Failed").into_response(),
    }
}

async fn get_top_players(State(state): State<Arc<AppState>>, Query(query): Query<TopQuery>) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let conn = Connection::open(&state.db_path).unwrap();
    let mut stmt = conn.prepare(&format!("SELECT id, username, metadata FROM players ORDER BY json_extract(metadata, '$.ratings[{}]') DESC LIMIT ?", query.dim)).unwrap();
    let players: Vec<Player> = stmt.query_map([limit], |row| {
        Ok(Player {
            id: row.get(0)?,
            username: row.get(1)?,
            metadata: serde_json::from_str(&row.get::<_, String>(2)?).unwrap(),
            rank: None, // We'll fill this in the next step
        })
    }).unwrap().map(|r| r.unwrap()).enumerate().map(|(i, mut p)| {
        p.rank = Some(i + 1);
        p
    }).collect();
    Json(players)
}

async fn search_players(State(state): State<Arc<AppState>>, Query(query): Query<SearchQuery>) -> impl IntoResponse {
    let conn = Connection::open(&state.db_path).unwrap();
    let q = format!("{}*", query.q);
    // Use a subquery to find the rank for each player in the current dimension
    let sql = format!(
        "SELECT p.id, p.username, p.metadata, \
         (SELECT count(*) + 1 FROM players p2 WHERE json_extract(p2.metadata, '$.ratings[{}]') > json_extract(p.metadata, '$.ratings[{}]')) as rank \
         FROM players p JOIN players_fts f ON p.id = f.rowid \
         WHERE players_fts MATCH ? ORDER BY rank LIMIT 20",
        query.dim, query.dim
    );
    let mut stmt = conn.prepare(&sql).unwrap();
    let players: Vec<Player> = stmt.query_map([q], |row| {
        Ok(Player {
            id: row.get(0)?,
            username: row.get(1)?,
            metadata: serde_json::from_str(&row.get::<_, String>(2)?).unwrap(),
            rank: Some(row.get::<_, usize>(3)?),
        })
    }).unwrap().map(|r| r.unwrap()).collect();
    Json(players)
}

async fn get_hardest_beatmaps(State(state): State<Arc<AppState>>, Query(query): Query<TopQuery>) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let conn = Connection::open(&state.db_path).unwrap();
    let mut stmt = conn.prepare(&format!("SELECT id, artist, title, version, metadata FROM beatmaps ORDER BY json_extract(metadata, '$.difficulties[{}]') DESC LIMIT ?", query.dim)).unwrap();
    let beatmaps: Vec<Beatmap> = stmt.query_map([limit], |row| {
        Ok(Beatmap {
            id: row.get(0)?,
            artist: row.get(1)?,
            title: row.get(2)?,
            version: row.get(3)?,
            metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap(),
        })
    }).unwrap().map(|r| r.unwrap()).collect();
    Json(beatmaps)
}

async fn get_beatmap(State(state): State<Arc<AppState>>, Path(id): Path<i64>) -> impl IntoResponse {
    let conn = Connection::open(&state.db_path).unwrap();
    let res: Result<Beatmap, _> = conn.query_row("SELECT id, artist, title, version, metadata FROM beatmaps WHERE id = ?", [id], |row| {
        Ok(Beatmap {
            id: row.get(0)?,
            artist: row.get(1)?,
            title: row.get(2)?,
            version: row.get(3)?,
            metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap(),
        })
    });
    match res {
        Ok(b) => (StatusCode::OK, Json(b)).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Not Found").into_response(),
    }
}

async fn get_player(State(state): State<Arc<AppState>>, Path(id): Path<i64>) -> impl IntoResponse {
    let conn = Connection::open(&state.db_path).unwrap();
    let res: Result<Player, _> = conn.query_row("SELECT id, username, metadata FROM players WHERE id = ?", [id], |row| {
        Ok(Player {
            id: row.get(0)?,
            username: row.get(1)?,
            metadata: serde_json::from_str(&row.get::<_, String>(2)?).unwrap(),
            rank: None,
        })
    });
    match res {
        Ok(p) => (StatusCode::OK, Json(p)).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Not Found").into_response(),
    }
}

async fn get_player_scores(State(state): State<Arc<AppState>>, Path(player_id): Path<i64>, Query(query): Query<TopQuery>) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let conn = Connection::open(&state.db_path).unwrap();
    let mut stmt = conn.prepare(&format!(
        "SELECT s.player_id, s.beatmap_id, b.title, b.artist, b.version, s.mod, s.metadata FROM scores s \
         JOIN beatmaps b ON s.beatmap_id = b.id \
         WHERE s.player_id = ? ORDER BY json_extract(s.metadata, '$.scores[{}]') DESC LIMIT ?",
        query.dim
    )).unwrap();
    let scores: Vec<Score> = stmt.query_map(params![player_id, limit], |row| {
        Ok(Score {
            player_id: row.get(0)?,
            player_username: None,
            beatmap_id: row.get(1)?,
            beatmap_title: Some(row.get(2)?),
            beatmap_artist: Some(row.get(3)?),
            beatmap_version: Some(row.get(4)?),
            mod_str: row.get(5)?,
            metadata: serde_json::from_str(&row.get::<_, String>(6)?).unwrap(),
        })
    }).unwrap().map(|r| r.unwrap()).collect();
    Json(scores)
}

async fn get_beatmap_scores(State(state): State<Arc<AppState>>, Path(beatmap_id): Path<i64>, Query(query): Query<TopQuery>) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let conn = Connection::open(&state.db_path).unwrap();
    let mut stmt = conn.prepare(&format!(
        "SELECT s.player_id, p.username, s.beatmap_id, s.mod, s.metadata FROM scores s \
         JOIN players p ON s.player_id = p.id \
         WHERE s.beatmap_id = ? ORDER BY json_extract(s.metadata, '$.scores[{}]') DESC LIMIT ?",
        query.dim
    )).unwrap();
    let scores: Vec<Score> = stmt.query_map(params![beatmap_id, limit], |row| {
        Ok(Score {
            player_id: row.get(0)?,
            player_username: Some(row.get(1)?),
            beatmap_id: row.get(2)?,
            beatmap_title: None,
            beatmap_artist: None,
            beatmap_version: None,
            mod_str: row.get(3)?,
            metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap(),
        })
    }).unwrap().map(|r| r.unwrap()).collect();
    Json(scores)
}
