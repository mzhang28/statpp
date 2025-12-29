import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pymysql
import sqlite3
import json
from collections import defaultdict
from typing import Dict, Tuple, List
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OsuRatingSystem:
    def __init__(self,
                 n_dimensions: int = 32,
                 learning_rate: float = 0.01,
                 regularization: float = 0.001,
                 device: str = None):
        """
        Initialize the rating system.

        Args:
            n_dimensions: Number of latent dimensions for skills/difficulties
            learning_rate: Learning rate for optimization
            regularization: L2 regularization strength
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.n_dimensions = n_dimensions
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.player_embeddings = None
        self.beatmap_embeddings = None
        self.player_id_map = {}
        self.beatmap_id_map = {}
        self.inverse_player_map = {}
        self.inverse_beatmap_map = {}

    def load_data_from_mysql(self,
                            host: str,
                            user: str,
                            password: str,
                            database: str) -> Dict:
        """Load and preprocess data from MySQL database."""

        connection = pymysql.connect(host=host, user=user, password=password,
                                   database=database, cursorclass=pymysql.cursors.DictCursor)

        try:
            with connection.cursor() as cursor:
                # Note: Assuming beatmap_id exists in statpp_scores table
                # If not, you'll need to add a JOIN or separate lookup mechanism
                logger.info("Loading scores...")
                cursor.execute("""
                    SELECT score_id, beatmap_id, player_id, score, mods
                    FROM statpp_scores
                    WHERE score > 0
                """)
                scores = cursor.fetchall()

                if not scores:
                    # Try without beatmap_id if the column doesn't exist
                    logger.warning("beatmap_id not found in statpp_scores, attempting alternate query...")
                    # You would need to implement an alternative method here
                    # For now, raising an error
                    raise ValueError("beatmap_id column not found in statpp_scores table. "
                                   "Please ensure the scores table has a beatmap_id column.")

                # Load beatmap metadata for reference
                logger.info("Loading beatmap metadata...")
                cursor.execute("""
                    SELECT b.beatmap_id, b.difficulty_name, bs.artist, bs.title
                    FROM statpp_beatmap b
                    LEFT JOIN statpp_beatmapset bs ON b.beatmapset_id = bs.beatmapset_id
                """)
                beatmap_metadata = {row['beatmap_id']: row for row in cursor.fetchall()}

                # Process scores to get personal bests per (player, beatmap, mods) combination
                logger.info("Processing personal bests...")
                personal_bests = defaultdict(lambda: defaultdict(int))
                beatmap_mod_combos = set()

                for score_data in tqdm(scores, desc="Processing scores"):
                    beatmap_id = score_data['beatmap_id']
                    player_id = score_data['player_id']
                    score = score_data['score']

                    # Parse mods - now it's a direct varchar field
                    mods_str = score_data.get('mods', '')
                    if mods_str and mods_str.strip():
                        # Assuming mods are comma-separated or similar format
                        # Adjust parsing based on actual format in your database
                        if ',' in mods_str:
                            mods = sorted([m.strip() for m in mods_str.split(',') if m.strip()])
                        else:
                            # Could be space-separated or concatenated
                            # For concatenated like "HDDT", split into pairs
                            mods = []
                            mod_str_clean = mods_str.strip().upper()
                            # Common mod abbreviations in osu!
                            mod_abbrevs = ['NF', 'EZ', 'TD', 'HD', 'HR', 'SD', 'DT', 'NC',
                                         'HT', 'FL', 'SO', 'PF', 'RX', 'AP', 'TP']
                            i = 0
                            while i < len(mod_str_clean):
                                for abbrev in mod_abbrevs:
                                    if mod_str_clean[i:].startswith(abbrev):
                                        mods.append(abbrev)
                                        i += len(abbrev)
                                        break
                                else:
                                    i += 1
                            mods = sorted(mods)
                    else:
                        mods = []

                    # Create unique beatmap-mod combination identifier
                    mod_str = '_'.join(mods) if mods else 'nomod'
                    beatmap_mod_id = f"{beatmap_id}_{mod_str}"
                    beatmap_mod_combos.add((beatmap_id, mod_str))

                    # Track personal best
                    current_best = personal_bests[player_id][beatmap_mod_id]
                    if score > current_best:
                        personal_bests[player_id][beatmap_mod_id] = score

                # Get play counts for confidence weighting
                logger.info("Calculating play counts...")
                play_counts = defaultdict(int)
                for player_scores in personal_bests.values():
                    for beatmap_mod_id in player_scores:
                        play_counts[beatmap_mod_id] += 1

                return {
                    'personal_bests': personal_bests,
                    'beatmap_mod_combos': beatmap_mod_combos,
                    'play_counts': play_counts,
                    'beatmap_metadata': beatmap_metadata
                }

        finally:
            connection.close()

    def prepare_training_data(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert data to tensors for training."""

        personal_bests = data['personal_bests']
        play_counts = data['play_counts']

        # Create ID mappings
        unique_players = list(personal_bests.keys())
        unique_beatmaps = list(set(bm_id for player_scores in personal_bests.values()
                                  for bm_id in player_scores.keys()))

        self.player_id_map = {player_id: idx for idx, player_id in enumerate(unique_players)}
        self.beatmap_id_map = {beatmap_id: idx for idx, beatmap_id in enumerate(unique_beatmaps)}
        self.inverse_player_map = {idx: player_id for player_id, idx in self.player_id_map.items()}
        self.inverse_beatmap_map = {idx: beatmap_id for beatmap_id, idx in self.beatmap_id_map.items()}

        # Create training data
        player_indices = []
        beatmap_indices = []
        scores = []
        confidence_weights = []

        max_score = 1000000  # osu! max score

        for player_id, player_scores in personal_bests.items():
            player_idx = self.player_id_map[player_id]
            for beatmap_mod_id, score in player_scores.items():
                if beatmap_mod_id in self.beatmap_id_map:
                    beatmap_idx = self.beatmap_id_map[beatmap_mod_id]
                    player_indices.append(player_idx)
                    beatmap_indices.append(beatmap_idx)
                    scores.append(score / max_score)  # Normalize scores

                    # Confidence weight based on play count (logarithmic)
                    count = play_counts[beatmap_mod_id]
                    confidence = np.log1p(count) / np.log1p(max(play_counts.values()))
                    confidence_weights.append(confidence)

        # Convert to tensors
        player_indices = torch.LongTensor(player_indices).to(self.device)
        beatmap_indices = torch.LongTensor(beatmap_indices).to(self.device)
        scores = torch.FloatTensor(scores).to(self.device)
        confidence_weights = torch.FloatTensor(confidence_weights).to(self.device)

        return player_indices, beatmap_indices, scores, confidence_weights

    def initialize_embeddings(self, n_players: int, n_beatmaps: int):
        """Initialize player and beatmap embeddings."""

        # Xavier initialization for better convergence
        self.player_embeddings = nn.Parameter(
            torch.randn(n_players, self.n_dimensions, device=self.device) * 0.01
        )
        self.beatmap_embeddings = nn.Parameter(
            torch.randn(n_beatmaps, self.n_dimensions, device=self.device) * 0.01
        )

        # Also store bias terms
        self.player_bias = nn.Parameter(torch.zeros(n_players, device=self.device))
        self.beatmap_bias = nn.Parameter(torch.zeros(n_beatmaps, device=self.device))
        self.global_bias = nn.Parameter(torch.tensor(0.5, device=self.device))

    def predict_scores(self, player_indices: torch.Tensor, beatmap_indices: torch.Tensor) -> torch.Tensor:
        """Predict scores using dot product of embeddings plus biases."""

        player_vecs = self.player_embeddings[player_indices]
        beatmap_vecs = self.beatmap_embeddings[beatmap_indices]

        # Dot product + biases
        predictions = (player_vecs * beatmap_vecs).sum(dim=1)
        predictions += self.player_bias[player_indices]
        predictions += self.beatmap_bias[beatmap_indices]
        predictions += self.global_bias

        # Sigmoid to keep in [0, 1] range
        predictions = torch.sigmoid(predictions)

        return predictions

    def asymmetric_loss(self, predictions: torch.Tensor,
                       targets: torch.Tensor,
                       confidence_weights: torch.Tensor) -> torch.Tensor:
        """
        Custom loss function that only updates when actual > predicted.
        This implements the "only update on better than expected" rule.
        """

        # Calculate base MSE loss
        base_loss = (predictions - targets) ** 2

        # Apply asymmetric weighting: only penalize when actual > predicted
        # This encourages the system to underestimate rather than overestimate
        asymmetric_mask = (targets > predictions).float()

        # Weight by confidence (play count)
        weighted_loss = base_loss * asymmetric_mask * confidence_weights

        # Add L2 regularization
        reg_loss = self.regularization * (
            torch.norm(self.player_embeddings) ** 2 +
            torch.norm(self.beatmap_embeddings) ** 2
        ) / (self.player_embeddings.shape[0] + self.beatmap_embeddings.shape[0])

        return weighted_loss.mean() + reg_loss

    def train(self, data: Dict, n_epochs: int = 100, batch_size: int = 4096):
        """Train the rating system."""

        # Prepare data
        player_indices, beatmap_indices, scores, confidence_weights = self.prepare_training_data(data)

        n_players = len(self.player_id_map)
        n_beatmaps = len(self.beatmap_id_map)

        logger.info(f"Training with {n_players} players and {n_beatmaps} beatmap-mod combinations")
        logger.info(f"Total scores: {len(scores)}")

        # Initialize embeddings
        self.initialize_embeddings(n_players, n_beatmaps)

        # Setup optimizer
        parameters = [self.player_embeddings, self.beatmap_embeddings,
                     self.player_bias, self.beatmap_bias, self.global_bias]
        optimizer = optim.Adam(parameters, lr=self.learning_rate)

        # Training loop
        n_samples = len(scores)
        indices = torch.arange(n_samples)

        for epoch in range(n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            indices = indices[perm]

            epoch_loss = 0
            n_batches = 0

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]

                batch_players = player_indices[batch_idx]
                batch_beatmaps = beatmap_indices[batch_idx]
                batch_scores = scores[batch_idx]
                batch_confidence = confidence_weights[batch_idx]

                # Forward pass
                predictions = self.predict_scores(batch_players, batch_beatmaps)
                loss = self.asymmetric_loss(predictions, batch_scores, batch_confidence)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.6f}")

    def apply_time_adjustments(self, play_counts: Dict[str, int]):
        """
        Apply time-based adjustments to beatmap ratings.
        Low play count maps get boosted to incentivize play.
        """

        max_count = max(play_counts.values()) if play_counts else 1

        with torch.no_grad():
            for beatmap_mod_id, idx in self.beatmap_id_map.items():
                count = play_counts.get(beatmap_mod_id, 1)

                # Inverse confidence: low play count = high boost
                confidence = count / max_count
                boost_factor = 1.0 + (1.0 - confidence) * 0.1  # Up to 10% boost

                # Apply boost to beatmap difficulty (making it "harder")
                self.beatmap_embeddings[idx] *= boost_factor

    def calculate_ratings(self) -> Tuple[Dict, Dict]:
        """Calculate final scalar ratings from embeddings."""

        with torch.no_grad():
            # Calculate player ratings (L2 norm of skill vector)
            player_ratings = {}
            player_vectors = self.player_embeddings.cpu().numpy()
            for idx, player_id in self.inverse_player_map.items():
                # Use weighted sum with bias as overall rating
                rating = np.linalg.norm(player_vectors[idx]) * 1000  # Scale to reasonable range
                rating += self.player_bias[idx].cpu().item() * 100
                player_ratings[player_id] = float(rating)

            # Calculate beatmap ratings
            beatmap_ratings = {}
            beatmap_vectors = self.beatmap_embeddings.cpu().numpy()
            for idx, beatmap_id in self.inverse_beatmap_map.items():
                rating = np.linalg.norm(beatmap_vectors[idx]) * 1000
                rating += self.beatmap_bias[idx].cpu().item() * 100
                beatmap_ratings[beatmap_id] = float(rating)

        return player_ratings, beatmap_ratings

    def save_to_sqlite(self, db_path: str, player_ratings: Dict, beatmap_ratings: Dict,
                       beatmap_metadata: Dict = None):
        """Save ratings to SQLite database."""

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_ratings (
                player_id INTEGER PRIMARY KEY,
                rating REAL,
                skill_vector TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS beatmap_ratings (
                beatmap_mod_id TEXT PRIMARY KEY,
                beatmap_id INTEGER,
                mods TEXT,
                rating REAL,
                difficulty_vector TEXT,
                difficulty_name TEXT,
                artist TEXT,
                title TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_score_ratings (
                player_id INTEGER,
                beatmap_mod_id TEXT,
                predicted_score REAL,
                actual_score REAL,
                PRIMARY KEY (player_id, beatmap_mod_id)
            )
        """)

        # Save player ratings
        with torch.no_grad():
            for player_id, rating in player_ratings.items():
                idx = self.player_id_map[player_id]
                skill_vector = self.player_embeddings[idx].cpu().numpy().tolist()
                cursor.execute(
                    "INSERT OR REPLACE INTO player_ratings VALUES (?, ?, ?)",
                    (player_id, rating, json.dumps(skill_vector))
                )

        # Save beatmap ratings
        with torch.no_grad():
            for beatmap_mod_id, rating in beatmap_ratings.items():
                idx = self.beatmap_id_map[beatmap_mod_id]
                difficulty_vector = self.beatmap_embeddings[idx].cpu().numpy().tolist()

                # Parse beatmap_id and mods from combined ID
                parts = beatmap_mod_id.split('_', 1)
                beatmap_id = int(parts[0])
                mods = parts[1] if len(parts) > 1 else 'nomod'

                # Get metadata if available
                difficulty_name = artist = title = None
                if beatmap_metadata and beatmap_id in beatmap_metadata:
                    meta = beatmap_metadata[beatmap_id]
                    difficulty_name = meta.get('difficulty_name')
                    artist = meta.get('artist')
                    title = meta.get('title')

                cursor.execute(
                    "INSERT OR REPLACE INTO beatmap_ratings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (beatmap_mod_id, beatmap_id, mods, rating, json.dumps(difficulty_vector),
                     difficulty_name, artist, title)
                )

        conn.commit()
        conn.close()
        logger.info(f"Saved ratings to {db_path}")

def main():
    # Configuration
    config = {
        'mysql': {
            'host': 'localhost',
            'user': 'root',
            'password': 'root',
            'database': 'osu'  # Updated to match your database name
        },
        'model': {
            'n_dimensions': 32,  # Number of skill/difficulty dimensions
            'learning_rate': 0.01,
            'regularization': 0.001,
            'n_epochs': 100,
            'batch_size': 4096
        },
        'output_db': 'osu_ratings.db'
    }

    # Initialize system
    rating_system = OsuRatingSystem(
        n_dimensions=config['model']['n_dimensions'],
        learning_rate=config['model']['learning_rate'],
        regularization=config['model']['regularization']
    )

    # Load data
    logger.info("Loading data from MySQL...")
    data = rating_system.load_data_from_mysql(**config['mysql'])

    # Train model
    logger.info("Training rating system...")
    rating_system.train(
        data,
        n_epochs=config['model']['n_epochs'],
        batch_size=config['model']['batch_size']
    )

    # Apply time-based adjustments
    logger.info("Applying time-based adjustments...")
    rating_system.apply_time_adjustments(data['play_counts'])

    # Calculate final ratings
    logger.info("Calculating final ratings...")
    player_ratings, beatmap_ratings = rating_system.calculate_ratings()

    # Save to database
    rating_system.save_to_sqlite(config['output_db'], player_ratings, beatmap_ratings,
                                data.get('beatmap_metadata'))

    logger.info("Rating calculation complete!")

    # Print some statistics
    logger.info(f"Average player rating: {np.mean(list(player_ratings.values())):.2f}")
    logger.info(f"Average beatmap rating: {np.mean(list(beatmap_ratings.values())):.2f}")
    logger.info(f"Top player rating: {max(player_ratings.values()):.2f}")
    logger.info(f"Top beatmap rating: {max(beatmap_ratings.values()):.2f}")

if __name__ == "__main__":
    main()
