
import mysql.connector
import argparse
from tqdm import tqdm

# --- Database Configuration ---
DB_CONFIG = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'port': 3306,
    'database': 'osu'
}

# --- Mod Mapping ---
MOD_MAPPING = {
    1: "NF",
    2: "EZ",
    4: "NV",
    8: "HD",
    16: "HR",
    32: "SD",
    64: "DT",
    128: "RX",
    256: "HT",
    512: "NC",
    1024: "FL",
    2048: "AT",
    4096: "SO",
    8192: "AP",
    16384: "PF",
}

def get_mods_list(enabled_mods):
    mods = []
    for mod_val, mod_name in MOD_MAPPING.items():
        if enabled_mods & mod_val:
            mods.append(mod_name)
    return sorted(mods)

def create_tables(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS statpp_beatmap (
            beatmap_id MEDIUMINT UNSIGNED PRIMARY KEY,
            beatmapset_id MEDIUMINT UNSIGNED,
            difficulty_name VARCHAR(80)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS statpp_beatmapset (
            beatmapset_id MEDIUMINT UNSIGNED PRIMARY KEY,
            artist VARCHAR(80),
            title VARCHAR(80)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS statpp_scores (
            score_id BIGINT UNSIGNED PRIMARY KEY,
            beatmap_id MEDIUMINT,
            player_id INT UNSIGNED,
            score INT,
            mods VARCHAR(255)
        )
    """)

def clear_tables(cursor):
    cursor.execute("TRUNCATE TABLE statpp_beatmap")
    cursor.execute("TRUNCATE TABLE statpp_beatmapset")
    cursor.execute("TRUNCATE TABLE statpp_scores")

def synthesize_data(cursor, cnx, sample_mode=False):
    batch_size = 1000

    def process_table(table_name, columns, order_by_column, insert_sql, transform_func=None, sample_mode=False):
        count_cursor = cnx.cursor(buffered=True)
        if sample_mode:
            total_records = 1000
        else:
            count_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = count_cursor.fetchone()[0]
        count_cursor.close()

        with tqdm(total=total_records, unit="records") as pbar:
            last_id = 0
            processed_records = 0

            records_to_process = 1000 if sample_mode else total_records

            while processed_records < records_to_process:
                select_cursor = cnx.cursor(buffered=True)
                limit = min(batch_size, records_to_process - processed_records)
                query = f"SELECT {columns} FROM {table_name} WHERE {order_by_column} > {last_id} ORDER BY {order_by_column} ASC LIMIT {limit}"
                select_cursor.execute(query)
                records = select_cursor.fetchall()
                select_cursor.close()

                if not records:
                    break

                if transform_func:
                    records = transform_func(records)

                insert_cursor = cnx.cursor()
                insert_cursor.executemany(insert_sql, records)
                insert_cursor.close()

                pbar.update(len(records))
                processed_records += len(records)
                last_id = records[-1][0]

    print("Synthesizing beatmaps...")
    process_table(
        table_name="osu_beatmaps",
        columns="beatmap_id, beatmapset_id, version",
        order_by_column="beatmap_id",
        insert_sql="INSERT INTO statpp_beatmap (beatmap_id, beatmapset_id, difficulty_name) VALUES (%s, %s, %s)",
        sample_mode=sample_mode
    )

    print("Synthesizing beatmapsets...")
    process_table(
        table_name="osu_beatmapsets",
        columns="beatmapset_id, artist, title",
        order_by_column="beatmapset_id",
        insert_sql="INSERT INTO statpp_beatmapset (beatmapset_id, artist, title) VALUES (%s, %s, %s)",
        sample_mode=sample_mode
    )

    def transform_scores(scores):
        transformed_scores = []
        # last_id is the score_id, which is the first column
        for score_id, beatmap_id, user_id, score, enabled_mods in scores:
            mods_list = get_mods_list(enabled_mods)
            mods_str = "|".join(mods_list)
            transformed_scores.append((score_id, beatmap_id, user_id, score, mods_str))
        return transformed_scores

    print("Synthesizing scores...")
    process_table(
        table_name="osu_scores_high",
        columns="score_id, beatmap_id, user_id, score, enabled_mods",
        order_by_column="score_id",
        insert_sql="INSERT INTO statpp_scores (score_id, beatmap_id, player_id, score, mods) VALUES (%s, %s, %s, %s, %s)",
        transform_func=transform_scores,
        sample_mode=sample_mode
    )


def main():
    parser = argparse.ArgumentParser(description="Synthesize statpp tables.")
    parser.add_argument("--sample", action="store_true", help="Run in sample mode.")
    args = parser.parse_args()

    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()

        print("Creating tables...")
        create_tables(cursor)

        print("Clearing tables...")
        clear_tables(cursor)

        print(f"Synthesizing data... (Sample mode: {args.sample})")
        synthesize_data(cursor, cnx, sample_mode=args.sample)

        cnx.commit()
        print("Data synthesis complete.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'cnx' in locals() and cnx.is_connected():
            cursor.close()
            cnx.close()

if __name__ == "__main__":
    main()
