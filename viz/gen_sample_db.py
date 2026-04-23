import sqlite3
import json
import random

def gen_db(n=3):
    conn = sqlite3.connect('statpp.db')
    c = conn.cursor()
    
    c.execute('DROP TABLE IF EXISTS meta')
    c.execute('DROP TABLE IF EXISTS players')
    c.execute('DROP TABLE IF EXISTS beatmaps')
    c.execute('DROP TABLE IF EXISTS scores')
    
    c.execute('CREATE TABLE meta (key TEXT, value TEXT)')
    c.execute('INSERT INTO meta VALUES (?, ?)', ('dimensions', str(n)))
    
    c.execute('CREATE TABLE players (id INTEGER PRIMARY KEY, username TEXT, metadata TEXT)')
    for i in range(100):
        username = f'player_{i}'
        metadata = {'ratings': [random.random() for _ in range(n)]}
        c.execute('INSERT INTO players VALUES (?, ?, ?)', (i, username, json.dumps(metadata)))
        
    c.execute('CREATE TABLE beatmaps (id INTEGER PRIMARY KEY, artist TEXT, title TEXT, version TEXT, metadata TEXT)')
    for i in range(50):
        artist = f'Artist {i % 5}'
        title = f'Song Title {i}'
        version = f'Difficulty {random.choice(["Easy", "Normal", "Hard", "Insane", "Expert"])}'
        metadata = {'difficulties': [random.random() * 10 for _ in range(n)]}
        c.execute('INSERT INTO beatmaps VALUES (?, ?, ?, ?, ?)', (i, artist, title, version, json.dumps(metadata)))
        
    c.execute('CREATE TABLE scores (player_id INTEGER, beatmap_id INTEGER, mod TEXT, metadata TEXT)')
    for i in range(1000):
        pid = random.randint(0, 99)
        bid = random.randint(0, 49)
        metadata = {
            'scores': [random.random() * 1000000 for _ in range(n)],
            'accuracy': random.random()
        }
        c.execute('INSERT INTO scores VALUES (?, ?, ?, ?)', (pid, bid, 'NM', json.dumps(metadata)))
        
    conn.commit()
    conn.close()

if __name__ == '__main__':
    gen_db()
    print("Sample database 'statpp.db' generated with updated schema.")
