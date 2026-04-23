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
    
    c.execute('CREATE TABLE players (id INTEGER PRIMARY KEY, metadata TEXT)')
    for i in range(100):
        metadata = {
            'ratings': [random.random() for _ in range(n)],
            'username': f'player_{i}'
        }
        c.execute('INSERT INTO players VALUES (?, ?)', (i, json.dumps(metadata)))
        
    c.execute('CREATE TABLE beatmaps (id INTEGER PRIMARY KEY, metadata TEXT)')
    for i in range(50):
        metadata = {
            'difficulties': [random.random() * 10 for _ in range(n)],
            'title': f'beatmap_{i}'
        }
        c.execute('INSERT INTO beatmaps VALUES (?, ?)', (i, json.dumps(metadata)))
        
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
    print("Sample database 'statpp.db' generated.")
