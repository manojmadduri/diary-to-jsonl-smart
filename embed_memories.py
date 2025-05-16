#!/usr/bin/env python3
import json
import sqlite3
from sentence_transformers import SentenceTransformer

# ------------------------------
# SETUP
# ------------------------------

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and lightweight

# Connect to local SQLite database (creates if not exists)
conn = sqlite3.connect('memory_embeddings.db')
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input TEXT,
        output TEXT,
        date TEXT,
        topic TEXT,
        embedding BLOB
    )
''')
conn.commit()

# ------------------------------
# PROCESS
# ------------------------------

# Load your memories.jsonl
with open('memories.jsonl', 'r', encoding='utf-8') as f:
    memories = [json.loads(line.strip()) for line in f.readlines()]

print(f"✅ Loaded {len(memories)} memories.")

# Embed and insert
for mem in memories:
    full_text = f"Q: {mem['input']} A: {mem['output']}"
    emb = embed_model.encode(full_text)

    # Convert numpy array to bytes for storage
    emb_bytes = emb.tobytes()

    cursor.execute('''
        INSERT INTO memories (input, output, date, topic, embedding)
        VALUES (?, ?, ?, ?, ?)
    ''', (mem['input'], mem['output'], mem['date'], mem['topic'], emb_bytes))

conn.commit()
conn.close()

print("✅ All memories embedded and stored in 'memory_embeddings.db'!")
