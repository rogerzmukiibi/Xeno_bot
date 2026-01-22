#!/usr/bin/env python3
# setup_sqlite.py
import sys
sys.path.append('src')

print("🔧 Setting up SQLite Memory Database...")

import sqlite3
from src.config import SQLITE_DB_PATH

conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

conn.commit()

# Verify
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

print(f"✅ SQLite DB created at {SQLITE_DB_PATH}")
print(f"   Tables: {', '.join(tables)}")

conn.close()
