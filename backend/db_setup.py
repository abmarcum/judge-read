import os
import json
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_db():
    # Attempt to load db config from config.json
    db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/judgeread")
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                cfg = json.load(f)
                if all(k in cfg for k in ["pgUser", "pgPassword", "pgHost", "pgPort", "pgDb"]):
                    db_url = f"postgresql://{cfg['pgUser']}:{cfg['pgPassword']}@{cfg['pgHost']}:{cfg['pgPort']}/{cfg['pgDb']}"
        except Exception as e:
            print(f"Warning: Could not read config.json for DB credentials: {e}")

    # Wait, the format in LangChain is postgresql+psycopg2://
    # Psycopg2 expects postgresql://
    db_url_clean = db_url.replace("+psycopg2", "")
    
    try:
        conn = psycopg2.connect(db_url_clean)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    print("Setting up pgvector extension...")
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    except psycopg2.Error as e:
        print(f"Warning: Could not create pgvector extension: {e}")
        print("If it's not already installed, please ask a database superuser to run: CREATE EXTENSION vector;")

    print("Creating sessions table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS username VARCHAR(255);")

    print("Creating messages table for Session Memory...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY,
            session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role VARCHAR(50),
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    print("Creating analytics table for Telemetry...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics_queries (
            id SERIAL PRIMARY KEY,
            session_id UUID,
            query TEXT,
            metadata_filters JSONB,
            results_returned INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    print("Creating case_annotations table for user notes and highlights...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS case_annotations (
            id SERIAL PRIMARY KEY,
            session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
            case_id VARCHAR(255),
            highlighted_text TEXT NOT NULL,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    print("Creating full_cases table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS full_cases (
            case_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(500),
            reporter VARCHAR(255),
            court VARCHAR(255),
            jurisdiction VARCHAR(255),
            year INT,
            status VARCHAR(50),
            full_text TEXT
        );
    """)

    print("Creating search_cache table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            id SERIAL PRIMARY KEY,
            query_hash VARCHAR(64) UNIQUE,
            query_text TEXT NOT NULL,
            response_answer TEXT NOT NULL,
            response_sources JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Note: Full-Text Search (Hybrid Search) index should be added to the langchain table
    # However, langchain_pg_embedding table is created by LangChain at runtime during ingestion.
    # We can create a trigger or simply add an index if the table exists.
    try:
        cursor.execute("""
            ALTER TABLE langchain_pg_embedding ADD COLUMN IF NOT EXISTS tsvector_doc tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS tsvector_doc_idx ON langchain_pg_embedding USING GIN (tsvector_doc);
        """)
        print("Added FTS column and index for Hybrid Search on langchain_pg_embedding.")
    except Exception as e:
        print("Note: Could not add FTS index. (This is expected if data_pipeline.py hasn't run yet).", e)
        conn.rollback()

    cursor.close()
    conn.close()
    print("Database setup complete.")

if __name__ == "__main__":
    setup_db()
