from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import uvicorn
import psycopg2
import psycopg2.extras
from typing import Optional, List
from langsmith import traceable
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

app = FastAPI(title="Judge Read API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_key: Optional[str] = ""
    llm_engine: str = "claude"
    api_key: str = ""
    # Metadata filters
    filter_year: Optional[int] = None
    filter_court: Optional[str] = None
    filter_jurisdiction: Optional[str] = None
    filter_status: Optional[str] = None
    filter_judge: Optional[str] = None
    filter_topic: Optional[str] = None
    # Tracing
    langsmith_key: Optional[str] = None
    # Reranking
    cohere_key: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list
    session_id: str

class ConfigModel(BaseModel):
    embeddingModel: Optional[str] = "text-embedding-3-small"
    embeddingKey: Optional[str] = ""
    llmEngine: Optional[str] = "claude"
    apiKey: Optional[str] = ""
    langsmithKey: Optional[str] = ""
    cohereKey: Optional[str] = ""
    pgHost: Optional[str] = "localhost"
    pgPort: Optional[str] = "5432"
    pgUser: Optional[str] = "user"
    pgPassword: Optional[str] = "password"
    pgDb: Optional[str] = "judgeread"

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def get_db_connection():
    pg_host = "localhost"
    pg_port = "5432"
    pg_user = "user"
    pg_password = "password"
    pg_db = "judgeread"
    
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                pg_host = cfg.get("pgHost") or pg_host
                pg_port = cfg.get("pgPort") or pg_port
                pg_user = cfg.get("pgUser") or pg_user
                pg_password = cfg.get("pgPassword") or pg_password
                pg_db = cfg.get("pgDb") or pg_db
        except Exception:
            pass
            
    db_url_from_config = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    db_url = os.getenv("DATABASE_URL", db_url_from_config).replace("+psycopg2", "")
    return psycopg2.connect(db_url)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/config")
def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

@app.post("/api/config")
def save_config(config: ConfigModel):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config.dict(), f, indent=4)
    return {"status": "saved"}

@app.post("/api/search", response_model=QueryResponse)
def search_cases(req: QueryRequest):
    # Setup LangSmith Tracing FIRST before calling the traced function
    if req.langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = req.langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = "Judge_Read"
    elif "LANGCHAIN_TRACING_V2" in os.environ:
        # Prevent leaking previous request's traces if no key provided
        del os.environ["LANGCHAIN_TRACING_V2"]
        
    return _run_search_pipeline(req)

@traceable(name="postgres_hybrid_search", run_type="retriever")
def fetch_from_postgres(cursor, hybrid_search_sql, query, filter_params):
    try:
        cursor.execute(hybrid_search_sql, [query] + filter_params)
        return cursor.fetchall()
    except Exception as e:
        print(f"Hybrid search failed, maybe tables aren't setup: {e}")
        cursor.connection.rollback()
        return []

@traceable(name="llm_generation", run_type="llm")
def generate_answer(llm_engine: str, api_key: str, sources: list, cursor, session_id: int):
    # Fetch chat history
    cursor.execute("SELECT role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC", (session_id,))
    history_rows = cursor.fetchall()
    
    messages = []
    
    # System Prompt with Context
    context_text = "\n\n".join([f"CASE {i+1}: {s['name']} ({s['reporter']})\nSTATUS: {s['status']}\nTEXT: {s['text']}" for i, s in enumerate(sources)])
    sys_prompt = f"""You are 'Judge Read', an expert AI legal assistant.
You will be provided with retrieved US case law. Answer the user's legal question based ONLY on these cases.
If a case is OVERRULED, you MUST mention that it is no longer good law and should not be relied upon.

CONTEXT:
{context_text}
"""
    messages.append(SystemMessage(content=sys_prompt))
    
    # Add history
    for row in history_rows:
        if row['role'] == 'user':
            messages.append(HumanMessage(content=row['content']))
        elif row['role'] == 'assistant':
            messages.append(AIMessage(content=row['content']))
            
    model_map = {
        "gpt-5.5-pro": "gpt-4o",
        "gpt-5.5": "gpt-4o-mini",
        "chat-latest": "gpt-4o",
        "o1": "o1-preview",
        "claude-sonnet-5": "claude-3-5-sonnet-20240620",
        "claude-fable-5": "claude-3-haiku-20240307",
        "claude-opus-4-8": "claude-3-opus-20240229"
    }
    actual_model = model_map.get(llm_engine, "gpt-4o-mini")

    try:
        if llm_engine.startswith("claude"):
            llm = ChatAnthropic(model_name=actual_model, anthropic_api_key=api_key)
        elif llm_engine == "ollama":
            llm = ChatOllama(model="qwen3-coder", base_url=api_key)
        else:
            llm = ChatOpenAI(model=actual_model, api_key=api_key)
            
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"LLM Generation failed: {e}")
        return f"Error communicating with LLM ({llm_engine}): {e}\n\nFallback Answer: Found {len(sources)} cases."

@traceable(name="judge_read_search_pipeline", run_type="chain")
def _run_search_pipeline(req: QueryRequest):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Session Handling
    session_id = req.session_id
    if not session_id:
        cursor.execute("INSERT INTO chat_sessions DEFAULT VALUES RETURNING id;")
        session_id = cursor.fetchone()[0]
        conn.commit()
    
    # Save user message to Chat History
    cursor.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)", 
                   (session_id, 'user', req.query))
    conn.commit()
    
    # Create Embedding for the query
    try:
        if req.embedding_model == "ollama":
            from langchain_ollama import OllamaEmbeddings
            embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=req.embedding_key)
        else:
            if req.embedding_key:
                os.environ["OPENAI_API_KEY"] = req.embedding_key
            embeddings = OpenAIEmbeddings(model=req.embedding_model)
            
        query_embedding = embeddings.embed_query(req.query)
    except Exception as e:
        print(f"Embedding failed (using mock vector): {e}")
        query_embedding = [0.0] * 1536 if req.embedding_model != "ollama" else [0.0] * 768
        
    # Hybrid Search Query + Metadata Filtering
    vector_query = f"'[{','.join(map(str, query_embedding))}]'"
    
    filter_sql = ""
    filter_params = []
    if req.filter_year:
        filter_sql += " AND (cmetadata->>'year')::int >= %s"
        filter_params.append(req.filter_year)
    if req.filter_court:
        filter_sql += " AND cmetadata->>'court' = %s"
        filter_params.append(req.filter_court)
    if req.filter_jurisdiction:
        if req.filter_jurisdiction == 'State':
            filter_sql += " AND cmetadata->>'jurisdiction' != 'Federal'"
        else:
            filter_sql += " AND cmetadata->>'jurisdiction' = %s"
            filter_params.append(req.filter_jurisdiction)
    if req.filter_status == 'good_law':
        filter_sql += " AND cmetadata->>'status' = 'good_law'"
    if req.filter_judge:
        filter_sql += " AND cmetadata->>'judge' ILIKE %s"
        filter_params.append(f"%{req.filter_judge}%")
    if req.filter_topic:
        filter_sql += " AND cmetadata->>'topic' = %s"
        filter_params.append(req.filter_topic)

    hybrid_search_sql = f"""
        SELECT 
            e.document, 
            e.cmetadata, 
            e.embedding <=> {vector_query} AS vector_distance,
            ts_rank_cd(e.tsvector_doc, plainto_tsquery('english', %s)) AS fts_rank,
            substring(f.full_text from '"case_name_full":\s*"([^"]+)"') AS case_name_full
        FROM langchain_pg_embedding e
        LEFT JOIN full_cases f ON (e.cmetadata->>'case_id') = f.case_id
        WHERE 1=1 {filter_sql.replace('cmetadata', 'e.cmetadata')}
        ORDER BY (e.embedding <=> {vector_query}) ASC, e.fts_rank DESC
        LIMIT 30;
    """
    
    results = fetch_from_postgres(cursor, hybrid_search_sql, req.query, filter_params)

    # Cohere Reranking
    if req.cohere_key and len(results) > 0:
        @traceable(name="cohere_rerank", run_type="retriever")
        def _perform_cohere_rerank(query: str, docs: list[str], api_key: str):
            import cohere
            co_client = cohere.Client(api_key)
            return co_client.rerank(
                query=query,
                documents=docs,
                model="rerank-english-v3.0",
                top_n=5
            )

        try:
            documents = [row['document'] for row in results]
            rerank_response = _perform_cohere_rerank(req.query, documents, req.cohere_key)
            
            reranked_results = []
            for r in rerank_response.results:
                reranked_results.append(results[r.index])
            results = reranked_results
            print(f"Successfully reranked {len(documents)} docs down to {len(results)} using Cohere.")
        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            results = results[:5]
    else:
        results = results[:5]

    # Format sources and extract Citator / Overruled status
    sources = []
    for row in results:
        meta = row['cmetadata']
        status = meta.get('status', 'good_law')
        sources.append({
            "case_id": meta.get('case_id'),
            "name": row.get('case_name_full') or meta.get('name', f"Case from {meta.get('year', 'Unknown')}"),
            "reporter": meta.get('court', 'Unknown Court'),
            "text": row['document'][:200] + "...",
            "status": status,
            "overruled": status == "overruled"
        })

    # Analytics / Telemetry Logging
    cursor.execute("""
        INSERT INTO analytics_queries (session_id, query, metadata_filters, results_returned)
        VALUES (%s, %s, %s, %s)
    """, (session_id, req.query, psycopg2.extras.Json({
        "year": req.filter_year, 
        "court": req.filter_court,
        "jurisdiction": req.filter_jurisdiction,
        "status": req.filter_status,
        "judge": req.filter_judge,
        "topic": req.filter_topic
    }), len(sources)))
    conn.commit()

    # Generate Answer using LLM
    answer = generate_answer(req.llm_engine, req.api_key, sources, cursor, session_id)
    
    # Save Assistant message
    cursor.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)", 
                   (session_id, 'assistant', answer))
    conn.commit()

    cursor.close()
    conn.close()

    return QueryResponse(answer=answer, sources=sources, session_id=session_id)

@app.get("/api/sessions/{session_id}/history")
def get_chat_history(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("SELECT role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC", (session_id,))
    messages = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}

from fastapi import HTTPException

@app.get("/api/cases/{case_id}")
def get_full_case(case_id: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("SELECT name, reporter, court, jurisdiction, year, status, full_text FROM full_cases WHERE case_id = %s", (case_id,))
    case_row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not case_row:
        raise HTTPException(status_code=404, detail="Full case document not found in database.")
        
    return dict(case_row)

@app.get("/api/cases")
def list_cases(
    skip: int = 0, 
    limit: int = 50, 
    search: str = None,
    system: str = None,
    state: str = None,
    court: str = None,
    status: str = None,
    judge: str = None,
    topic: str = None,
    year: str = None
):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    query = "SELECT case_id, COALESCE(substring(full_text from '\"case_name_full\":\s*\"([^\"]+)\"'), name) as name, reporter, court, jurisdiction, year, status FROM full_cases WHERE 1=1"
    params = []
    
    if search:
        query += " AND name ILIKE %s"
        params.append(f"%{search}%")
    if system == "Federal":
        query += " AND jurisdiction = 'Federal'"
    elif system == "State":
        query += " AND jurisdiction != 'Federal'"
        if state:
            query += " AND jurisdiction = %s"
            params.append(state)
    if court:
        query += " AND court = %s"
        params.append(court)
    if status == "good_law":
        query += " AND status = 'good_law'"
    if year:
        query += " AND year = %s"
        params.append(year)
        
    query += " ORDER BY year DESC NULLS LAST LIMIT %s OFFSET %s"
    params.extend([limit, skip])
    
    cursor.execute(query, params)
    cases = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"cases": [dict(c) for c in cases]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
