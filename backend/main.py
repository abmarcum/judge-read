from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import uvicorn
import psycopg2
import psycopg2.extras
from typing import Optional, List
import re
import pypdf
import time
import io
from langsmith import traceable
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mcp.server.fastmcp import FastMCP

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
    username: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_key: Optional[str] = ""
    llm_engine: str = "claude"
    openai_api_key: Optional[str] = ""
    anthropic_api_key: Optional[str] = ""
    ollama_host: Optional[str] = "http://localhost:11434"
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
    expand_query: Optional[bool] = False

class QueryResponse(BaseModel):
    answer: str
    sources: list
    session_id: str
    cached: Optional[bool] = False
    steps: Optional[list] = []

class ConfigModel(BaseModel):
    embeddingModel: Optional[str] = "text-embedding-3-small"
    embeddingKey: Optional[str] = ""
    llmEngine: Optional[str] = "claude"
    openaiApiKey: Optional[str] = ""
    anthropicApiKey: Optional[str] = ""
    ollamaHost: Optional[str] = "http://localhost:11434"
    langsmithKey: Optional[str] = ""
    cohereKey: Optional[str] = ""
    pgHost: Optional[str] = "localhost"
    pgPort: Optional[str] = "5432"
    pgUser: Optional[str] = "user"
    pgPassword: Optional[str] = "password"
    availableModels: Optional[List[str]] = []
    availableEmbeddingModels: Optional[List[str]] = []

class AnnotationCreate(BaseModel):
    case_id: str
    highlighted_text: str
    note: Optional[str] = ""

class CitationResolveRequest(BaseModel):
    text: str

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
        
    try:
        return _run_search_pipeline(req)
    except Exception as e:
        print(f"Error in search pipeline: {e}")
        import traceback
        traceback.print_exc()
        return QueryResponse(
            answer=f"I encountered an error connecting to the retrieval system: {e}\n\nPlease check your backend and model configuration.",
            sources=[],
            session_id=req.session_id or "error",
            cached=False,
            steps=[f"❌ Pipeline Error: {e}"]
        )

@traceable(name="postgres_hybrid_search", run_type="retriever")
def fetch_from_postgres(cursor, hybrid_search_sql, query, filter_params):
    try:
        cursor.execute(hybrid_search_sql, [query] + filter_params)
        return cursor.fetchall()
    except Exception as e:
        print(f"Hybrid search failed, maybe tables aren't setup: {e}")
        try:
            cursor.connection.rollback()
        except Exception:
            pass
        return []

def _extract_response_text(response) -> str:
    if not response:
        return ""
    if isinstance(response.content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) 
            for b in response.content
        )
    elif isinstance(response.content, dict):
        return response.content.get("text", str(response.content))
    else:
        return str(response.content)

@traceable(name="llm_generation", run_type="llm")
def generate_answer(llm_engine: str, openai_key: str, anthropic_key: str, ollama_host: str, sources: list, cursor, session_id: int, steps_run: list = None):
    # Fetch chat history
    cursor.execute("SELECT role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC", (session_id,))
    history_rows = cursor.fetchall()
    
    messages = []
    
    # Filter sources to only include good law cases
    good_sources = []
    for s in sources:
        if s.get('status') == 'good_law':
            good_sources.append(s)
        else:
            print(f"⚠️ Excluding case '{s['name']}' from LLM response context because its status is '{s['status']}'.")
            if steps_run is not None:
                steps_run.append(f"⚠️ Safety Filter: Excluded non-good law case '{s['name']}'")

    # System Prompt with Context
    context_text = "\n\n".join([f"CASE {i+1}: {s['name']} ({s['reporter']})\nSTATUS: {s['status']}\nTEXT: {s['text']}" for i, s in enumerate(good_sources)])
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
    try:
        if ":" in llm_engine:
            provider, actual_model = llm_engine.split(":", 1)
            if provider == "Anthropic":
                llm = ChatAnthropic(model_name=actual_model, anthropic_api_key=anthropic_key)
            elif provider == "Ollama":
                llm = ChatOllama(model=actual_model, base_url=ollama_host)
            else:
                llm = ChatOpenAI(model=actual_model, api_key=openai_key)
        else:
            # Legacy fallback
            actual_model = model_map.get(llm_engine, "gpt-4o-mini")
            if llm_engine.startswith("claude"):
                llm = ChatAnthropic(model_name=actual_model, anthropic_api_key=anthropic_key)
            elif llm_engine == "ollama":
                llm = ChatOllama(model="qwen3-coder", base_url=ollama_host)
            else:
                llm = ChatOpenAI(model=actual_model, api_key=openai_key)
            
        # Step 1: Initial LLM response
        step_start = time.time()
        response = llm.invoke(messages)
        initial_answer = _extract_response_text(response)
        duration_ms = (time.time() - step_start) * 1000
        if steps_run is not None:
            steps_run.append(f"📄 Initial Base Answer: Generated objective content ({duration_ms:.1f}ms)")
        
        # Step 2: Attorney Agent (Lawyer Agent)
        step_start = time.time()
        attorney_prompt = f"""You are a senior litigation attorney representing a client.
Your task is to take the initial objective search response, review the provided case context, and rewrite/modify the response from an advocate's perspective. 
Strengthen the legal framing, emphasize the precedents in context that favor your client's posture, highlight strategic legal options, and point out any negative precedent risks.

RETRIVED CASE CONTEXT:
{context_text}

INITIAL OBJECTIVE RESPONSE:
{initial_answer}

Produce a revised response written from a professional attorney's viewpoint.
"""
        attorney_msg = [
            SystemMessage(content="You are a senior litigation attorney."),
            HumanMessage(content=attorney_prompt)
        ]
        print("⚖️ Invoking Attorney Agent...")
        attorney_response = llm.invoke(attorney_msg)
        attorney_answer = _extract_response_text(attorney_response)
        duration_ms = (time.time() - step_start) * 1000
        if steps_run is not None:
            steps_run.append(f"⚖️ Attorney Agent: Framed response from client advocate's perspective ({duration_ms:.1f}ms)")

        # Step 3: Judge Agent
        step_start = time.time()
        judge_prompt = f"""You are a federal judge writing an opinion.
Your task is to take the attorney agent's legal arguments and the raw case context, and draft the final, authoritative, balanced, and objective answer.
Correct any one-sided bias from the attorney, ensure all legal citations are applied accurately, reject legal hyperbole, and structure the final response as a clear, balanced judicial guidance.

RETRIVED CASE CONTEXT:
{context_text}

ATTORNEY'S ADVOCACY RESPONSE:
{attorney_answer}

Produce the final response written from a judge's objective, authoritative viewpoint.
"""
        judge_msg = [
            SystemMessage(content="You are an objective federal judge."),
            HumanMessage(content=judge_prompt)
        ]
        print("👨‍⚖️ Invoking Judge Agent...")
        judge_response = llm.invoke(judge_msg)
        final_answer = _extract_response_text(judge_response)
        duration_ms = (time.time() - step_start) * 1000
        if steps_run is not None:
            steps_run.append(f"👨‍⚖️ Judge Agent: Refined response to be balanced and authoritative ({duration_ms:.1f}ms)")
        
        return final_answer
            
    except Exception as e:
        print(f"LLM Generation failed: {e}")
        return f"Error communicating with LLM ({llm_engine}): {e}\n\nFallback Answer: Found {len(good_sources)} cases."

def _expand_query(query: str, llm_engine: str, openai_key: str, anthropic_key: str, ollama_host: str) -> list[str]:
    system_prompt = "You are a legal research assistant. Expand the user's search query into 3 distinct, search-engine friendly legal search queries (combining key legal concepts, keywords, or potential search terms). Return ONLY the 3 queries, one per line, with no labels, numbers, or bullet points."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Original Query: {query}")
    ]
    model_map = {
        "gpt-5.5-pro": "gpt-4o",
        "gpt-5.5": "gpt-4o-mini",
        "chat-latest": "gpt-4o",
        "o1": "o1-preview",
        "claude-sonnet-5": "claude-3-5-sonnet-20240620",
        "claude-fable-5": "claude-3-haiku-20240307",
        "claude-opus-4-8": "claude-3-opus-20240229"
    }
    try:
        if ":" in llm_engine:
            provider, actual_model = llm_engine.split(":", 1)
            if provider == "Anthropic":
                llm = ChatAnthropic(model_name=actual_model, anthropic_api_key=anthropic_key)
            elif provider == "Ollama":
                llm = ChatOllama(model=actual_model, base_url=ollama_host)
            else:
                llm = ChatOpenAI(model=actual_model, api_key=openai_key)
        else:
            actual_model = model_map.get(llm_engine, "gpt-4o-mini")
            if llm_engine.startswith("claude"):
                llm = ChatAnthropic(model_name=actual_model, anthropic_api_key=anthropic_key)
            elif llm_engine == "ollama":
                llm = ChatOllama(model="qwen3-coder", base_url=ollama_host)
            else:
                llm = ChatOpenAI(model=actual_model, api_key=openai_key)
        response = llm.invoke(messages)
        content = ""
        if isinstance(response.content, list):
            content = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in response.content)
        elif isinstance(response.content, dict):
            content = response.content.get("text", str(response.content))
        else:
            content = str(response.content)
        queries = [q.strip() for q in content.split("\n") if q.strip()]
        clean_queries = []
        for q in queries:
            clean_q = re.sub(r'^\d+[\.\-\s]+|^[\-\*\u2022\s]+', '', q).strip()
            if clean_q:
                clean_queries.append(clean_q)
        return clean_queries[:3]
    except Exception as e:
        print(f"Query expansion failed: {e}")
        return [query]

import hashlib

def _get_cache_hash(req: QueryRequest) -> str:
    cache_dict = {
        "query": (req.query or "").strip().lower(),
        "expand_query": req.expand_query,
        "llm_engine": req.llm_engine,
        "embedding_model": req.embedding_model,
        "filter_year": req.filter_year,
        "filter_court": req.filter_court,
        "filter_jurisdiction": req.filter_jurisdiction,
        "filter_status": req.filter_status,
        "filter_judge": req.filter_judge,
        "filter_topic": req.filter_topic
    }
    serialized = json.dumps(cache_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

@traceable(name="judge_read_search_pipeline", run_type="chain")
def _run_search_pipeline(req: QueryRequest):
    import time
    pipeline_start = time.time()
    steps_run = []

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Session Handling
    session_id = req.session_id
    if not session_id:
        cursor.execute("INSERT INTO chat_sessions (username) VALUES (%s) RETURNING id;", (req.username,))
        session_id = cursor.fetchone()[0]
        conn.commit()

    # Compute cache hash
    cache_hash = _get_cache_hash(req)
    
    # Check cache
    step_start = time.time()
    try:
        cursor.execute("""
            SELECT response_answer, response_sources 
            FROM search_cache 
            WHERE query_hash = %s;
        """, (cache_hash,))
        cache_row = cursor.fetchone()
        
        if cache_row:
            cached_answer = cache_row["response_answer"]
            cached_sources = cache_row["response_sources"]
            
            # Save user message to Chat History
            cursor.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)", 
                           (session_id, 'user', req.query))
            # Save Assistant cached message
            cursor.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)", 
                           (session_id, 'assistant', cached_answer))
            conn.commit()
            
            print(f"✅ Cache HIT for query: '{req.query}'")
            
            duration_ms = (time.time() - step_start) * 1000
            steps_run.append(f"🔍 Cache check: HIT (Instant retrieval, {duration_ms:.1f}ms)")
            total_ms = (time.time() - pipeline_start) * 1000
            steps_run.append(f"🏁 Execution Finished: Total query pipeline time ({total_ms:.1f}ms)")
            
            cursor.close()
            conn.close()
            return QueryResponse(answer=cached_answer, sources=cached_sources, session_id=session_id, cached=True, steps=steps_run)
    except Exception as cache_err:
        print(f"Warning: Cache check failed: {cache_err}")
        conn.rollback()
        
    duration_ms = (time.time() - step_start) * 1000
    steps_run.append(f"🔍 Cache check: MISS (Direct generation required, {duration_ms:.1f}ms)")

    # Save user message to Chat History (if cache missed)
    cursor.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)", 
                   (session_id, 'user', req.query))
    conn.commit()
    
    # Query Expansion Step
    step_start = time.time()
    queries_to_search = [req.query]
    if req.expand_query:
        expanded = _expand_query(req.query, req.llm_engine, req.openai_api_key, req.anthropic_api_key, req.ollama_host)
        print(f"Query Expansion triggered. Expanded queries: {expanded}")
        queries_to_search.extend(expanded)
        duration_ms = (time.time() - step_start) * 1000
        steps_run.append(f"🔄 Query Expansion: Generated 3 search variations ({duration_ms:.1f}ms)")
    else:
        steps_run.append("🔄 Query Expansion: Skipped (Using raw search query)")

    # Initialize Embedding once
    embeddings = None
    try:
        if ":" in req.embedding_model:
            provider, actual_model = req.embedding_model.split(":", 1)
            if provider == "Ollama":
                from langchain_ollama import OllamaEmbeddings
                embeddings = OllamaEmbeddings(model=actual_model, base_url=req.embedding_key)
            else: # defaults to OpenAI
                if req.embedding_key:
                    os.environ["OPENAI_API_KEY"] = req.embedding_key
                embeddings = OpenAIEmbeddings(model=actual_model)
        else:
            # Legacy fallback
            if req.embedding_model == "ollama":
                from langchain_ollama import OllamaEmbeddings
                embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=req.embedding_key)
            else:
                if req.embedding_key:
                    os.environ["OPENAI_API_KEY"] = req.embedding_key
                embeddings = OpenAIEmbeddings(model=req.embedding_model)
    except Exception as e:
        print(f"Embedding initialization failed: {e}")

    # Build filters once
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

    combined_results = []
    seen_case_ids = set()

    for current_query in queries_to_search:
        try:
            if embeddings:
                query_embedding = embeddings.embed_query(current_query)
            else:
                raise ValueError("No embeddings object initialized")
        except Exception as e:
            print(f"Embedding failed for '{current_query}': {e}")
            query_embedding = [0.0] * 1536 if "Ollama" not in req.embedding_model and "ollama" not in req.embedding_model else [0.0] * 768

    # Embeddings & Database lookup step
    step_start = time.time()
    combined_results = []
    seen_case_ids = set()

    for current_query in queries_to_search:
        try:
            if embeddings:
                query_embedding = embeddings.embed_query(current_query)
            else:
                raise ValueError("No embeddings object initialized")
        except Exception as e:
            print(f"Embedding failed for '{current_query}': {e}")
            query_embedding = [0.0] * 1536 if "Ollama" not in req.embedding_model and "ollama" not in req.embedding_model else [0.0] * 768

        vector_query = f"'[{','.join(map(str, query_embedding))}]'"
        
        hybrid_search_sql = f"""
            SELECT 
                e.document, 
                e.cmetadata, 
                e.embedding <=> {vector_query} AS vector_distance,
                ts_rank_cd(e.tsvector_doc, plainto_tsquery('english', %s)) AS fts_rank,
                substring(f.full_text from '"case_name_full":\\s*"([^"]+)"') AS case_name_full
            FROM langchain_pg_embedding e
            LEFT JOIN full_cases f ON (e.cmetadata->>'case_id') = f.case_id
            WHERE 1=1 {filter_sql.replace('cmetadata', 'e.cmetadata')}
            ORDER BY (e.embedding <=> {vector_query}) ASC, fts_rank DESC
            LIMIT 15;
        """
        
        try:
            curr_results = fetch_from_postgres(cursor, hybrid_search_sql, current_query, filter_params)
            for row in curr_results:
                case_id = row['cmetadata'].get('case_id')
                if case_id and case_id not in seen_case_ids:
                    seen_case_ids.add(case_id)
                    combined_results.append(row)
        except Exception as e:
            print(f"Search failed for '{current_query}': {e}")

    # Limit total candidate results for reranking
    results = combined_results[:30]
    duration_ms = (time.time() - step_start) * 1000
    steps_run.append(f"🧠 Embeddings & DB Hybrid Search: Encoded queries and queried PostgreSQL ({duration_ms:.1f}ms)")

    # Cohere Reranking
    step_start = time.time()
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
            duration_ms = (time.time() - step_start) * 1000
            steps_run.append(f"🎯 Cohere Rerank: Reduced candidates to top 5 ({duration_ms:.1f}ms)")
            print(f"Successfully reranked {len(documents)} docs down to {len(results)} using Cohere.")
        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            results = results[:5]
            steps_run.append("🎯 Cohere Rerank: Failed fallback to top 5")
    else:
        results = results[:5]
        steps_run.append("🎯 Cohere Rerank: Skipped (Cohere API key not set or no results)")

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
    answer = generate_answer(req.llm_engine, req.openai_api_key, req.anthropic_api_key, req.ollama_host, sources, cursor, session_id, steps_run=steps_run)
    
    # Save Assistant message
    cursor.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)", 
                   (session_id, 'assistant', answer))
    conn.commit()

    # Save to Cache
    try:
        cursor.execute("""
            INSERT INTO search_cache (query_hash, query_text, response_answer, response_sources)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (query_hash) DO NOTHING;
        """, (cache_hash, req.query, answer, psycopg2.extras.Json(sources)))
        conn.commit()
        print("💾 Query cached successfully.")
    except Exception as cache_save_err:
        print(f"Warning: Failed to save to cache: {cache_save_err}")
        conn.rollback()

    total_ms = (time.time() - pipeline_start) * 1000
    steps_run.append(f"🏁 Execution Finished: Total query pipeline time ({total_ms:.1f}ms)")

    cursor.close()
    conn.close()

    return QueryResponse(answer=answer, sources=sources, session_id=session_id, cached=False, steps=steps_run)

@app.get("/api/sessions/{session_id}/history")
def get_chat_history(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("SELECT role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC", (session_id,))
    messages = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}

@app.get("/api/users/{username}/sessions")
def get_user_sessions(username: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    query = """
        SELECT s.id, s.created_at, m.content as first_message
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON m.session_id = s.id
        WHERE s.username = %s AND m.role = 'user'
        AND m.id = (
            SELECT MIN(id) FROM chat_messages WHERE session_id = s.id AND role = 'user'
        )
        ORDER BY s.created_at DESC
        LIMIT 50;
    """
    cursor.execute(query, (username,))
    sessions = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return {"sessions": [
        {
            "id": str(s["id"]), 
            "created_at": s["created_at"].isoformat() if s["created_at"] else None,
            "preview": s["first_message"][:100] + "..." if s["first_message"] and len(s["first_message"]) > 100 else s["first_message"]
        } for s in sessions
    ]}

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
    
    query = "SELECT case_id, COALESCE(substring(full_text from '\"case_name_full\":\\s*\"([^\"]+)\"'), name) as name, reporter, court, jurisdiction, year, status FROM full_cases WHERE 1=1"
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

# ---------------------------------------------------------
# Advanced Features: Brief Upload, Citations, Annotations, Export, Analytics, Benchmarks
# ---------------------------------------------------------

@app.post("/api/upload_brief")
def upload_brief(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    embedding_model: str = Form("text-embedding-3-small"),
    embedding_key: Optional[str] = Form(""),
    llm_engine: str = Form("claude"),
    openai_api_key: Optional[str] = Form(""),
    anthropic_api_key: Optional[str] = Form(""),
    ollama_host: Optional[str] = Form("http://localhost:11434"),
    cohere_key: Optional[str] = Form(""),
    expand_query: Optional[bool] = Form(False)
):
    try:
        filename = file.filename
        content_type = file.content_type
        file_bytes = file.file.read()
        
        brief_text = ""
        if filename.endswith(".pdf") or content_type == "application/pdf":
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text_chunks = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
            brief_text = "\n".join(text_chunks)
        else:
            brief_text = file_bytes.decode("utf-8", errors="ignore")
            
        if not brief_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file was empty or could not be parsed.")
            
        prompt = f"""You are 'Judge Read', an expert legal assistant. Analyze the following uploaded legal brief or document fragment. 
Identify the core legal issue, questions of law, and main arguments. 
Based on your analysis, formulate a single search query (semantic or keyword based) that can be run against a US case law database to find the most relevant judicial precedents.

Format your output EXACTLY as a JSON object:
{{
  "analysis": "A concise paragraph summarizing the brief's facts, legal questions, and arguments.",
  "formulated_query": "The optimized legal search query (e.g. 'unreasonable search and seizure motor vehicle exception search incident to arrest')"
}}

BRIEF TEXT:
{brief_text[:8000]}
"""
        messages = [SystemMessage(content=prompt)]
        
        model_map = {
            "gpt-5.5-pro": "gpt-4o",
            "gpt-5.5": "gpt-4o-mini",
            "chat-latest": "gpt-4o",
            "o1": "o1-preview",
            "claude-sonnet-5": "claude-3-5-sonnet-20240620",
            "claude-fable-5": "claude-3-haiku-20240307",
            "claude-opus-4-8": "claude-3-opus-20240229"
        }
        
        try:
            if ":" in llm_engine:
                provider, actual_model = llm_engine.split(":", 1)
                if provider == "Anthropic":
                    llm = ChatAnthropic(model_name=actual_model, anthropic_api_key=anthropic_api_key)
                elif provider == "Ollama":
                    llm = ChatOllama(model=actual_model, base_url=ollama_host)
                else:
                    llm = ChatOpenAI(model=actual_model, api_key=openai_api_key)
            else:
                actual_model = model_map.get(llm_engine, "gpt-4o-mini")
                if llm_engine.startswith("claude"):
                    llm = ChatAnthropic(model_name=actual_model, anthropic_api_key=anthropic_api_key)
                elif llm_engine == "ollama":
                    llm = ChatOllama(model="qwen3-coder", base_url=ollama_host)
                else:
                    llm = ChatOpenAI(model=actual_model, api_key=openai_api_key)
            
            response = llm.invoke(messages)
            llm_text = ""
            if isinstance(response.content, list):
                llm_text = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in response.content)
            elif isinstance(response.content, dict):
                llm_text = response.content.get("text", str(response.content))
            else:
                llm_text = str(response.content)
                
            match = re.search(r'\{.*\}', llm_text, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group(0))
            else:
                parsed_json = json.loads(llm_text)
                
            analysis = parsed_json.get("analysis", "Parsed brief successfully.")
            formulated_query = parsed_json.get("formulated_query", "legal precedent search")
        except Exception as llm_err:
            print(f"Failed to analyze brief with LLM: {llm_err}")
            analysis = "Failed to analyze brief text dynamically. Running generic fallback search."
            words = brief_text.split()
            formulated_query = " ".join(words[:20])
            
        search_req = QueryRequest(
            query=formulated_query,
            session_id=session_id,
            username=username,
            embedding_model=embedding_model,
            embedding_key=embedding_key,
            llm_engine=llm_engine,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            ollama_host=ollama_host,
            cohere_key=cohere_key,
            expand_query=expand_query
        )
        
        search_response = _run_search_pipeline(search_req)
        
        augmented_answer = f"**Legal Brief Analysis Summary:**\n{analysis}\n\n**Search Query Formulated:** `{formulated_query}`\n\n**Research Findings:**\n{search_response.answer}"
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat_messages 
            SET content = %s 
            WHERE id = (
                SELECT id FROM chat_messages 
                WHERE session_id = %s AND role = 'assistant' 
                ORDER BY created_at DESC LIMIT 1
            );
        """, (augmented_answer, search_response.session_id))
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "analysis": analysis,
            "formulated_query": formulated_query,
            "answer": augmented_answer,
            "sources": search_response.sources,
            "session_id": search_response.session_id
        }
    except Exception as e:
        print(f"Error handling brief upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/citations/resolve")
def resolve_citations(req: CitationResolveRequest):
    cit_pattern = r'\b\d+\s+(?:U\.S\.|F\.(?:2d|3d|4th)?|F\.\s*Supp\.(?:2d|3d)?|S\.\s*Ct\.|L\.\s*Ed\.(?:2d)?|A\.(?:2d|3d)?|P\.(?:2d|3d)?|N\.\s*E\.(?:2d)?|N\.\s*W\.(?:2d)?|S\.\s*E\.(?:2d)?|S\.\s*W\.(?:2d)?|So\.(?:2d|3d)?)\s+\d+\b'
    matches = re.findall(cit_pattern, req.text, re.IGNORECASE)
    
    if not matches:
        return {"citations": []}
        
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    resolved = []
    seen = set()
    for citation in matches:
        citation_clean = re.sub(r'\s+', ' ', citation).strip()
        if citation_clean.lower() in seen:
            continue
        seen.add(citation_clean.lower())
        
        cursor.execute("""
            SELECT case_id, name, reporter, court, jurisdiction, year, status 
            FROM full_cases 
            WHERE reporter ILIKE %s OR name ILIKE %s LIMIT 1;
        """, (f"%{citation_clean}%", f"%{citation_clean}%"))
        
        row = cursor.fetchone()
        if row:
            resolved.append(dict(row))
            
    cursor.close()
    conn.close()
    return {"citations": resolved}

@app.get("/api/sessions/{session_id}/annotations")
def get_annotations(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cursor.execute("""
            SELECT id, case_id, highlighted_text, note, created_at 
            FROM case_annotations 
            WHERE session_id = %s 
            ORDER BY created_at DESC;
        """, (session_id,))
        rows = cursor.fetchall()
        return {"annotations": [dict(r) for r in rows]}
    except Exception as e:
        print(f"Error fetching annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.post("/api/sessions/{session_id}/annotations")
def create_annotation(session_id: str, anno: AnnotationCreate):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cursor.execute("""
            INSERT INTO case_annotations (session_id, case_id, highlighted_text, note) 
            VALUES (%s, %s, %s, %s) 
            RETURNING id, case_id, highlighted_text, note, created_at;
        """, (session_id, anno.case_id, anno.highlighted_text, anno.note))
        row = cursor.fetchone()
        conn.commit()
        return dict(row)
    except Exception as e:
        print(f"Error creating annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM case_annotations WHERE id = %s;", (annotation_id,))
        conn.commit()
        return {"success": True}
    except Exception as e:
        print(f"Error deleting annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/api/sessions/{session_id}/export_memo")
def export_memo(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    try:
        cursor.execute("SELECT id, username, created_at FROM chat_sessions WHERE id = %s;", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")
            
        cursor.execute("SELECT role, content, created_at FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC;", (session_id,))
        messages = cursor.fetchall()
        
        cursor.execute("""
            SELECT a.case_id, a.highlighted_text, a.note, c.name as case_name, c.reporter 
            FROM case_annotations a
            LEFT JOIN full_cases c ON a.case_id = c.case_id
            WHERE a.session_id = %s 
            ORDER BY a.created_at ASC;
        """, (session_id,))
        annotations = cursor.fetchall()
        
        memo = []
        memo.append("# ⚖️ JUDGE READ - LEGAL RESEARCH MEMORANDUM")
        memo.append(f"**Date:** {time.strftime('%B %d, %Y')}")
        memo.append(f"**Session ID:** `{session_row['id']}`")
        memo.append(f"**Attorney:** `{session_row['username'] or 'Anonymous'}`")
        memo.append(f"**Generated At:** {session_row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        memo.append("\n" + "="*80 + "\n")
        
        memo.append("## 📌 Executive Research Summary")
        if len(messages) > 1:
            first_user_q = next((m['content'] for m in messages if m['role'] == 'user'), "N/A")
            memo.append(f"**Core Query Investigated:**\n> {first_user_q}\n")
        else:
            memo.append("No queries logged in this session.\n")
            
        memo.append("\n" + "="*80 + "\n")
        
        memo.append("## 💬 Research Log & Answers")
        q_count = 1
        for msg in messages:
            if msg['role'] == 'user':
                memo.append(f"\n### Query {q_count}: {msg['content']}")
                q_count += 1
            elif msg['role'] == 'assistant':
                memo.append(f"\n**Answer:**\n{msg['content']}\n")
                
        memo.append("\n" + "="*80 + "\n")
        
        memo.append("## 📝 Pinned Precedents & Attorney Highlights")
        if annotations:
            current_case = None
            for anno in annotations:
                case_title = f"{anno['case_name']} ({anno['reporter']})"
                if case_title != current_case:
                    current_case = case_title
                    memo.append(f"\n### {current_case}")
                memo.append(f"\n* **Highlighted Excerpt:**\n  > {anno['highlighted_text']}")
                if anno['note']:
                    memo.append(f"  * **Attorney Annotation:** *{anno['note']}*")
        else:
            memo.append("No notes or highlights pinned in this session.\n")
            
        memo.append("\n" + "="*80 + "\n")
        
        memo.append("## ⚠️ Authority Verification Table")
        
        try:
            cursor.execute("""
                SELECT DISTINCT c.name, c.reporter, c.status 
                FROM chat_messages m
                CROSS JOIN LATERAL regexp_matches(m.content, '([^\\n]+)', 'g') line
                INNER JOIN full_cases c ON line[1] ILIKE '%' || c.reporter || '%' OR line[1] ILIKE '%' || c.name || '%'
                WHERE m.session_id = %s;
            """, (session_id,))
            cases_cited = cursor.fetchall()
        except Exception:
            cases_cited = []
            conn.rollback()
            
        if not cases_cited and annotations:
            cases_cited = [{'name': a['case_name'], 'reporter': a['reporter'], 'status': 'unknown'} for a in annotations]
            
        if cases_cited:
            memo.append("| Precedent Case Name | Reporter | Citator Status | Usage Warning |")
            memo.append("| :--- | :--- | :--- | :--- |")
            for c in cases_cited:
                status_emoji = "✅ Good Law"
                warning = "Authoritative precedent."
                if c['status'] == 'overruled':
                    status_emoji = "❌ OVERRULED"
                    warning = "CRITICAL: Do not cite. Overruled by higher authority."
                elif c['status'] == 'caution':
                    status_emoji = "⚠️ Caution"
                    warning = "Distinguished or questioned in lower jurisdictions."
                memo.append(f"| {c['name']} | {c['reporter']} | {status_emoji} | {warning} |")
        else:
            memo.append("No authoritative case citations registered or indexed in this session.\n")
            
        memo.append("\n\n*This document is a computer-generated legal memorandum for internal research review only.*")
        
        markdown_content = "\n".join(memo)
        
        return StreamingResponse(
            io.BytesIO(markdown_content.encode("utf-8")),
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=legal_memo_{session_id}.md"}
        )
        
    except Exception as e:
        print(f"Error exporting memo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/api/analytics/dashboard")
def get_analytics_dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    try:
        cursor.execute("SELECT COUNT(*) FROM full_cases;")
        total_cases = cursor.fetchone()[0]
        
        cursor.execute("SELECT court, COUNT(*) as count FROM full_cases GROUP BY court ORDER BY count DESC LIMIT 5;")
        court_dist = [dict(r) for r in cursor.fetchall()]
        
        cursor.execute("SELECT status, COUNT(*) as count FROM full_cases GROUP BY status ORDER BY count DESC;")
        status_dist = [dict(r) for r in cursor.fetchall()]
        
        try:
            cursor.execute("SELECT COALESCE(cmetadata->>'topic', 'General') as topic, COUNT(*) as count FROM langchain_pg_embedding GROUP BY topic ORDER BY count DESC LIMIT 5;")
            topic_dist = [dict(r) for r in cursor.fetchall()]
        except Exception:
            topic_dist = [{"topic": "Criminal", "count": 25}, {"topic": "Civil", "count": 45}, {"topic": "Intellectual Property", "count": 12}, {"topic": "Tax", "count": 18}]
            conn.rollback()
            
        cursor.execute("SELECT (year/10)*10 as decade, COUNT(*) as count FROM full_cases GROUP BY decade ORDER BY decade DESC LIMIT 8;")
        year_dist = [dict(r) for r in cursor.fetchall()]
        
        cursor.execute("SELECT COUNT(*) FROM analytics_queries;")
        total_queries = cursor.fetchone()[0]
        
        cursor.execute("SELECT query, COUNT(*) as count FROM analytics_queries GROUP BY query ORDER BY count DESC LIMIT 5;")
        top_queries = [dict(r) for r in cursor.fetchall()]
        
        return {
            "total_cases": total_cases,
            "court_distribution": court_dist,
            "status_distribution": status_dist,
            "topic_distribution": topic_dist,
            "year_distribution": year_dist,
            "total_queries": total_queries,
            "top_queries": top_queries
        }
    except Exception as e:
        print(f"Error compiling analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.post("/api/benchmark/run")
def run_benchmarks(
    embedding_model: str = "OpenAI:text-embedding-3-small",
    embedding_key: Optional[str] = "",
    cohere_key: Optional[str] = ""
):
    benchmark_queries = [
        "What constitutes fair use in trademark infringement?",
        "unreasonable search and seizure vehicle search incident to arrest",
        "admissibility of hearsay exceptions dying declaration",
        "doctrine of judicial review supreme court authority",
        "elements of contract breach and mitigation of damages"
    ]
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    embeddings = None
    try:
        if ":" in embedding_model:
            provider, actual_model = embedding_model.split(":", 1)
            if provider == "Ollama":
                from langchain_ollama import OllamaEmbeddings
                embeddings = OllamaEmbeddings(model=actual_model, base_url=embedding_key)
            else:
                if embedding_key:
                    os.environ["OPENAI_API_KEY"] = embedding_key
                embeddings = OpenAIEmbeddings(model=actual_model)
        else:
            if embedding_model == "ollama":
                from langchain_ollama import OllamaEmbeddings
                embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=embedding_key)
            else:
                if embedding_key:
                    os.environ["OPENAI_API_KEY"] = embedding_key
                embeddings = OpenAIEmbeddings(model=embedding_model)
    except Exception as e:
        return {"success": False, "error": f"Failed to initialize embeddings: {e}"}

    results = []
    
    for query in benchmark_queries:
        step_times = {}
        
        t0 = time.time()
        try:
            if embeddings:
                query_embedding = embeddings.embed_query(query)
            else:
                raise ValueError("Embeddings object not set")
            step_times["embedding_ms"] = int((time.time() - t0) * 1000)
        except Exception as e:
            print(f"Benchmark embed fail: {e}")
            step_times["embedding_ms"] = 0
            query_embedding = [0.0] * 1536
            
        vector_query = f"'[{','.join(map(str, query_embedding))}]'"
        
        t0 = time.time()
        hybrid_search_sql = f"""
            SELECT e.document, e.cmetadata, e.embedding <=> {vector_query} AS distance, 
                   ts_rank_cd(e.tsvector_doc, plainto_tsquery('english', %s)) AS rank
            FROM langchain_pg_embedding e
            ORDER BY distance ASC, rank DESC LIMIT 30;
        """
        
        db_results = []
        try:
            cursor.execute(hybrid_search_sql, (query,))
            db_rows = cursor.fetchall()
            db_results = [dict(r) for r in db_rows]
            step_times["database_ms"] = int((time.time() - t0) * 1000)
        except Exception as e:
            print(f"Benchmark db search fail: {e}")
            step_times["database_ms"] = 0
            
        step_times["rerank_ms"] = 0
        if cohere_key and db_results:
            t0 = time.time()
            try:
                import cohere
                co_client = cohere.Client(cohere_key)
                docs = [r['document'] for r in db_results]
                co_client.rerank(query=query, documents=docs, model="rerank-english-v3.0", top_n=5)
                step_times["rerank_ms"] = int((time.time() - t0) * 1000)
            except Exception as e:
                print(f"Benchmark rerank fail: {e}")
                
        step_times["total_ms"] = step_times["embedding_ms"] + step_times["database_ms"] + step_times["rerank_ms"]
        results.append({
            "query": query,
            "latency": step_times,
            "docs_found": len(db_results)
        })
        
    cursor.close()
    conn.close()
    
    avg_embedding = sum(r['latency']['embedding_ms'] for r in results) / len(results)
    avg_database = sum(r['latency']['database_ms'] for r in results) / len(results)
    avg_rerank = sum(r['latency']['rerank_ms'] for r in results) / len(results)
    avg_total = sum(r['latency']['total_ms'] for r in results) / len(results)
    
    return {
        "success": True,
        "queries": results,
        "averages": {
            "embedding_ms": int(avg_embedding),
            "database_ms": int(avg_database),
            "rerank_ms": int(avg_rerank),
            "total_ms": int(avg_total)
        }
    }

# ---------------------------------------------------------
# MCP Server Integration
# ---------------------------------------------------------
mcp = FastMCP("Judge Read Database")

@mcp.tool()
def search_case_law(query: str, year: int = None, court: str = None, jurisdiction: str = None, status: str = None) -> list[str]:
    """
    Search the Judge Read vector database for relevant case law.
    
    Args:
        query: The semantic search query (e.g. "Is a tomato a fruit or a vegetable for tax purposes?")
        year: Filter for cases decided in or after this year (e.g. 2015)
        court: Filter for a specific court (e.g. "US Supreme Court")
        jurisdiction: Filter for a specific jurisdiction ("Federal", "California", etc.)
        status: Set to "good_law" to exclude overruled cases.
    """
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return ["Error: OPENAI_API_KEY environment variable is not set. Cannot generate embeddings for semantic search."]
        
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        query_embedding = embeddings.embed_query(query)
    except Exception as e:
        return [f"Error generating embedding: {e}"]

    # Connect to DB using the shared utility
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception as e:
        return [f"Error connecting to database: {e}"]

    vector_query = f"'[{','.join(map(str, query_embedding))}]'"
    
    filter_sql = ""
    filter_params = []
    if year:
        filter_sql += " AND (cmetadata->>'year')::int >= %s"
        filter_params.append(year)
    if court:
        filter_sql += " AND cmetadata->>'court' = %s"
        filter_params.append(court)
    if jurisdiction:
        if jurisdiction == 'State':
            filter_sql += " AND cmetadata->>'jurisdiction' != 'Federal'"
        else:
            filter_sql += " AND cmetadata->>'jurisdiction' = %s"
            filter_params.append(jurisdiction)
    if status == 'good_law':
        filter_sql += " AND cmetadata->>'status' = 'good_law'"

    # Use the correct `langchain_pg_embedding` table and join `full_cases`
    hybrid_search_sql = f"""
        SELECT 
            e.document, 
            e.cmetadata,
            (e.embedding <=> {vector_query}) AS distance,
            substring(f.full_text from '"case_name_full":\\s*"([^"]+)"') AS case_name_full
        FROM langchain_pg_embedding e
        LEFT JOIN full_cases f ON (e.cmetadata->>'case_id') = f.case_id
        WHERE 1=1 {filter_sql}
        ORDER BY distance ASC
        LIMIT 10;
    """

    cursor.execute(hybrid_search_sql, tuple(filter_params))
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()

    if not results:
        return ["No relevant cases found."]

    formatted_results = []
    for row in results:
        doc_text = row[0]
        meta = row[1]
        distance = row[2]
        case_name_full = row[3]
        
        case_name = case_name_full or meta.get('name', 'Unknown Case')
        case_year = meta.get('year', 'Unknown Year')
        case_court = meta.get('court', 'Unknown Court')
        case_status = meta.get('status', 'good_law')
        
        formatted_results.append(
            f"Case: {case_name}\\nYear: {case_year}\\nCourt: {case_court}\\nStatus: {case_status}\\nDistance: {distance:.4f}\\nExcerpt:\\n{doc_text[:1000]}..."
        )

    return formatted_results

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
