import urllib.request
import urllib.error
import psycopg2
import sys
import time
import json
import os
import argparse
import re

def print_result(name, success, message=""):
    status = f"✅ SUCCESS {message}".strip() if success else f"❌ FAILED {message}".strip()
    print(f"{name:<50} {status}")
    return success

def test_frontend(host, port=5173):
    url = f"http://{host}:{port}/"
    success = True
    try:
        req = urllib.request.urlopen(url, timeout=5)
        if req.getcode() == 200:
            print_result(f"Frontend UI Reachable ({url})", True)
            
            # Asset Pipeline Verification
            html = req.read().decode('utf-8')
            if '<script type="module"' in html or 'vite' in html.lower() or '<div id="root">' in html:
                print_result("Frontend UI HTML/Asset Payload", True)
            else:
                print_result("Frontend UI HTML/Asset Payload", False, "(Missing standard React/Vite tags)")
                success = False
        else:
            print_result(f"Frontend UI Reachable ({url})", False, f"(HTTP {req.getcode()})")
            success = False
    except Exception as e:
        print_result(f"Frontend UI Reachable ({url})", False, f"({e})")
        success = False
    return success

def test_backend(host, port=8000):
    base_url = f"http://{host}:{port}"
    success = True
    
    # Test Config Endpoint
    try:
        req = urllib.request.urlopen(f"{base_url}/api/config", timeout=5)
        if req.getcode() == 200:
            print_result("Backend Config API Reachable", True)
        else:
            print_result("Backend Config API Reachable", False, f"(HTTP {req.getcode()})")
            success = False
    except Exception as e:
        print_result("Backend Config API Reachable", False, f"({e})")
        success = False

    # Test Live Cases Endpoint
    try:
        req = urllib.request.urlopen(f"{base_url}/api/cases?limit=1", timeout=5)
        if req.getcode() == 200:
            print_result("Backend Cases DB Query Endpoint", True)
        else:
            print_result("Backend Cases DB Query Endpoint", False, f"(HTTP {req.getcode()})")
            success = False
    except Exception as e:
        print_result("Backend Cases DB Query Endpoint", False, f"({e})")
        success = False

    # Test CORS headers
    try:
        request = urllib.request.Request(f"{base_url}/api/cases", method="OPTIONS")
        request.add_header("Origin", "http://localhost:5173")
        request.add_header("Access-Control-Request-Method", "GET")
        response = urllib.request.urlopen(request, timeout=5)
        headers = {k.lower(): v for k, v in response.getheaders()}
        if "access-control-allow-origin" in headers:
            print_result("Backend CORS Configuration", True)
        else:
            print_result("Backend CORS Configuration", False, "(Missing Access-Control-Allow-Origin)")
            success = False
    except Exception as e:
        print_result("Backend CORS Configuration", False, f"({e})")
        success = False

    return success

def test_database(host, port, user, password, dbname):
    success = True
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Connection
        cursor.execute("SELECT 1;")
        if cursor.fetchone()[0] == 1:
            print_result(f"Database Connection ({host}:{port})", True)
        else:
            print_result(f"Database Connection ({host}:{port})", False)
            success = False

        # Extension Check
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone():
            print_result("pgvector Extension Enabled", True)
        else:
            print_result("pgvector Extension Enabled", False, "(Extension not installed)")
            success = False

        # Tables Check
        tables_to_check = ["full_cases", "langchain_pg_embedding", "chat_messages", "search_cache"]
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        missing_tables = [t for t in tables_to_check if t not in existing_tables]
        if not missing_tables:
            print_result("Required Schema Tables Exist", True)
        else:
            print_result("Required Schema Tables Exist", False, f"(Missing: {', '.join(missing_tables)})")
            success = False

        # Data volume check
        if "full_cases" in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM full_cases;")
            count = cursor.fetchone()[0]
            if count > 0:
                print_result("Database Ingestion Status", True)
                print_result("Total Cases In Database", True, f"({count} cases)")
            else:
                print_result("Database Ingestion Status", False, "(0 cases found, run data_pipeline.py)")
                success = False

        # FTS Verification
        cursor.execute("SELECT ts_rank_cd(to_tsvector('english', 'judge read system'), plainto_tsquery('english', 'judge'));")
        fts_result = cursor.fetchone()
        if fts_result and fts_result[0] > 0:
            print_result("Full-Text Search (FTS) Verification", True)
        else:
            print_result("Full-Text Search (FTS) Verification", False, "(Failed to compute rank)")
            success = False

        cursor.close()
        conn.close()
    except Exception as e:
        print_result(f"Database Connection ({host}:{port})", False, f"({e})")
        success = False

    return success

def test_ollama(config):
    embedding_key = config.get("embeddingKey", "")
    embedding_model = config.get("embeddingModel", "")
    
    if embedding_model != "ollama" or not embedding_key:
        print_result("Ollama Connectivity", True, "(Skipped: Not using Ollama in config)")
        return True

    success = True
    # If the user put the base_url like http://192.168.1.159:11434
    if not embedding_key.endswith("/"):
        embedding_key += "/"
    url = f"{embedding_key}api/tags"
    
    try:
        req = urllib.request.urlopen(url, timeout=5)
        if req.getcode() == 200:
            print_result(f"Ollama Connectivity ({embedding_key})", True)
        else:
            print_result(f"Ollama Connectivity ({embedding_key})", False, f"(HTTP {req.getcode()})")
            success = False
    except Exception as e:
        print_result(f"Ollama Connectivity ({embedding_key})", False, f"({e})")
        success = False

    return success

def test_mcp_server():
    success = True
    try:
        from main import mcp, search_case_law
        print_result("MCP Server Object Initialized", True)
        
        if getattr(mcp, "name", "") == "Judge Read Database":
            print_result("MCP Server Name Verified", True)
        else:
            print_result("MCP Server Name Verified", False, "(Incorrect FastMCP name)")
            success = False
            
        if callable(search_case_law):
            print_result("MCP Tool 'search_case_law' Registered", True)
        else:
            print_result("MCP Tool 'search_case_law' Registered", False, "(Function not callable)")
            success = False
            
    except ImportError as e:
        print_result("MCP Server Integration", False, f"({e})")
        success = False
    except Exception as e:
        print_result("MCP Server Validation", False, f"({e})")
        success = False
        
    return success

def test_search_pipeline_features(host, port=8000):
    success = True
    base_url = f"http://{host}:{port}/api/search"
    
    # We will send a test search request
    payload = {
        "query": f"test case prior art obviousness {time.time()}",
        "username": "diagnostics_test_user",
        "llm_engine": "OpenAI:gpt-5.5-pro",
        "openai_api_key": "sk-mock-key", # The server handles fallbacks or mocks if config key is active
        "embedding_model": "Ollama:nomic-embed-text",
        "embedding_key": "http://localhost:11434",
        "cohere_key": "",
        "expand_query": False
    }
    
    # Load valid keys from config.json if they exist
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                payload["openai_api_key"] = cfg.get("openaiApiKey", payload["openai_api_key"])
                payload["embedding_key"] = cfg.get("embeddingKey", payload["embedding_key"])
                payload["embedding_model"] = cfg.get("embeddingModel", payload["embedding_model"])
        except Exception:
            pass

    # Helper to parse Server-Sent Events stream from response
    def parse_sse_response(resp):
        result = {}
        steps = []
        try:
            content = resp.read().decode('utf-8')
        except Exception:
            return {}
            
        if not content:
            return {}
            
        # Handle legacy raw JSON response format
        if content.strip().startswith("{"):
            try:
                return json.loads(content)
            except Exception:
                pass
                
        # Handle SSE stream format
        lines = content.split("\n")
        for line_str in lines:
            line_str = line_str.strip()
            if line_str.startswith("data: "):
                try:
                    data = json.loads(line_str[6:])
                    if data.get("type") == "step":
                        steps.append(data.get("step"))
                    elif data.get("type") == "result":
                        result = data
                except Exception:
                    pass
        if result:
            if "steps" not in result or not result["steps"]:
                result["steps"] = steps
        return result

    # First request: Cache Miss
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(base_url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        
        step_start = time.time()
        response = urllib.request.urlopen(req, timeout=600)
        duration = time.time() - step_start
        
        if response.getcode() == 200:
            res_payload = parse_sse_response(response)
            
            # Verify fields
            has_answer = "answer" in res_payload
            has_steps = "steps" in res_payload and isinstance(res_payload["steps"], list)
            has_cached = "cached" in res_payload
            
            if has_answer and has_steps and has_cached:
                print_result(f"Search API Pipeline Miss Check ({duration:.1f}s)", True)
                print_result("Search Response Trace Steps Returned", True, f"({len(res_payload['steps'])} trace logs found)")
                
                # Check for agent steps
                has_agents = any("Attorney Agent" in step or "Judge Agent" in step for step in res_payload["steps"])
                print_result("Attorney/Judge Multi-Agent Trace Logged", has_agents)
            else:
                print_result("Search API Pipeline Miss Check", False, "(Missing payload fields)")
                success = False
                return False
                
            # Second request: Cache Hit Check
            step_start = time.time()
            # Send same query again
            response_hit = urllib.request.urlopen(req, timeout=120)
            duration_hit = time.time() - step_start
            
            if response_hit.getcode() == 200:
                res_payload_hit = parse_sse_response(response_hit)
                is_cached = res_payload_hit.get("cached") == True
                
                if is_cached:
                    print_result(f"Search API Pipeline Caching Hit Check ({duration_hit * 1000:.1f}ms)", True)
                    
                    has_cache_step = any("Checking query signature" in s for s in res_payload_hit.get("steps", []))
                    print_result("Pipeline Cache HIT Trace Logged", has_cache_step)
                else:
                    print_result("Search API Pipeline Caching Hit Check", False, "(Returned cached=False on repeat query)")
                    success = False
            else:
                print_result("Search API Pipeline Caching Hit Check", False, f"(HTTP {response_hit.getcode()})")
                success = False
                
        else:
            print_result("Search API Pipeline Integration Test", False, f"(HTTP {response.getcode()})")
            success = False
    except Exception as e:
        print_result("Search API Pipeline Integration Test", False, f"({e})")
        success = False
        
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Diagnostic Tests for Judge Read")
    parser.add_argument("--ui-ip", type=str, help="IP address of the Frontend UI")
    parser.add_argument("--backend-ip", type=str, help="IP address of the Backend API")
    args = parser.parse_args()

    print("="*75)
    print("Starting Advanced Judge Read Diagnostic Tests")
    print("="*75)
    
    # Load config.json
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read config.json: {e}")
        sys.exit(1)
        
    try:
        db_host = config["pgHost"]
        db_port = config["pgPort"]
        db_user = config["pgUser"]
        db_pass = config["pgPassword"]
        db_name = config["pgDb"]
    except KeyError as e:
        print(f"❌ Missing required database configuration in config.json: {e}")
        sys.exit(1)
    
    # Use CLI args if provided, otherwise default to the database host from config
    ui_host = args.ui_ip if args.ui_ip else db_host
    backend_host = args.backend_ip if args.backend_ip else db_host
    
    frontend_ok = test_frontend(host=ui_host)
    print("-" * 75)
    
    backend_ok = test_backend(host=backend_host)
    print("-" * 75)
    
    # Run new pipeline features tests (Cache hits & trace logs steps)
    pipeline_features_ok = test_search_pipeline_features(host=backend_host)
    print("-" * 75)
    
    ollama_ok = test_ollama(config)
    print("-" * 75)
    
    mcp_ok = test_mcp_server()
    print("-" * 75)
    
    database_ok = test_database(host=db_host, port=db_port, user=db_user, password=db_pass, dbname=db_name)
    
    print("="*75)
    if frontend_ok and backend_ok and pipeline_features_ok and database_ok and ollama_ok and mcp_ok:
        print("🎉 ALL TESTS PASSED! Your full stack is online and fully functional.")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED. Please review the logs above for specific component failures.")
        sys.exit(1)
