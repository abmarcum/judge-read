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
        tables_to_check = ["full_cases", "langchain_pg_embedding", "chat_messages"]
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
    
    ollama_ok = test_ollama(config)
    print("-" * 75)
    
    mcp_ok = test_mcp_server()
    print("-" * 75)
    
    database_ok = test_database(host=db_host, port=db_port, user=db_user, password=db_pass, dbname=db_name)
    
    print("="*75)
    if frontend_ok and backend_ok and database_ok and ollama_ok and mcp_ok:
        print("🎉 ALL TESTS PASSED! Your full stack is online and fully functional.")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED. Please review the logs above for specific component failures.")
        sys.exit(1)
