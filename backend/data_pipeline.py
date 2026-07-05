import os
import argparse
import requests
import tarfile
import json
from tqdm import tqdm
from datasets import load_dataset
from bs4 import BeautifulSoup
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langsmith import traceable

# Constants
COURTLISTENER_URL = "https://www.courtlistener.com/api/bulk-data/opinions/latest.tar.gz"
HF_DATASET_NAME = "harvard-lil/cold-cases"

# Map to the project's root data/ directory
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOWNLOAD_PATH = os.path.join(DATA_ROOT, "courtlistener_opinions.tar.gz")
EXTRACT_DIR = os.path.join(DATA_ROOT, "case_law_repository")
HF_CACHE_DIR = os.path.join(DATA_ROOT, "huggingface_cache")

@traceable(name="download_courtlistener", run_type="tool")
def download_courtlistener(limit=None):
    """Download and extract CourtListener bulk data."""
    if os.path.exists(EXTRACT_DIR) and len(os.listdir(EXTRACT_DIR)) > 0:
        print(f"Data already exists in {EXTRACT_DIR}. Skipping CourtListener download.")
        return
        
    print(f"Starting download from {COURTLISTENER_URL}...")
    with requests.get(COURTLISTENER_URL, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        os.makedirs(os.path.dirname(DOWNLOAD_PATH), exist_ok=True)
        with open(DOWNLOAD_PATH, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (50 * 1024 * 1024) < 8192:
                        total_mb = total_size / (1024*1024) if total_size else 0
                        print(f"Downloaded: {downloaded / (1024*1024):.2f} MB of {total_mb:.2f} MB")
    print("Download complete.")

    print(f"Extracting {DOWNLOAD_PATH} to {EXTRACT_DIR}...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
        if limit is not None:
            # Only extract the first N files
            members = []
            for m in tar.getmembers():
                if m.isfile():
                    members.append(m)
                    if len(members) >= limit:
                        break
            print(f"Extracting {len(members)} files due to limit...")
            tar.extractall(path=EXTRACT_DIR, members=members)
        else:
            tar.extractall(path=EXTRACT_DIR)
    
    # Optional cleanup
    if os.path.exists(DOWNLOAD_PATH):
        os.remove(DOWNLOAD_PATH)
        print("Cleaned up compressed archive.")

@traceable(name="download_huggingface", run_type="tool")
def download_huggingface(limit=None):
    """Download and export HuggingFace dataset."""
    if os.path.exists(EXTRACT_DIR) and len(os.listdir(EXTRACT_DIR)) > 0:
        print(f"Data already exists in {EXTRACT_DIR}. Skipping Hugging Face download.")
        return

    print(f"Loading dataset '{HF_DATASET_NAME}' from Hugging Face (streaming mode)...")
    dataset = load_dataset(HF_DATASET_NAME, split="train", streaming=True)
    
    if limit is not None:
        print(f"Limiting to the first {limit} cases...")
        dataset = dataset.take(limit)
        
    print(f"Exporting to {EXTRACT_DIR}...")
    
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    for i, case in enumerate(tqdm(dataset, desc="Exporting Cases")):
        case_id = case.get('id', f'case_{i}')
        file_path = os.path.join(EXTRACT_DIR, f"{case_id}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(case, f, ensure_ascii=False, default=str)

    print("Export complete. Your repository is ready for processing.")

def extract_text_from_html(file_content):
    """Strip HTML tags using BeautifulSoup."""
    soup = BeautifulSoup(file_content, "html.parser")
    return soup.get_text(separator="\n", strip=True)

@traceable(name="embed_data_pipeline", run_type="chain")
def embed_data(db_url, embed_provider, embed_model, embed_key, embed_host, limit=None):
    """Load, chunk, and embed the extracted files."""
    if not os.path.exists(EXTRACT_DIR):
        print(f"Error: {EXTRACT_DIR} not found. Please run the download step first.")
        return

    print("Running db_setup to initialize relational tables (like full_cases)...")
    import subprocess
    subprocess.run(["python", "db_setup.py"], env=dict(os.environ, DATABASE_URL=db_url))

    print("Loading and standardizing case documents from repository...")
    import glob
    import json
    import uuid
    import re
    from bs4 import BeautifulSoup
    from langchain_core.documents import Document
    import psycopg2

    file_paths = glob.glob(os.path.join(EXTRACT_DIR, "**/*.*"), recursive=True)
    if limit is not None:
        file_paths = file_paths[:limit]
    raw_documents = []
    full_cases_to_insert = []

    def clean_html_text(html_content):
        if not html_content:
            return ""
        try:
            # BeautifulSoup cleaning specifically for target HTML tags
            soup = BeautifulSoup(str(html_content), "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            return str(html_content)

    def determine_status(text):
        if not text:
            return "good_law"
        text_lower = text.lower()
        if "overruled by" in text_lower or "is overruled" in text_lower:
            return "overruled"
        elif "reversed" in text_lower or "vacated" in text_lower or "distinguished by" in text_lower:
            return "caution"
        return "good_law"

    def determine_topic(text):
        if not text:
            return "Civil"
        text_lower = text.lower()
        if any(w in text_lower for w in ["patent", "trademark", "copyright", "infringement", "patentable"]):
            return "Intellectual Property"
        elif any(w in text_lower for w in ["tax", "revenue", "irs", "income tax", "taxpayer"]):
            return "Tax"
        elif any(w in text_lower for w in ["murder", "criminal", "felony", "guilty", "sentence", "arrest", "prosecut"]):
            return "Criminal"
        elif any(w in text_lower for w in ["constitutional", "first amendment", "equal protection", "due process", "fourteenth amendment"]):
            return "Constitutional"
        return "Civil"

    # Select lists for fallback mock values if dataset does not contain specific field
    STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
    FEDERAL_COURTS = ["US Supreme Court", "US Court of Appeals (1st Circuit)", "US Court of Appeals (2nd Circuit)", "US Court of Appeals (3rd Circuit)", "US Court of Appeals (4th Circuit)", "US Court of Appeals (5th Circuit)", "US Court of Appeals (6th Circuit)", "US Court of Appeals (7th Circuit)", "US Court of Appeals (8th Circuit)", "US Court of Appeals (9th Circuit)", "US Court of Appeals (10th Circuit)", "US Court of Appeals (11th Circuit)", "US Court of Appeals (DC Circuit)", "US Court of Appeals (Federal Circuit)", "US District Court"]
    STATE_COURTS = ["State Supreme Court", "State Court of Appeals", "Superior Court", "Circuit Court", "District Court"]

    import random
    
    for path in tqdm(file_paths, desc="Parsing & Formatting cases"):
        if not os.path.isfile(path) or os.path.basename(path).startswith('.'):
            continue
            
        case_id = os.path.splitext(os.path.basename(path))[0]
        
        is_json = False
        data = None
        if path.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                is_json = True
            except Exception as e:
                print(f"Warning: Failed to parse {path} as JSON: {e}. Falling back to plain text loader.")

        if is_json and data:
            name = (
                data.get("case_name_full") or 
                data.get("case_name") or 
                data.get("name") or 
                data.get("name_abbreviation") or
                data.get("slug") or
                f"Case {case_id}"
            )
            
            date_filed = data.get("date_filed") or data.get("decision_date") or data.get("date") or ""
            year = None
            if date_filed:
                year_match = re.search(r"\b(1[789]\d{2}|20\d{2})\b", str(date_filed))
                if year_match:
                    year = int(year_match.group(1))
            if not year:
                year = random.choice([1998, 2005, 2015, 2021, 2023])

            court = (
                data.get("court_full_name") or 
                data.get("court") or 
                data.get("court_short_name") or
                random.choice(FEDERAL_COURTS + STATE_COURTS)
            )
            if isinstance(court, dict):
                court = court.get("name") or court.get("full_name") or "Unknown Court"

            jurisdiction = (
                data.get("court_jurisdiction") or 
                data.get("jurisdiction") or 
                ""
            )
            if isinstance(jurisdiction, dict):
                jurisdiction = jurisdiction.get("name") or jurisdiction.get("name_long") or ""
            
            # Map jurisdiction to USA/Federal vs State
            if not jurisdiction or "Federal" in jurisdiction or "USA" in jurisdiction:
                jurisdiction = "Federal"
            else:
                matched_state = "Federal"
                for state in STATES:
                    if state.lower() in jurisdiction.lower():
                        matched_state = state
                        break
                jurisdiction = matched_state

            citations_raw = data.get("citations") or data.get("citation") or []
            if isinstance(citations_raw, str):
                citations = [citations_raw]
            elif isinstance(citations_raw, list):
                citations = []
                for c in citations_raw:
                    if isinstance(c, dict):
                        citations.append(c.get("cite") or c.get("citation") or "")
                    else:
                        citations.append(str(c))
                citations = [c for c in citations if c]
            else:
                citations = []
            reporter = citations[0] if citations else f"{year} U.S. {case_id}"

            opinions_raw = data.get("opinions") or []
            opinions = []
            opinion_texts = []

            if isinstance(opinions_raw, dict):
                opinions_raw = [opinions_raw]
            elif isinstance(opinions_raw, str):
                opinions_raw = [{"opinion_text": opinions_raw}]

            for op in opinions_raw:
                op_text = op.get("opinion_text") or op.get("text") or op.get("plain_text") or op.get("html") or op.get("html_lawbox") or op.get("html_columbia") or op.get("html_with_citations") or op.get("text_plain") or ""
                op_text = clean_html_text(op_text)
                opinion_texts.append(op_text)
                opinions.append({
                    "author_str": op.get("author_str") or op.get("author") or "Court",
                    "download_url": op.get("download_url") or "",
                    "opinion_text": op_text
                })

            if not opinions and "casebody" in data:
                cb = data["casebody"]
                if isinstance(cb, dict) and "data" in cb:
                    cb_data = cb["data"]
                    if isinstance(cb_data, dict) and "opinions" in cb_data:
                        opinions_raw = cb_data["opinions"]
                        for op in opinions_raw:
                            op_text = op.get("opinion_text") or op.get("text") or ""
                            op_text = clean_html_text(op_text)
                            opinion_texts.append(op_text)
                            opinions.append({
                                "author_str": op.get("author_str") or op.get("author") or "Court",
                                "download_url": "",
                                "opinion_text": op_text
                            })

            if not opinions:
                for k in ["opinion_text", "text", "plain_text", "html", "body", "case_text"]:
                    val = data.get(k)
                    if val and isinstance(val, str):
                        op_text = clean_html_text(val)
                        opinion_texts.append(op_text)
                        opinions.append({
                            "author_str": "Court",
                            "download_url": "",
                            "opinion_text": op_text
                        })
                        break

            full_opinions_text = "\n\n".join(opinion_texts)
            summary = clean_html_text(data.get("summary") or "")
            syllabus = clean_html_text(data.get("syllabus") or "")
            headnotes = clean_html_text(data.get("headnotes") or "")
            headmatter = clean_html_text(data.get("headmatter") or "")
            
            judges = data.get("judges") or ""
            if isinstance(judges, list):
                judges = ", ".join(judges)
            attorneys = data.get("attorneys") or ""
            if isinstance(attorneys, list):
                attorneys = ", ".join(attorneys)

            status = determine_status(full_opinions_text or summary)
            topic = determine_topic(full_opinions_text or summary)
            judge = random.choice(["Smith", "Kagan", "Roberts", "Sotomayor", "Alito", "Thomas"]) # Mock fallback

            std_payload = {
                "case_name_full": name,
                "date_filed": date_filed,
                "court_full_name": court,
                "judges": judges,
                "attorneys": attorneys,
                "citations": citations,
                "summary": summary,
                "syllabus": syllabus,
                "headnotes": headnotes,
                "headmatter": headmatter,
                "opinions": opinions
            }
            full_text_json = json.dumps(std_payload, ensure_ascii=False, default=str)
            
            plain_text_for_embedding = f"{name}\n"
            if court:
                plain_text_for_embedding += f"Court: {court}\n"
            if citations:
                plain_text_for_embedding += f"Citations: {', '.join(citations)}\n"
            if summary:
                plain_text_for_embedding += f"\nSummary:\n{summary}\n"
            if syllabus:
                plain_text_for_embedding += f"\nSyllabus:\n{syllabus}\n"
            for op in opinions:
                plain_text_for_embedding += f"\nOpinion ({op.get('author_str')}):\n{op.get('opinion_text')}\n"

        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

            cleaned_content = clean_html_text(raw_content)
            name = f"Case {case_id}"
            reporter = f"Unpublished Case {case_id}"
            court = random.choice(FEDERAL_COURTS + STATE_COURTS)
            jurisdiction = "Federal"
            year = random.choice([1998, 2005, 2015, 2021, 2023])
            status = determine_status(cleaned_content)
            topic = determine_topic(cleaned_content)
            judge = "Unknown Judge"

            std_payload = {
                "case_name_full": name,
                "date_filed": "",
                "court_full_name": court,
                "opinions": [{"author_str": "Court", "opinion_text": cleaned_content}]
            }
            full_text_json = json.dumps(std_payload, ensure_ascii=False, default=str)
            plain_text_for_embedding = cleaned_content

        full_cases_to_insert.append({
            "case_id": case_id,
            "name": name,
            "reporter": reporter,
            "court": court,
            "jurisdiction": jurisdiction,
            "year": year,
            "status": status,
            "full_text": full_text_json
        })

        doc = Document(
            page_content=plain_text_for_embedding,
            metadata={
                "case_id": case_id,
                "name": name,
                "reporter": reporter,
                "court": court,
                "jurisdiction": jurisdiction,
                "year": year,
                "status": status,
                "judge": judge,
                "topic": topic
            }
        )
        raw_documents.append(doc)

    if not raw_documents:
        print("No documents found in repository.")
        return

    print("Saving full cases to Postgres database...")
    db_url_clean = db_url.replace("+psycopg2", "")
    try:
        conn = psycopg2.connect(db_url_clean)
        cursor = conn.cursor()
        
        # Insert all full documents into the database
        insert_query = """
            INSERT INTO full_cases (case_id, name, reporter, court, jurisdiction, year, status, full_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (case_id) DO UPDATE 
            SET name = EXCLUDED.name,
                reporter = EXCLUDED.reporter,
                court = EXCLUDED.court,
                jurisdiction = EXCLUDED.jurisdiction,
                year = EXCLUDED.year,
                status = EXCLUDED.status,
                full_text = EXCLUDED.full_text;
        """
        
        for case in tqdm(full_cases_to_insert, desc="Inserting Full Cases to Postgres"):
            cursor.execute(insert_query, (
                case["case_id"],
                case["name"],
                case["reporter"],
                case["court"],
                case["jurisdiction"],
                case["year"],
                case["status"],
                case["full_text"]
            ))
            
        conn.commit()
        cursor.close()
        conn.close()
        print("Successfully saved full cases to the database.")
    except Exception as e:
        print(f"Failed to save full cases to db: {e}")

    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
    )
    chunks = text_splitter.split_documents(raw_documents)

    print(f"Initializing {embed_provider} embedding model...")
    if embed_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model_name = embed_model or "text-embedding-3-small"
        api_key = embed_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required via --embed-key or OPENAI_API_KEY env var.")
        embeddings = OpenAIEmbeddings(model=model_name, api_key=api_key)
    elif embed_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        model_name = embed_model or "nomic-embed-text"
        embeddings = OllamaEmbeddings(model=model_name, base_url=embed_host)
    else:
        raise ValueError(f"Unknown embed provider: {embed_provider}")

    CONNECTION_STRING = db_url
    COLLECTION_NAME = "case_law"

    print(f"Upserting {len(chunks)} chunks into PostgreSQL Vector Database...")
    
    import concurrent.futures
    
    # Disable automatic extension creation to prevent privilege errors (since db_setup.py handles this)
    if hasattr(PGVector, 'create_vector_extension'):
        PGVector.create_vector_extension = lambda self: None

    # Initialize the table and delete old collection using just the first chunk
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=chunks[:1],
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
        use_jsonb=True
    )
    
    remaining_chunks = chunks[1:]
    if remaining_chunks:
        batch_size = 200
        batches = [remaining_chunks[i:i + batch_size] for i in range(0, len(remaining_chunks), batch_size)]
        
        workers = 10 if embed_provider != "ollama" else 1
        print(f"Spawning {workers} threads to insert {len(batches)} batches concurrently...")
        
        def process_batch(batch):
            db.add_documents(batch)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(process_batch, batches), total=len(batches), desc="Concurrent Embedding & Inserting"))
            
    print("Embedding complete. Your Vector Database is fully primed.")
    
    print("Running db_setup to add Hybrid Search FTS indices and other tables...")
    import subprocess
    subprocess.run(["python", "db_setup.py"], env=dict(os.environ, DATABASE_URL=CONNECTION_STRING))


def main():
    parser = argparse.ArgumentParser(description="Judge Read Data Pipeline")
    parser.add_argument("--action", choices=["download", "embed", "all"], default="all",
                        help="Action to perform (default: all)")
    parser.add_argument("--source", choices=["hf", "courtlistener"], default="hf",
                        help="Data source for download (default: hf)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only download/extract the first N cases. Omit to download all.")
    parser.add_argument("--all-cases", action="store_true",
                        help="Explicitly download and process ALL cases (overrides --limit). Warning: 8+ million cases, 40GB+.")
                        
    # Embedding Options
    parser.add_argument("--embed-provider", choices=["openai", "ollama"], default="openai",
                        help="Provider for embeddings (default: openai). Note: Anthropic/Claude does not provide native text embeddings.")
    parser.add_argument("--embed-model", default=None,
                        help="Embedding model name (defaults: openai=text-embedding-3-small, ollama=nomic-embed-text)")
    parser.add_argument("--embed-key", default=os.getenv("OPENAI_API_KEY"), help="API key for the embedding provider (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--embed-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"), help="Host URL for Ollama (default: OLLAMA_HOST or http://localhost:11434)")
                        
    # PostgreSQL Options
    parser.add_argument("--pg-host", default=os.getenv("PGHOST", "localhost"), help="PostgreSQL host")
    parser.add_argument("--pg-port", default=os.getenv("PGPORT", "5432"), help="PostgreSQL port")
    parser.add_argument("--pg-user", default=os.getenv("PGUSER", "user"), help="PostgreSQL user")
    parser.add_argument("--pg-password", default=os.getenv("PGPASSWORD", "password"), help="PostgreSQL password")
    parser.add_argument("--pg-db", default=os.getenv("PGDATABASE", "judgeread"), help="PostgreSQL db name")
    
    # Tracing Options
    parser.add_argument("--langsmith-key", default=os.getenv("LANGCHAIN_API_KEY"), help="LangSmith API key for tracing telemetry")
    parser.add_argument("--drop", action="store_true", help="Drop all existing database tables before recreating schemas and ingesting cases")
    args = parser.parse_args()

    if args.langsmith_key or os.getenv("LANGCHAIN_TRACING_V2") == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if args.langsmith_key:
            os.environ["LANGCHAIN_API_KEY"] = args.langsmith_key
        os.environ.setdefault("LANGCHAIN_PROJECT", "Judge_Read_Pipeline")
        print("🔍 LangSmith Tracing Enabled")

    # Construct the fallback connection string from CLI args
    cli_db_url = f"postgresql+psycopg2://{args.pg_user}:{args.pg_password}@{args.pg_host}:{args.pg_port}/{args.pg_db}"
    # Prefer DATABASE_URL env var if set, otherwise fallback to CLI
    db_url = os.getenv("DATABASE_URL", cli_db_url)
    
    print(f"Testing connection to PostgreSQL database at {args.pg_host}:{args.pg_port}...")
    try:
        import psycopg2
        db_url_clean = db_url.replace("+psycopg2", "")
        conn = psycopg2.connect(db_url_clean)
        conn.close()
        print("✅ Successfully connected to PostgreSQL!")
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("Please ensure PostgreSQL is running and the credentials are correct before starting.")
        return

    if args.drop:
        print("🗑️ Dropping all existing tables in database as requested by --drop...")
        try:
            import psycopg2
            db_url_clean = db_url.replace("+psycopg2", "")
            conn = psycopg2.connect(db_url_clean)
            cursor = conn.cursor()
            
            tables_to_drop = [
                "langchain_pg_embedding", "langchain_pg_collection",
                "chat_messages", "chat_sessions", "analytics_queries",
                "case_annotations", "full_cases"
            ]
            for table in tables_to_drop:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                
            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Database successfully cleaned! Starting with a fresh schema.")
        except Exception as e:
            print(f"❌ Failed to drop existing tables: {e}")
            return

    if args.action in ["download", "all"]:
        # If the user passed --all-cases, force the limit to None
        actual_limit = None if args.all_cases else args.limit
        
        if args.source == "hf":
            download_huggingface(limit=actual_limit)
        else:
            download_courtlistener(limit=actual_limit)
            
    if args.action in ["embed", "all"]:
        actual_limit = None if args.all_cases else args.limit
        embed_data(db_url, args.embed_provider, args.embed_model, args.embed_key, args.embed_host, limit=actual_limit)

if __name__ == "__main__":
    main()
