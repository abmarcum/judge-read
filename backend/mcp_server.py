import os
import psycopg2
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings

# Create the MCP Server
mcp = FastMCP("Judge Read Database")

# Assuming standard db connections similar to main.py
DB_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost:5432/judgeread")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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
    if not OPENAI_API_KEY:
        return ["Error: OPENAI_API_KEY environment variable is not set. Cannot generate embeddings for semantic search."]
        
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        query_embedding = embeddings.embed_query(query)
    except Exception as e:
        return [f"Error generating embedding: {e}"]

    # Connect to DB
    try:
        conn = psycopg2.connect(DB_URL)
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

    hybrid_search_sql = f"""
        SELECT 
            document, 
            cmetadata,
            (embedding <=> {vector_query}) AS distance
        FROM case_law
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
        
        case_name = meta.get('name', 'Unknown Case')
        case_year = meta.get('year', 'Unknown Year')
        case_court = meta.get('court', 'Unknown Court')
        case_status = meta.get('status', 'good_law')
        
        formatted_results.append(
            f"Case: {case_name}\nYear: {case_year}\nCourt: {case_court}\nStatus: {case_status}\nDistance: {distance:.4f}\nExcerpt:\n{doc_text[:1000]}..."
        )

    return formatted_results

if __name__ == "__main__":
    # Allow execution via `python mcp_server.py` or via the `mcp` CLI
    mcp.run()
