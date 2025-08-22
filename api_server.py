from flask import Flask, request, jsonify
import os
import sys
import logging
from datetime import datetime
from functools import wraps
from flask_cors import CORS

# Add the app directory to Python path
sys.path.append('/app')

from query_system import QueryHistoryVectorDB, load_ontology_prompt, generate_sparql_query, query_fuseki, format_results_raw, format_results_with_llm
from config import FAISS_VOLUME_PATH, API_HOST, API_PORT, INTERNAL_FUSEKI_PORT

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/query_logs.log'),  # Log to file
        logging.StreamHandler(sys.stdout)  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('x-api-key')
        client_ip = request.remote_addr
        
        if not key or key != os.getenv('API_KEY'):
            logger.warning(f"Unauthorized access attempt from {client_ip} - Invalid API key")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# Initialize the vector database with your existing volume path
vector_db = QueryHistoryVectorDB(volume_path='/data')
ontology_prompt = load_ontology_prompt()

@app.route('/', methods=['GET'])
@require_api_key
def home():
    client_ip = request.remote_addr
    logger.info(f"Home endpoint accessed from {client_ip}")
    
    return jsonify({
        "message": "German Cuisine Query API with FAISS Vector Search",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "history": "/history",
            "logs": "/logs (GET)",
            "fuseki_proxy": "/german-food-inferred/query (maintains compatibility)"
        },
        "version": "1.0.0"
    })

@app.route('/health', methods=['GET'])
@require_api_key
def health_check():
    from config import FUSEKI_ENDPOINT
    
    client_ip = request.remote_addr
    logger.info(f"Health check from {client_ip}")
    
    # Test if we can reach Fuseki
    fuseki_status = "unknown"
    try:
        import requests
        response = requests.get(f"http://localhost:{INTERNAL_FUSEKI_PORT}/$/ping", timeout=5)
        fuseki_status = "running" if response.status_code == 200 else "error"
    except:
        fuseki_status = "unreachable"
    
    return jsonify({
        "status": "healthy", 
        "queries_in_history": len(vector_db.metadata),
        "fuseki_endpoint": FUSEKI_ENDPOINT,
        "fuseki_status": fuseki_status,
        "openai_configured": bool(os.getenv('OPENAI_API_KEY')),
        "ports": {
            "fuseki_internal": INTERNAL_FUSEKI_PORT,
            "api_public": "5000"
        }
    })

@app.route('/query', methods=['POST'])
@require_api_key
def process_query():
    client_ip = request.remote_addr
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        force_new_query = data.get('force_new_query', False)

        if not question:
            logger.warning(f"❌ Empty question from {client_ip}")
            return jsonify({"error": "Question is required"}), 400

        # Log the incoming query
        logger.info(f"📝 NEW QUERY from {client_ip}: '{question}' (force_new: {force_new_query})")

        # Check for similar queries (now with automatic paraphrasing if needed)
        similar_query = vector_db.search_similar_queries(question)
        
        #if similar_query and similar_query.get('sparql_query'):
            #logger.info(f"🔍 Similar query found for '{question}': '{similar_query.get('question', 'N/A')}' (score: {similar_query.get('score', 'N/A')})")

        # Prepare the prompt for SPARQL generation
        prompt_context = load_ontology_prompt()

        # Only add similar query context if we have a complete similar query (above threshold)
        if similar_query and similar_query.get('sparql_query') and not force_new_query:
            prompt_context += (
                "\n\n# Note: A similar question was asked previously.\n"
                f"Similar question: {similar_query['question']}\n"
                f"Similar SPARQL query: {similar_query['sparql_query']}\n"
                "Please generate a SPARQL query that answers the current question distinctly."
            )

        # Generate new query
        #logger.info(f"🔄 Generating SPARQL for: '{question}'")
        sparql_query = generate_sparql_query(question, prompt_context)
        
        if sparql_query.startswith("⚠️"):
            logger.warning(f"❌ SPARQL generation failed for '{question}': {sparql_query}")
            return jsonify({"error": sparql_query}), 400
        
        #logger.info(f"✅ Generated SPARQL: {sparql_query}")
        
        # Query Fuseki
        results = query_fuseki(sparql_query)
        results_count = len(results) if results else 0
        #logger.info(f"📊 Fuseki returned {results_count} results for '{question}'")
        
        raw_formatted = format_results_raw(results)
        formatted_results = format_results_with_llm(raw_formatted, question)

        # Prepare response
        response = {
            "question": question,
            "from_cache": False,
            "similar_query": None,
            "similarity_search_log": similar_query.get("search_log", []) if similar_query else []
        }

        # Add similar query info to response
        if similar_query and similar_query.get('sparql_query'):
            response['similar_query'] = {
                "question": similar_query.get('question', "N/A"),
                "timestamp": similar_query.get('timestamp', "N/A"),
                "sparql_query": similar_query.get('sparql_query', "N/A"),
                "answer_preview": similar_query.get('formatted_answer', "N/A")[:200] if similar_query.get('formatted_answer') else "N/A",
                "score": similar_query.get('score', "N/A"),
                "matched_via_paraphrase": similar_query.get('matched_via_paraphrase')
            }
        elif similar_query:
            response['similar_query'] = {
                "question": similar_query.get('question', "N/A"),
                "score": similar_query.get('score', "N/A"),
                "sparql_query": None,
                "timestamp": None,
                "answer_preview": None,
                "best_paraphrase": similar_query.get('best_paraphrase')
            }

        # Check rules before adding to history
        if results:
            # Check if similar query exists (using original question for similarity)
            existing_similar = vector_db.search_similar_queries(question)
            
            if not existing_similar or not existing_similar.get('sparql_query'):
                vector_db.add_query_to_history(question, sparql_query, results, formatted_results)
                logger.info(f"💾 Query '{question}' added to FAISS history")
            else:
                logger.info(f"⚠️ Similar query exists for '{question}', not adding to history")
        else:
            logger.warning(f"⚠️ Query '{question}' returned no results, not adding to history")

        # Update response with results
        response.update({
            "answer": formatted_results,
            "sparql_query": sparql_query,
            "raw_results_count": results_count
        })
        
        # Log completion time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"✅ Query '{question}' completed in {processing_time:.2f}s from {client_ip}")
        
        return jsonify(response)
        
    except Exception as e:
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.error(f"💥 Query processing failed after {processing_time:.2f}s from {client_ip}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
@require_api_key
def get_history():
    """Get query history"""
    client_ip = request.remote_addr
    try:
        limit = int(request.args.get('limit', 10))
        history = vector_db.metadata[-limit:] if vector_db.metadata else []
        
        logger.info(f"📚 History requested from {client_ip} (limit: {limit})")
        
        return jsonify({
            "total_queries": len(vector_db.metadata),
            "recent_queries": [
                {
                    "question": item['question'],
                    "timestamp": item['timestamp']
                }
                for item in reversed(history)
            ]
        })
    except Exception as e:
        logger.error(f"❌ History request failed from {client_ip}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# New endpoint to view recent logs
@app.route('/logs', methods=['GET'])
@require_api_key
def get_logs():
    """Get recent query logs"""
    client_ip = request.remote_addr
    try:
        lines = int(request.args.get('lines', 100))
        log_file = '/data/query_logs.log'
        
        if not os.path.exists(log_file):
            return jsonify({"logs": [], "message": "No log file found"})
        
        # Read the last N lines from the log file
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        logger.info(f"📋 Logs requested from {client_ip} (last {lines} lines)")
        
        return jsonify({
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "showing_lines": len(recent_lines)
        })
        
    except Exception as e:
        logger.error(f"❌ Logs request failed from {client_ip}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Proxy routes to maintain compatibility with existing Fuseki endpoints
@app.route('/german-food-inferred/<path:path>', methods=['GET', 'POST'])
@require_api_key
def fuseki_inferred_proxy(path):
    """Direct proxy to your inference endpoint"""
    import requests
    
    client_ip = request.remote_addr
    logger.info(f"🔗 Fuseki inference proxy: {path} from {client_ip}")
    
    fuseki_url = f"http://localhost:{INTERNAL_FUSEKI_PORT}/german-food-inferred/{path}"
    
    if request.method == 'GET':
        resp = requests.get(fuseki_url, params=request.args, headers=dict(request.headers))
    else:
        # Log SPARQL queries sent directly to Fuseki
        query_data = request.get_data(as_text=True)
        if 'query=' in query_data:
            logger.info(f"🔍 Direct SPARQL query to Fuseki from {client_ip}: {query_data[:200]}...")
        
        resp = requests.post(fuseki_url, 
                           data=request.get_data(), 
                           headers=dict(request.headers),
                           params=request.args)
    
    return resp.content, resp.status_code, dict(resp.headers)

@app.route('/german-food/<path:path>', methods=['GET', 'POST'])
@require_api_key
def fuseki_raw_proxy(path):
    """Proxy to raw data service"""
    import requests
    
    client_ip = request.remote_addr
    logger.info(f"🔗 Fuseki raw proxy: {path} from {client_ip}")
    
    fuseki_url = f"http://localhost:{INTERNAL_FUSEKI_PORT}/german-food/{path}"
    
    if request.method == 'GET':
        resp = requests.get(fuseki_url, params=request.args, headers=dict(request.headers))
    else:
        resp = requests.post(fuseki_url, 
                           data=request.get_data(), 
                           headers=dict(request.headers),
                           params=request.args)
    
    return resp.content, resp.status_code, dict(resp.headers)

@app.route('/clear-history', methods=['POST'])
@require_api_key
def clear_history():
    """Clear query history (use with caution)"""
    client_ip = request.remote_addr
    try:
        # Remove files
        if os.path.exists(vector_db.index_file):
            os.remove(vector_db.index_file)
        if os.path.exists(vector_db.metadata_file):
            os.remove(vector_db.metadata_file)
        
        # Recreate empty index
        vector_db.load_or_create_index()
        
        logger.warning(f"🗑️ History cleared by {client_ip}")
        return jsonify({"message": "History cleared successfully"})
    except Exception as e:
        logger.error(f"❌ History clear failed from {client_ip}: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info(f"🚀 Starting German Cuisine Query API on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False)