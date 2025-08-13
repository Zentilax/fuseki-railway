import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')

# Fuseki Endpoint Configuration
# Since we're in the same container, use localhost with internal port
#INTERNAL_FUSEKI_PORT = os.getenv('INTERNAL_FUSEKI_PORT', '3031')
#FUSEKI_ENDPOINT = os.getenv('FUSEKI_ENDPOINT', f'http://localhost:{INTERNAL_FUSEKI_PORT}/german-food-inferred/query')
INTERNAL_FUSEKI_PORT = os.getenv('INTERNAL_FUSEKI_PORT', '3031')
FUSEKI_ENDPOINT = f"http://localhost:{INTERNAL_FUSEKI_PORT}/german-food-inferred/query"

# FAISS Configuration
FAISS_SIMILARITY_THRESHOLD = float(os.getenv('FAISS_SIMILARITY_THRESHOLD', '0.8'))
FAISS_VOLUME_PATH = os.getenv('FAISS_VOLUME_PATH', '/app/data')

# Flask API Configuration 
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '5000'))