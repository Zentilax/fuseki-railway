from openai import OpenAI
import requests
import faiss
import numpy as np
import pickle
import os
import json
from datetime import datetime
from config import OPENAI_API_KEY, FUSEKI_ENDPOINT, FAISS_SIMILARITY_THRESHOLD, FAISS_VOLUME_PATH

client = OpenAI(api_key=OPENAI_API_KEY)

class QueryHistoryVectorDB:
    def __init__(self, volume_path=FAISS_VOLUME_PATH, similarity_threshold=FAISS_SIMILARITY_THRESHOLD):
        self.volume_path = volume_path
        self.similarity_threshold = similarity_threshold
        self.index_file = os.path.join(volume_path, "query_history.faiss")
        self.metadata_file = os.path.join(volume_path, "query_metadata.pkl")
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        
        # Ensure volume directory exists
        os.makedirs(volume_path, exist_ok=True)
        
        self.index = None
        self.metadata = []
        self.load_or_create_index()
    
    def load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                print("📚 Loading existing query history index...")
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"✅ Loaded {len(self.metadata)} historical queries")
            else:
                print("🆕 Creating new query history index...")
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.metadata = []
                self.save_index()
        except Exception as e:
            print(f"⚠️ Error loading index, creating new one: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
    
    def get_embedding(self, text):
        """Get embedding for text using OpenAI"""
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"⚠️ Error getting embedding: {e}")
            return None
    
    def search_similar_queries(self, question):
        """Search for similar queries in history"""
        if self.index.ntotal == 0:
            return None
        
        embedding = self.get_embedding(question)
        if embedding is None:
            return None
        
        # Search for most similar query
        scores, indices = self.index.search(embedding.reshape(1, -1), 1)
        
        if scores[0][0] >= self.similarity_threshold:
            similar_query_idx = indices[0][0]
            similar_query_data = self.metadata[similar_query_idx]
            
            print(f"🎯 Found similar query (similarity: {scores[0][0]:.3f})")
            print(f"   Previous: {similar_query_data['question']}")
            
            return similar_query_data
        
        return None
    
    def add_query_to_history(self, question, sparql_query, results, formatted_answer):
        """Add a new query and its results to the history"""
        embedding = self.get_embedding(question)
        if embedding is None:
            return
        
        # Add to index
        self.index.add(embedding.reshape(1, -1))
        
        # Add metadata
        query_data = {
            'question': question,
            'sparql_query': sparql_query,
            'results': results,
            'formatted_answer': formatted_answer,
            'timestamp': datetime.now().isoformat()
        }
        self.metadata.append(query_data)
        
        # Save to volume
        self.save_index()
        print(f"💾 Query saved to history (total: {len(self.metadata)})")
    
    def save_index(self):
        """Save index and metadata to volume"""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"⚠️ Error saving index: {e}")

def load_ontology_prompt():
    try:
        with open("/app/ontology_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a SPARQL query generator for a German cuisine knowledge base. Generate only the SPARQL query to answer the user's question."

def load_formatting_prompt():
    try:
        with open("/app/formatting_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Format the following SPARQL query results in a user-friendly way."

def generate_sparql_query(question, ontology_prompt):
    messages = [
        {"role": "system", "content": ontology_prompt},
        {"role": "user", "content": f"User Question: {question}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",  # Using a valid model
            messages=messages
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"⚠️ Error generating SPARQL query: {str(e)}"

def query_fuseki(sparql_query):
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.post(FUSEKI_ENDPOINT, data={"query": sparql_query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"SPARQL query failed: {response.status_code} - {response.text}")
    
    results = response.json()["results"]["bindings"]
    return results

def format_results_raw(results):
    """Convert raw SPARQL results to a simple text format for LLM processing."""
    if not results:
        return "No results found."

    output = ""
    for row in results:
        output += ", ".join(f"{key}: {val['value']}" for key, val in row.items()) + "\n"
    return output

def format_results_with_llm(raw_results, original_question):
    if not raw_results or raw_results == "No results found.":
        return raw_results
    
    formatting_prompt = load_formatting_prompt()

    try:
        messages = [
            {"role": "system", "content": formatting_prompt},
            {"role": "user", "content": f"Original question: {original_question}\n\nRaw results from database:\n{raw_results}\n\nPlease format this information in a user-friendly way."}
        ]
        
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️ Formatting failed, showing raw results: {e}")
        return raw_results