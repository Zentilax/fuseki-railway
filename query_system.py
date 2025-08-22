from openai import OpenAI
import requests
import faiss
import numpy as np
import pickle
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
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
        search_log = []  # Track what happened during search
        
        if self.index.ntotal == 0:
            search_log.append("No queries in history yet")
            return {"search_log": search_log}
    
        embedding = self.get_embedding(question)
        if embedding is None:
            search_log.append("Failed to get embedding for question")
            return {"search_log": search_log}
        
        # Search for most similar query
        scores, indices = self.index.search(embedding.reshape(1, -1), 1)
        best_score = scores[0][0]
        best_idx = indices[0][0]
        
        if best_idx == -1:
            search_log.append("No results in FAISS search")
            return {"search_log": search_log}
        
        similar_query_data = self.metadata[best_idx]
        search_log.append(f"Original question similarity: {best_score:.3f} (threshold: {self.similarity_threshold})")
        
        # Debug the comparison
        is_above_threshold = best_score >= self.similarity_threshold
        search_log.append(f"Threshold check: {best_score:.3f} >= {self.similarity_threshold} = {is_above_threshold}")
        
        if is_above_threshold:
            search_log.append(f"Found match above threshold ({self.similarity_threshold}) with original question")
            search_log.append(f"Matched: {similar_query_data['question']}")
            return {
                **similar_query_data,   # unpack all original fields
                "score": float(best_score),
                "search_log": search_log
            }
        else:
            search_log.append(f"Original question below threshold ({self.similarity_threshold})")
            search_log.append(f"Closest match was: {similar_query_data['question']}")
            
            # Try paraphrases if original didn't meet threshold
            search_log.append("Generating paraphrases to improve similarity search...")
            variations, paraphrase_debug = generate_query_variations(question)
            
            # Add paraphrase generation details to search log
            search_log.extend(paraphrase_debug)
            search_log.append(f"Generated {len(variations)} paraphrases: {variations}")
            
            best_match = {
                "question": similar_query_data['question'],
                "score": float(best_score),
                "search_log": search_log
            }
            
            for i, variation in enumerate(variations):
                search_log.append(f"Trying paraphrase {i+1}: {variation}")
                var_embedding = self.get_embedding(variation)
                if var_embedding is None:
                    search_log.append(f"  Failed to get embedding for paraphrase {i+1}")
                    continue
                    
                var_scores, var_indices = self.index.search(var_embedding.reshape(1, -1), 1)
                var_best_score = var_scores[0][0]
                var_best_idx = var_indices[0][0]
                
                if var_best_idx != -1:
                    var_similar_data = self.metadata[var_best_idx]
                    search_log.append(f"  Paraphrase {i+1} similarity: {var_best_score:.3f}")
                    
                    if var_best_score >= self.similarity_threshold:
                        search_log.append(f"  ✅ Found match with paraphrase {i+1}!")
                        search_log.append(f"  Matched: {var_similar_data['question']}")
                        return {
                            **var_similar_data,
                            "score": float(var_best_score),
                            "search_log": search_log,
                            "matched_via_paraphrase": variation
                        }
                    
                    # Keep track of best score
                    if var_best_score > best_match["score"]:
                        best_match = {
                            "question": var_similar_data['question'],
                            "score": float(var_best_score),
                            "search_log": search_log,
                            "best_paraphrase": variation
                        }
                else:
                    search_log.append(f"  No results for paraphrase {i+1}")
            
            search_log.append("No paraphrases found matches above threshold")
            return best_match
    
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
    return """You are a SPARQL expert working with the following ontology:
            Classes:
            - Dish: a food item.
            - Subclasses of Dish : 
                -Appetizer
                -Condiment
                -Dessert
                -MainCourse
                -Salad
                -SideDish
                -Snack
                -Soup
            - Ingredient: food components like Pork, Spices, etc. (these are instances)
            - MainIngredient
            - MealEatenAtPartOfDay : anytime, breakfast,dinner etc.
            - MeatCut :
            - Subclasses of MeatCut: 
                -Beef
                -Chicken
                -Duck
                -Goose
                -Pork
                -Rabbit
                -Turkey
                -Veal
                -Vension
            - ServingTemperature
            - StateOfMainIngredient
            - Variation
            - FlavorProfile : aromatic,bitter,buttery etc. (these are instances)
            - Region
            - Subclasses of Region:
                -German : places like Bavaria, Saxony, etc. (these are instances)
                -NonGerman
            - Beverage:
            - Subclasses Of Beverage:
                -Alcoholic
                    Subclasses of Alcoholic: 
                        -Beer
                        -Brandy
                        -Cocktail
                        -Digestif
                        -FermentedAlcoholic
                        -Liquor
                        -MaltBeverage
                        -Spirit
                        -Spritzer
                        -Wine
                -NonAlcoholic
                    Subclasses of NonAlcoholic:
                        -Coffee
                        -HotChocolate
                        -Icetea
                        -Juice
                        -NonAlcoholicBeer
                        -Soda
                        -Tea
                        -Water
            - DietType: Halal,Kosher,Omnivore, Vegetarian, Vegan.

            Object Properties:
            - hasBeverageType (Alcoholic, NonAlcoholic)
            - hasDietType (dish -> DietType)
            - hasFlavorProfile (dish -> FlavorProfile)
            - hasIngredient (dish -> Ingredient)
            - hasMealEatenAtPartOfDay (dish -> MealEatenAtPartOfDay)
            - hasMeatCut (dish -> MeatCut)
            - hasPreparationMethod (dish -> hasPreparationMethod)
            - hasRegion (dish -> Region)
            - hasServingtemperature (dish -> ServingTemperature)
            - hasStateOfMainIngredient (dish -> StateOfMainIngredient)
            - hasVariation (dish -> Variation)

            Data Properties:
            - hasAlcoholContent (Beverage -> Decimal)
            - hasDescription (owl:Thing -> String)
            - hasPreparationTimeMinutes (Dish -> Decimal)
            - isCarbonated (Beverage -> boolean)
            - isGermanStaple (owl:Thing -> boolean)

            Prefix:
            PREFIX gc: <http://example.org/german-cuisine#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SPARQL Query Guidelines
            1. Reasoner Support
                Querying a class automatically returns instances of its subclasses (e.g., querying gc:Dish returns gc:MainCourse dishes).

            2. Do not use rdfs:subClassOf*.

            3. Instance Awareness
                If the user specifies a term that is an instance (e.g., Bavaria, Aromatic, Chicken), query it via the correct object property.
                Example: "chicken dishes" → ?dish gc:hasMeatCut gc:chicken
                Example: "dishes from Bavaria" → ?dish gc:hasRegion gc:Bavaria

            4. Allowed Vocabulary Only
                **IMPORTANT** Use only classes, object properties, and data properties listed above.
                If the user asks for something not in the ontology, explain it is unavailable.
                Never use hasName or similar — use ?entity or rdfs:label if present.
            5. Flexible Matching for Multi-Valued Properties
                When generating triple patterns for properties like rdf:type, gc:hasDietType, gc:hasIngredient, etc., do not assume exclusivity.
                Always match using patterns that allow entities with multiple values to be included if any value matches the user request.
                Example:
                "vegetarian dishes" → returns dishes that are gc:vegetarian even if they are also gc:omnivore.
            6. Always return the hasDescription or description of the selected items
            7. for every instance or class name, it always starts with a capital letter
            8. instances, classes and any other variables are with Camelcases, where the first letter is always Capital
            9 Object property always stars with 'has' the h is lowercase
            10. **CRITICAL** When looking up instances/dish names, Use CONTAINS in ?Dish
            11. **CRITICAL** Always Limit result by 10 rows
            12. Always use english names, whatever the user query languange is
            13. if he as for a certain class of ingredients e.g fruit, there are no fruit class. so just try apple,strawberry etc as instances
            14. **IMPORTANT** use OPTIONAL when contains in ?dish for name search e.g contains("cake"). but use object property first if possible
            15. if a user queries cake, or cupcake or something similar, use dessert, and optionally search for cake in the desc or name

            EXAMPLE QUERY
            ## To find Beverages with > 5 alcohol content ##
            PREFIX gc: <http://example.org/german-cuisine#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT DISTINCT ?bev ?alcoholContent WHERE {
            ?bev rdf:type gc:Beverage .
            ?bev gc:hasAlcoholContent ?alcoholContent .
            FILTER(?alcoholContent > 5)
            }

            **IMPORTANT** Always use the prefix provided above
            **REMEMBER: Output ONLY the executable SPARQL query with no additional formatting or text.**"""

def load_formatting_prompt():
    return """You are a helpful assistant that formats query results about German cuisine in a user-friendly way.  
                Your task is to take raw database results and present them in a clear, readable format for users asking about German food.

                Guidelines:
                - Make the response conversational and helpful
                - Remove technical URIs and database artifacts  
                - Focus on the food names and descriptions
                - Group or organize information logically if appropriate
                - Keep it concise but informative
                - Use emojis sparingly and appropriately if they enhance readability
                - Dont make it conversational because you are a QA system.

                Present the information as if you're a knowledgeable guide helping someone discover German cuisine.
                do not add information outside of the context and knowledge you have been given
                if there is no results, kindly explain that your domain of knowledge is unable to answer the users question"""

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

def generate_query_variations(question: str) -> tuple[List[str], List[str]]:
    """Generate paraphrases and variations of the query using LLM
    
    Returns:
        (variations_list, debug_log)
    """
    debug_log = []
    
    try:
        prompt = f"""Generate 3 alternative ways to ask this question: "{question}"

Focus on different ways to express the same concept:
- Different words for ingredients (contains, has, includes, made with, uses)
- Different phrasings for food/dish types
- Synonymous terms
- if the original query contains something like fruit, make it ingredients, because there is no fruit class
- if the original query is not in english, try make paraphrases that are translated into english

Return ONLY the alternative questions, one per line. Do not include numbers, bullets, or explanations.

Example for "german dish that has spinach as ingredient":
german dish that contains spinach
german dish made with spinach  
german cuisine with spinach ingredient
spinach-based german dish

Now generate alternatives for: "{question}"
"""

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
        )
        
        content = response.choices[0].message.content.strip()
        debug_log.append(f"Raw LLM response: {repr(content)}")
        
        if not content:
            debug_log.append("Empty response from LLM")
            return [], debug_log
            
        # Split by newlines and clean up
        lines = content.split('\n')
        variations = []
        
        for line in lines:
            # Clean up each line
            cleaned = line.strip()
            # Remove common prefixes (bullets, numbers, dashes)
            cleaned = cleaned.lstrip('- •*123456789.() ')
            # Remove quotes if present
            cleaned = cleaned.strip('"\'')
            
            # Only keep substantial variations (not too short, not empty)
            if len(cleaned) > 10 and cleaned.lower() != question.lower():
                variations.append(cleaned)
        
        debug_log.append(f"Processed {len(variations)} valid paraphrases from {len(lines)} lines")
        return variations, debug_log
        
    except Exception as e:
        debug_log.append(f"Error generating query variations: {e}")
        # Return some basic fallbacks for common cases
        fallbacks = []
        if "has" in question and "ingredient" in question:
            fallbacks.append(question.replace("has", "contains").replace("as one of the ingredient", ""))
            fallbacks.append(question.replace("has", "includes").replace("as one of the ingredient", ""))
        elif "dish" in question:
            fallbacks.append(question.replace("dish", "food"))
            fallbacks.append(question.replace("dish", "cuisine"))
        
        debug_log.append(f"Using {len(fallbacks)} fallback paraphrases due to error")
        return fallbacks, debug_log