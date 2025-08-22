import requests
import json

url = "https://fuseki-railway-production.up.railway.app/query"

# Human language question
payload = {
    "question": "i want to eat halal breakfast" #<--- CHANGE THIS PART ONLY
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": "" 
}

response = requests.post(url, json=payload, headers=headers)

print(response.status_code)
#print(response.text)

data = json.loads(response.text)


print("#########################################")
print("RESULTS")
print("#########################################")
print("Answer:\n", data.get("answer", "No answer found"), "\n")
print("\n","#########################################")
print("SPARQL RESULT")
print("#########################################","\n")
print("SPARQL Query:\n", data.get("sparql_query", "No SPARQL query found"), "\n")

print("\n","#########################################")
print("CHECK PARAPHRASING")
print("#########################################", "\n")
for i, line in enumerate(data["similarity_search_log"], 1):
    print(f"{i:02d}. {line}")

print("\n","#########################################")
print("SIMILAR QUERY INFO")
print("#########################################", "\n")
if "similar_query" in data and data["similar_query"]:
    print("Similar Query Info:")
    sim = data["similar_query"]
    print("with the score of: ",sim.get("score"))
    print(" Question:", sim.get("question"))
    print(" Answer preview:", sim.get("answer_preview"))
    print(" SPARQL:", sim.get("sparql_query"))
    print(" Timestamp:", sim.get("timestamp"))
else:
    print("No Similar Queries found")
    sim = data["similar_query"]
    print("the most similar question was: ",sim.get("question"))
    print("with the score of: ",sim.get("score"))