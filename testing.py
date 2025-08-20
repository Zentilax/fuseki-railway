import requests
import json

# Replace with your Railway API base URL
url = "https://fuseki-railway-production.up.railway.app/query"

# Human language question
payload = {
    "question": "german dish that has spinach as one of the ingredient"
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": "germanfoodchatbot"  # <-- your secret API key
}

response = requests.post(url, json=payload, headers=headers)

print(response.status_code)
print(response.text)

data = json.loads(response.text)

# Print neatly
print("Answer:\n", data.get("answer", "No answer found"), "\n")
print("SPARQL Query:\n", data.get("sparql_query", "No SPARQL query found"), "\n")
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