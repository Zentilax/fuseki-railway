import requests

# Replace with your Railway API base URL
url = "https://fuseki-railway-production.up.railway.app/query"

# Human language question
payload = {
    "question": "what are some dishes in bavaria"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.status_code)
print(response.text)



