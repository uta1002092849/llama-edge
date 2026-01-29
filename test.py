import requests
import json

# 1. Configuration
url = "http://localhost:8000/predict"
api_key = "5ca558f269ba1ae7b6fdd6d5fada3b65"  # Must match your server's API_KEY

# 2. Define the payload
payload = {
    "context_ids": [108, 110, 112, 115, 117, 234, 349, 421, 541, 582, 608, 831, 905, 912, 913, 940, 994, 995, 996, 1019, 1076, 1077, 1095, 1125, 1181, 1188, 1251, 1275, 1278, 1287, 1291, 1304, 1393, 1415, 1542, 1685, 1721, 1790, 1942, 1948, 1958, 1999, 2000, 2025, 47178924, 47185648],
    "top_k": 10
}

# 3. Set headers
headers = {
    "x-api-token": api_key,
    "Content-Type": "application/json"
}

# 4. Send request
try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    
    # 5. Handle response
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Success! Total predictions: {data['total_predictions']}")
        print("-" * 50)
        print(f"{'Rank':<5} {'ID':<10} {'Prob':<10} {'Label'}")
        print("-" * 50)
        
        for item in data['predictions']:
            print(f"{item['rank']:<5} {item['id']:<10} {item['probability']:.4f}     {item['label']}")
    else:
        print(f"\n❌ Error {response.status_code}:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n❌ Could not connect. Is the server running?")