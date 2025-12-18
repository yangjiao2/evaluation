import requests
from pymongo import MongoClient

# Configuration
EVALUATION_MS_URL = "http://localhost:7331"
GITEA_SERVER_URL = "http://localhost:3000"
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "evaluations"
COLLECTION_NAME = "Evaluations"# Initialize MongoDB client
client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


# Function to check connection to Evaluation MS
def check_evaluation_ms():
    try:
        response = requests.get(f"{EVALUATION_MS_URL}/v1/evaluations?page_size=1")
        return response.status_code == 200
    except:
        return False

# Function to check connection to MongoDB
def check_mongodb():
    try:
        client.admin.command('ping')
        return True
    except:
        return False

def check_gitea_server():
    try:
        response = requests.get(f"{GITEA_SERVER_URL}/api/v1/version")
        return response.status_code == 200
    except:
        return False
