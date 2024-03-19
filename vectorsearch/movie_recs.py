import pymongo
import requests
import openai

# Set your OpenAI API key and model name
openai.api_key = "<OPENAI_API_KEY>"
openai_model = "text-embedding-ada-002"

# Set your Hugging Face API token and model URL
hf_token = "<HUGGINGFACE_TOKEN>"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

client = pymongo.MongoClient("<MONGO_URI>")
db = client.sample_mflix
collection = db.movies

"""
The following code snippet demonstrates how to use the Hugging Face and OpenAI APIs to generate embeddings for movie plots and then use the embeddings to perform semantic search.
"""
def generate_embedding(api_name: str, text: str) -> list[float]:
    if api_name == "huggingface":
        response = requests.post(
            embedding_url,
            headers={"Authorization": f"Bearer {hf_token}"},
            json={"inputs": text},
        )
        if response.status_code != 200:
            raise ValueError(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
        return response.json()
    elif api_name == "openai":
        response = openai.Embedding.create(model=openai_model, input=text)
        return response["data"][0]["embedding"]


# for doc in collection.find({"plot": {"$exists": True}}).limit(50):
#     doc["plot_embedding_hf"] = generate_embedding(doc["plot"])
#     collection.replace_one({"_id": doc["_id"]}, doc)

query = "imaginary characters from outer space at war"

results = collection.aggregate(
    [
        {
            "$vectorSearch": {
                "queryVector": generate_embedding("huggingface", query),
                "path": "plot_embedding_hf",
                "numCandidates": 100,
                "limit": 4,
                "index": "PlotSemanticSearch",
            }
        }
    ]
)

for doc in results:
    print(
        f"Movie Name: {doc['title']}, Similarity: {doc['score']}, Plot: {doc['plot']}\n"
    )
