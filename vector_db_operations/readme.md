# MongoDB Vector Store Population and Retrieval

This guide helps you populate a MongoDB Atlas Vector Store for AI applications, like "About Me" LLMs, and perform vector similarity searches on your data to answer questions about you.

## 1. Project Overview

The `insert_data_into_vector_db.py` loads JSON data, chunks text, generates vector embeddings using sentence-transformers, and stores them in MongoDB. An optional `task2` can aggregate and embed data from other MongoDB collections.

## 2. Setting Up Your Data (JSON File)

Populate your personal information in a JSON file using the provided structure. Fill in the "text" fields for each "title". You can add new sections (objects with "title" and "text") or modify existing ones.

### Example Structure:

```json
[
   {"title":"General Information", "text":"Your general text."},
   {"title":"area of intrest", "text":"Your interests."},
   // ... other sections
] 
```

Note for task1: The process_json_data function expects the loaded JSON data to be a dictionary containing lists of strings. If your JSON is an array of objects (like the example above), consider wrapping it in a dictionary: {"my_data": [...]} or adapt task1 to directly process the list.

## 3. Configuration and Script Modifications
Edit mongodb_embedding_script.py to set up your MongoDB connections:

### Global Variables:

MONGO_URI: Your MongoDB Atlas connection string.

DB_NAME: Your database name.

COLLECTION_NAME: Your embedding collection name.

MONGO_URI = "YOUR_MONGODB_ATLAS_CONNECTION_STRING"
DB_NAME = "your_database_name"
COLLECTION_NAME = "your_embedding_collection_name"

## task2 Details (if used):

MONGO_URI2, DB_NAME2, COLLECTION_NAME2: For source collections (e.g. data stored in mongo db collections). Ensure collection is in DB_NAME2.

def task2():
    MONGO_URI2 = "YOUR_SECOND_MONGODB_ATLAS_CONNECTION_STRING"
    DB_NAME2 = "your_source_database_name"
    COLLECTION_NAME2 = "your_source_project_collection_name"
    # ...

JSON File Path: Update json_file_path in if __name__ == "__main__":.

```python

if __name__ == "__main__":
    json_file_path = r"C:\path\to\your\data_about_me.json"
    task1(json_file_path)
    # task2() # Uncomment to run task2
```

## 4. Running the Script
```
Install: pip install pymongo sentence-transformers tqdm langchain

Configure: Update mongodb_embedding_script.py settings.

Execute: python mongodb_embedding_script.py
```

This will populate your MongoDB collection.

## 5. Testing Retrieval Code (Similarity Search)
After populating, use langchain's MongoDBAtlasVectorSearch for retrieval.

Requirements:
MongoDB Atlas Vector Search Index: Create an index on COLLECTION_NAME for the embedding field. Crucially, replace "default" with your actual index name.
```

# Match your MongoDB connection details
MONGO_URI = "YOUR_MONGODB_ATLAS_CONNECTION_STRING"
DB_NAME = "your_database_name"
COLLECTION_NAME = "your_embedding_collection_name"

client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    text_key="text",
    embedding_key="embedding",
    index_name="YOUR_ATLAS_SEARCH_INDEX_NAME", # !!! IMPORTANT: Replace this !!!
    relevance_score_fn="cosine"
)

query = "Tell me about Ananth's work experience"
results = vector_store.similarity_search(query=query, k=5)

print(f"--- Top {len(results)} results for query: '{query}' ---")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:\nContent: {result.page_content}")
```