import os
import chromadb

# Path to the Chroma database directory (not the file itself)
db_path = r"D:\projects\Social_service_chatbot\indigobot\langchain_app\CCC_scraper\rag_data\.chromadb\openai"


# Verify if the database directory exists
if not os.path.exists(db_path):
    print(f"Database path does not exist: {db_path}")
else:
    try:
        # Connect to the existing Chroma database
        client = chromadb.PersistentClient(path=db_path)

        # List all collections in the database
        collections = client.list_collections()
        if not collections:
            print("The database contains no collections.")
        else:
            print("Available Collections:")
            for collection in collections:
                print(f" - {collection.name}")

            # Example: Access the first collection (replace with your desired collection)
            collection_name = collections[0].name  # Select the first collection
            collection = client.get_collection(collection_name)

            # Fetch all data from the collection
            results = collection.get()

            # Display the data
            print("\nCollection Data:")
            print("IDs:", results.get("ids", []))
            print("Embeddings:", results.get("embeddings", []))
            print("Metadata:", results.get("metadatas", []))

    except Exception as e:
        print(f"An error occurred: {e}")
