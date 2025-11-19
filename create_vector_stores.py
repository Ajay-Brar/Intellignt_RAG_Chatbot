import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Define paths
DOCUMENTS_PATH = 'documents'
VECTOR_STORE_PATH = 'vector_stores'

# Define the embedding model to use
# This is a popular, high-quality, and free model
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def create_vector_store_for_topic(topic):
    """
    Creates a FAISS vector store for a specific topic.
    """
    topic_doc_path = os.path.join(DOCUMENTS_PATH, topic)
    topic_vec_path = os.path.join(VECTOR_STORE_PATH, topic)

    print(f"--- Processing topic: {topic} ---")

    # 1. Check if the document directory exists
    if not os.path.exists(topic_doc_path):
        print(f"Directory not found: {topic_doc_path}. Skipping.")
        return

    # 2. (Optional) Clean up old vector store if it exists
    if os.path.exists(topic_vec_path):
        print(f"Removing old vector store: {topic_vec_path}")
        shutil.rmtree(topic_vec_path)

    # 3. Load documents from the topic directory
    # We specify TextLoader to ensure we only read .txt files
    loader = DirectoryLoader(
        topic_doc_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()
    
    if not documents:
        print(f"No .txt documents found in {topic_doc_path}. Skipping.")
        return

    print(f"Loaded {len(documents)} document(s) from {topic_doc_path}")

    # 4. Split documents into chunks
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    # 5. Create the FAISS vector store from the chunks
    print("Creating vector store...")
    vector_store = FAISS.from_documents(docs, embeddings)

    # 6. Save the vector store locally
    vector_store.save_local(topic_vec_path)
    print(f"Vector store saved successfully to {topic_vec_path}\n")


if __name__ == "__main__":
    # Get the list of topics from the subdirectories in 'documents'
    topics = [d for d in os.listdir(DOCUMENTS_PATH) if os.path.isdir(os.path.join(DOCUMENTS_PATH, d))]
    
    if not topics:
        print(f"No topic subdirectories found in '{DOCUMENTS_PATH}'.")
        print("Please create subdirectories like 'documents/tech', 'documents/biology', etc.")
    else:
        print(f"Found topics: {topics}")
        
        # Create the main vector_stores directory if it doesn't exist
        if not os.path.exists(VECTOR_STORE_PATH):
            os.makedirs(VECTOR_STORE_PATH)
        
        # Process each topic
        for topic in topics:
            create_vector_store_for_topic(topic)
            
        print("--- All vector stores created successfully! ---")