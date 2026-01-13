# Update code 

import os
import joblib
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Define Constants ---
VECTOR_STORE_PATH = 'vector_stores'
ROUTER_MODEL_PATH = 'router_model.joblib'
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- 3. Load All Components ---
print("Loading components...")

# Load Router
try:
    router_model = joblib.load(ROUTER_MODEL_PATH)
    print("ML router model loaded.")
except FileNotFoundError:
    print(f"Error: Router model not found at {ROUTER_MODEL_PATH}")
    exit()

# Load Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Load LLM (Updated to the correct working model)
llm = ChatGroq(model="llama-3.1-8b-instant")
print("Gen AI model (Groq/Llama3) loaded.")

# Load Vector Stores
vector_stores = {}
try:
    if os.path.exists(VECTOR_STORE_PATH):
        for topic in os.listdir(VECTOR_STORE_PATH):
            topic_path = os.path.join(VECTOR_STORE_PATH, topic)
            if os.path.isdir(topic_path):
                print(f"Loading vector store for topic: {topic}")
                vector_stores[topic] = FAISS.load_local(
                    topic_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True 
                )
    print(f"Loaded {len(vector_stores)} vector stores.")
except Exception as e:
    print(f"Error loading vector stores: {e}")
    exit()

# --- 4. Define the NEW "Hybrid" Logic ---

# [CHANGE 1] UPDATED PROMPT
# We changed the instructions to allow the AI to use its own knowledge
# if the context is empty or irrelevant.
prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert assistant. 

First, look at the "Context" provided below. 
- If the context contains the answer to the user's question, strictly use that information.
- If the context is empty, irrelevant, or does not contain the answer, IGNORE the context and answer the question using your own general knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)

def process_query(query_text: str) -> str:
    # === STEP 5.1: The "ML" part (Router) ===
    predicted_topic = router_model.predict([query_text])[0]
    print(f"   [Debug] Router prediction: {predicted_topic}")

    # === STEP 5.2: The "RAG" part (Retriever) ===
    context = ""
    
    if predicted_topic in vector_stores:
        selected_store = vector_stores[predicted_topic]
        
        # We retrieve 2 documents, but now the LLM can ignore them if they are bad matches
        retriever = selected_store.as_retriever(search_kwargs={"k": 2})
        context_docs = retriever.invoke(query_text)
        context = "\n---\n".join([doc.page_content for doc in context_docs])
    else:
        # [CHANGE 2] HANDLE MISSING STORES
        # If no store is found (or prediction is off), we pass empty context.
        # This forces the LLM to use its own knowledge immediately.
        print(f"   [Debug] No vector store for '{predicted_topic}'. Using general knowledge.")
        context = "" 

    # === STEP 5.3: The "Gen AI" part (Generator) ===
    rag_chain = (
        {
            "context": lambda x: context,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(query_text)
    return answer

# --- 5. Run the Application ---

if __name__ == "__main__":
    print("\n--- Intelligent Hybrid Chatbot ---")
    print("Ask ANYTHING. I will check my database first, then use my brain.")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            break
        
        print("...thinking...")
        try:
            response = process_query(query)
            print("\nAnswer:", response)
        except Exception as e:
            print(f"An error occurred: {e}")

# old Version

# import os
# import joblib
# from dotenv import load_dotenv

# # --- LangChain Imports ---
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # --- 1. Load Environment Variables (Groq API Key) ---
# load_dotenv()

# # --- 2. Define Constants ---
# VECTOR_STORE_PATH = 'vector_stores'
# ROUTER_MODEL_PATH = 'router_model.joblib'
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Must be the same as in Step 3

# # --- 3. Load All Components ---

# print("Loading components...")

# # Load ML Router Model
# try:
#     router_model = joblib.load(ROUTER_MODEL_PATH)
#     print("ML router model loaded.")
# except FileNotFoundError:
#     print(f"Error: Router model not found at {ROUTER_MODEL_PATH}")
#     print("Please run train_router.py first.")
#     exit()

# # Load Embedding Function
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# # Load LLM (Gen AI Model)
# # We use Llama 3 8B via Groq
# llm = ChatGroq(model="llama-3.1-8b-instant")
# print("Gen AI model (Groq/Llama3) loaded.")

# # Load all RAG Vector Stores into a dictionary
# vector_stores = {}
# try:
#     for topic in os.listdir(VECTOR_STORE_PATH):
#         topic_path = os.path.join(VECTOR_STORE_PATH, topic)
#         if os.path.isdir(topic_path):
#             print(f"Loading vector store for topic: {topic}")
#             # IMPORTANT: We need allow_dangerous_deserialization=True for FAISS
#             vector_stores[topic] = FAISS.load_local(
#                 topic_path, 
#                 embeddings, 
#                 allow_dangerous_deserialization=True 
#             )
#     print(f"Loaded {len(vector_stores)} vector stores.")
# except FileNotFoundError:
#     print(f"Error: Vector stores directory not found at {VECTOR_STORE_PATH}")
#     print("Please run create_vector_stores.py first.")
#     exit()

# if not vector_stores:
#     print("No vector stores were loaded. Exiting.")
#     exit()

# # --- 4. Define the RAG Logic (The "Chain") ---

# # This is the prompt template for the Gen AI
# # It instructs the AI to answer based *only* on the provided context
# prompt_template = ChatPromptTemplate.from_template(
#     """
# You are an expert assistant. Answer the user's question based ONLY 
# on the context provided below.

# If the context doesn't contain the answer, just say: 
# "I'm sorry, I don't have information on that topic."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

# def process_query(query_text: str) -> str:
#     """
#     This function processes the user's query from start to finish.
#     """
    
#     # === STEP 5.1: The "ML" part (Router) ===
#     # Use our trained ML model to predict the topic
#     predicted_topic = router_model.predict([query_text])[0]
#     print(f"   [Debug] Router prediction: {predicted_topic}")

#     # === STEP 5.2: The "RAG" part (Retriever) ===
#     # Select the correct vector store based on the prediction
#     if predicted_topic in vector_stores:
#         selected_store = vector_stores[predicted_topic]
        
#         # Get the retriever for this store (we'll ask for 2 relevant chunks)
#         retriever = selected_store.as_retriever(search_kwargs={"k": 2})
        
#         # Get the relevant documents
#         context_docs = retriever.invoke(query_text)
        
#         # Format the context to be passed to the LLM
#         context = "\n---\n".join([doc.page_content for doc in context_docs])
#     else:
#         print(f"   [Debug] Warning: No vector store for topic '{predicted_topic}'.")
#         context = "No information found."

#     # === STEP 5.3: The "Gen AI" part (Generator) ===
    
#     # We create a "chain" to define the data flow
#     rag_chain = (
#         {
#             "context": lambda x: context,  # Pass in the context we retrieved
#             "question": RunnablePassthrough() # Pass in the original question
#         }
#         | prompt_template
#         | llm
#         | StrOutputParser()
#     )
    
#     # Run the chain to get the final answer
#     answer = rag_chain.invoke(query_text)
#     return answer

# # --- 5. Run the Application ---

# if __name__ == "__main__":
#     print("\n--- Intelligent RAG Chatbot ---")
#     print("Ask a question about Python (tech), snakes (biology), or Monty Python (comedy).")
#     print("Type 'exit' to quit.")
    
#     while True:
#         query = input("\nYour Question: ")
#         if query.lower() == 'exit':
#             break
        
#         print("...thinking...")
#         response = process_query(query)

#         print("\nAnswer:", response)
