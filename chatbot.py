
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_mistralai.chat_models import ChatMistralAI   
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


load_dotenv()
DATA_PATH = "knowledge_base.txt"
def load_documents(file_path):
    """Loads documents from a text file."""
    print(f"Loading data from: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    return documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    return texts

def create_vector_store(texts, embeddings_model):
    """Creates a FAISS vector store from text chunks and an embedding model."""
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(texts, embeddings_model)
    print("Vector store created.")
    return vectorstore

def initialize_rag_pipeline():
    """
    Initializes the full RAG pipeline using Mistral AI:
    1. Loads data
    2. Splits documents
    3. Creates Mistral embeddings and vector store
    4. Sets up the Mistral LLM and RetrievalQA chain
    """
    # 1. Load Mistral API Key
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")

    # 2. Load and process documents
    documents = load_documents(DATA_PATH)
    texts = split_documents(documents)

    # 3. Initialize Mistral embeddings model
    print("Initializing Mistral embeddings model...")

    embeddings = MistralAIEmbeddings(api_key=mistral_api_key)
    print("Mistral embeddings model initialized.")

    # 4. Create vector store
    vectorstore = create_vector_store(texts, embeddings)

    # 5. Initialize Mistral LLM
    print("Initializing Mistral LLM...")
   
    llm = ChatMistralAI(
        api_key=mistral_api_key,
        model="mistral-small-latest",
        temperature=0.1,
        max_retries=3 
    )
    print("Mistral LLM initialized.")

    # 6. Create Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )

    # 7. Create a custom prompt
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
    Provide a concise answer. If the context includes specific details like pricing or names, please use them.

    Context:
    {context}

    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    # 8. Create RetrievalQA chain
    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("RAG pipeline initialized successfully with Mistral AI.")
    return qa_chain


def run_chatbot(qa_chain):
    """Runs the interactive command-line chatbot."""
    print("\n--- BFWA IntelliBot (Powered by Mistral AI) ---")
    print("Ask me anything about BuildFastWithAI or IntelliBot.")
    print("Type 'exit' or 'quit' to end the chat.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break

        if not user_query.strip():
            print("Bot: Please ask a question.")
            continue

        try:
            print("Bot: Thinking...")
            result_dict = qa_chain({"query": user_query})
            answer = result_dict.get('result', "Sorry, I couldn't process that.")
            
            print(f"Bot: {answer}")

            source_documents = result_dict.get('source_documents')
            if source_documents:
                print("\nSource documents used:")
                for i, doc in enumerate(source_documents):
                    print(f"  Source {i+1}: \"{doc.page_content[:100]}...\"")

        except Exception as e:
            print(f"Bot: An error occurred: {e}")

# Main execution
if __name__ == "__main__":
    try:
        rag_chain = initialize_rag_pipeline()
        run_chatbot(rag_chain)
    except ValueError as ve:
        print(f"Initialization Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")