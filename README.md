# RAG Chatbot with Mistral AI & LangChain

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on a custom knowledge base. It uses Mistral AI for language understanding, LangChain for the RAG pipeline, and Streamlit for a simple web interface.

## What it Does

The chatbot takes a user's question, searches a local text file (`knowledge_base.txt`) for relevant information, and then uses Mistral AI's language model to generate an answer based on that information. This ensures answers are grounded in the provided data.

## How It's Done (Core Technologies)

1.  **Knowledge Base:** A plain text file (`knowledge_base.txt`) stores the information the chatbot can access.
2.  **LangChain:**
    *   Loads and splits the knowledge base into manageable chunks.
    *   Uses `MistralAIEmbeddings` to create numerical representations (embeddings) of these chunks.
    *   Stores these embeddings in a `FAISS` vector store for efficient searching.
    *   Employs a `RetrievalQA` chain which:
        *   Takes a user query.
        *   Finds relevant text chunks from FAISS.
        *   Sends the query and relevant chunks to `ChatMistralAI`.
3.  **Mistral AI:**
    *   Provides the embedding model to convert text to vectors.
    *   Provides the chat model (`mistral-small-latest`) to generate answers.
4.  **Streamlit:**
    *   Creates an interactive chat interface (`app.py`) for users to ask questions.
5.  **Environment Variables:**
    *   Your Mistral AI API key is stored securely in a `.env` file (which is ignored by Git).

## How to Run

### Prerequisites

*   Python 3.9+
*   A Mistral AI API Key

### Setup

1.  **Clone the repository (if applicable).**
2.  **Create and activate a Python virtual environment (recommended).**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up API Key:**
    Create a `.env` file in the project root and add your Mistral API key:
    ```env
    MISTRAL_API_KEY="your_mistral_api_key_here"
    ```
    **(Ensure `.env` is in your `.gitignore` file!)**

### Running the Chatbot

You have two options:

1.  **Streamlit Web App (Recommended):**
    ```bash
    streamlit run app.py
    ```
    Open the URL provided in your terminal (usually `http://localhost:8501`).

2.  **Command-Line Interface:**
    ```bash
    python chatbot.py
    ```

---
*This project demonstrates building a RAG system with modern AI tools.*
