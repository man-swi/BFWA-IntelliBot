import streamlit as st
import os
from dotenv import load_dotenv

try:
    
    from chatbot import initialize_rag_pipeline
except ImportError:
    st.error("Failed to import 'initialize_rag_pipeline' from chatbot.py. Ensure it's in the same directory.")

    def initialize_rag_pipeline():
        st.warning("Using dummy RAG pipeline due to import error.")
        class DummyChain:
            def __call__(self, inputs): 
                return {"result": f"Dummy response to: {inputs['query']}", "source_documents": []}
        return DummyChain()


load_dotenv()


st.set_page_config(page_title="BFWA IntelliBot (Mistral AI)", layout="wide")
st.title("📚 BFWA IntelliBot - RAG Powered by Mistral AI")
st.markdown("""
Welcome to the BuildFastWithAI (BFWA) IntelliBot!
Ask questions about our company, products (like IntelliBot), pricing, or mission.
This chatbot uses a Retrieval-Augmented Generation (RAG) pipeline with **Mistral AI** to answer your questions based on our knowledge base.
""")

@st.cache_resource 
def get_rag_chain():
    """Initializes and returns the RAG chain (Mistral-based)."""
    try:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            st.error("MISTRAL_API_KEY not found. Please set it in your .env file or environment variables.")
            return None
        with st.spinner("Initializing RAG pipeline with Mistral AI... This may take a moment."):
        
            chain = initialize_rag_pipeline()
        st.success("RAG pipeline with Mistral AI initialized successfully!")
        return chain
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        return None

rag_chain = get_rag_chain()


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm powered by Mistral AI. How can I help you today regarding BFWA or IntelliBot?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.caption(f"Source {i+1}:")
                    st.markdown(f"_{source.page_content[:200]}..._")


# Chat input
if prompt := st.chat_input("Ask your question here..."):
    if not rag_chain:
        st.error("RAG chain is not initialized. Cannot process query.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""
            source_docs_content = [] # To store source documents for the current response

            with st.spinner("Thinking with Mistral AI..."):
                try:
                    response = rag_chain({"query": prompt})
                    
                    answer = response.get('result', "Sorry, I couldn't find an answer.")
                    full_response_content = answer

                    
                    source_documents = response.get('source_documents', [])
                    if source_documents:
                        source_docs_content = source_documents
                        

                except Exception as e:
                    full_response_content = f"Sorry, an error occurred: {str(e)}"
                    st.error(f"Error processing query: {e}")

            message_placeholder.markdown(full_response_content)
         
            if source_docs_content:
                 with st.expander("View Sources Used for This Response"):
                    for i, source in enumerate(source_docs_content):
                        st.caption(f"Source {i+1}:")
                        st.markdown(f"_{source.page_content[:200]}..._")


        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response_content,
            "sources": source_docs_content # Store sources with the message in session state
        })
else:

    if not rag_chain:
        st.warning("RAG chain (Mistral AI) failed to initialize. Please check your API key and console for errors.")

# Sidebar for information and actions
st.sidebar.header("About")
st.sidebar.info(
    "This is a demo RAG chatbot built with LangChain and Streamlit, powered by **Mistral AI**, for the BuildFastWithAI assignment. "
    "It answers questions based on a predefined knowledge base about BFWA."
)
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat History"):
  
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm powered by Mistral AI. How can I help you today regarding BFWA or IntelliBot?"}]
    st.rerun() 