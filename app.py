import streamlit as st
import ollama
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Setup Connections
AZURE_VM_IP = os.getenv("AZURE_VM_IP")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

o_client = ollama.Client(host=f'http://{AZURE_VM_IP}')
q_client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

st.title("Ronit's AI Portfolio Assistant")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Accept User Input
if prompt := st.chat_input("Ask me about Ronit's projects..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Retrieval Augmented Generation (RAG)
    with st.chat_message("assistant"):
        try:
            # Get Embeddings from Azure
            embed = o_client.embeddings(model="nomic-embed-text", prompt=prompt)['embedding']
            
            # Search Qdrant Cloud
            st.write(f"Embedding Length: {len(embed)}") # Should be 768
            docs = q_client.search(collection_name="portfolio_brain", query_vector=embed, limit=3)
            context = "\n".join([d.payload['content'] for d in docs])
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            context = "No portfolio context available yet."

        # Generate Response using Llama 3.2 1B
        response_stream = o_client.chat(
            model='llama3.2:1b',
            messages=[
                {'role': 'system', 'content': f"You are Ronit's Assistant. Context: {context}"},
                {'role': 'user', 'content': prompt},
            ],
            stream=True # Use streaming for better UX
        )
        
        # Display with Typewriter Effect
        def stream_data():
            for chunk in response_stream:
                yield chunk['message']['content']
        
        full_response = st.write_stream(stream_data())
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
