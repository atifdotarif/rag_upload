import streamlit as st
import os
import hashlib
from dotenv import load_dotenv
from rag_upload import load_text, smart_chunk, embed_chunks_openai, upsert_to_pinecone, ensure_pinecone_index, upsert_to_neo4j
from pinecone import Pinecone
from neo4j import GraphDatabase

load_dotenv()

# Environment
pc_key = os.getenv("PINECONE_API_KEY")
pc_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-demo-index")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# UI
st.title("📂 RAG Upload")
st.write("Upload a PDF, TXT, or DOCX and store embeddings in Pinecone + Neo4j.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])

if uploaded_file:
    # Save temp file
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1) Load text
    text, meta = load_text(uploaded_file.name)
    st.success(f"Loaded {len(text)} chars from {meta['filename']}")

    # 2) Chunk
    chunks = smart_chunk(text)
    st.write(f"✅ Created {len(chunks)} chunks")

    # 3) Embedding
    vectors = embed_chunks_openai(chunks, model="text-embedding-3-small")
    st.write(f"✅ Embedded {len(vectors)} chunks")

    # 4) Pinecone
    pc = Pinecone(api_key=pc_key)
    ensure_pinecone_index(pc, name=pc_index_name, dimension=len(vectors[0]))
    doc_id = f"{meta['filename']}::{hashlib.sha256((meta['filename']+str(len(text))).encode()).hexdigest()[:16]}"
    upsert_to_pinecone(pc, pc_index_name, chunks, vectors, doc_id)
    st.write("✅ Stored in Pinecone")

    # 5) Neo4j
    if neo4j_uri:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        upsert_to_neo4j(driver, chunks, vectors, meta, doc_id)
        st.write("✅ Stored in Neo4j")

    st.success("Upload completed successfully!")
    st.write("Here’s a preview of your first chunk:")
    st.code(chunks[0][:500])
