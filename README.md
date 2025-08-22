🔹 Overview

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline that:
Accepts a document (PDF, TXT, DOCX).
Extracts and chunks the text.
Generates vector embeddings using OpenAI Embedding Models (text-embedding-3-small).
Stores embeddings in Pinecone (vector database).
Stores metadata and relationships in Neo4j (graph database).
Provides a Streamlit web UI for file upload and monitoring.
The goal is to prepare data for downstream tasks such as semantic search, question-answering, or knowledge graph exploration.

🔹 Architecture
<img width="478" height="457" alt="image" src="https://github.com/user-attachments/assets/60ea2c96-ef77-46c2-b2b6-7158a3832fb3" />


🔹 Prerequisites

Python 3.10+
Dependencies:
pip install streamlit openai pinecone-client neo4j python-docx pypdf2 python-dotenv

Accounts & Keys:
OpenAI API Key
Pinecone API Key
Neo4j Aura
 (Free graph DB)

 🔹 Environment Setup
Create a .env file:
🔹 Environment Setup
OPENAI_API_KEY=your_openai_key

PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=rag-demo-index

NEO4J_URI=neo4j+ssc://<your-db-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

🔹 Streamlit UI (app.py)
Uploads a document.
Extracts & chunks text.
Generates embeddings.
Stores in Pinecone + Neo4j.
Shows progress + preview.

Run:
streamlit run app.py

Then open: http://localhost:8501

🔹 Backend Functions
Inside rag_upload.py you should have utilities like:
load_text(file) → Reads PDF/TXT/DOCX, returns plain text.
smart_chunk(text) → Splits into smaller overlapping chunks.
embed_chunks_openai(chunks) → Gets embeddings from OpenAI.
ensure_pinecone_index(pc, name, dimension) → Creates Pinecone index if not exists.
upsert_to_pinecone(pc, index_name, chunks, vectors, doc_id) → Stores vectors.
upsert_to_neo4j(driver, chunks, vectors, meta, doc_id) → Stores graph nodes.

🔹 Example Flow

Upload sample.pdf (668 chars).
System creates 1 chunk.
Generates 1536-dim embeddings with text-embedding-3-small.
Stores vector in Pinecone index (rag-demo-index).
Creates document + chunk nodes in Neo4j graph.
Confirms success in UI.

🔹 Neo4j Graph Schema
(:Document {id, filename, size})
(:Chunk {id, text, embedding_dim})

Relationship: (:Document)-[:HAS_CHUNK]->(:Chunk)
This allows queries like:
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
RETURN d.filename, count(c) as chunks

🔹 Pinecone Index Schema
Vector IDs: ${doc_id}::chunk_${i}
Vector values: 1536-dim embeddings.
Metadata: { "doc_id": ..., "chunk_id": ..., "text": ... }

