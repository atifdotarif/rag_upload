import os
import argparse
import hashlib
import time
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
import docx2txt

try:
    import tiktoken
except Exception:
    tiktoken = None

from openai import OpenAI

# Pinecone v4/v5 style client
from pinecone import Pinecone, ServerlessSpec

# Neo4j driver
from neo4j import GraphDatabase


# ----------------------------
# Helpers: Load & Chunk
# ----------------------------

def load_text(file_path: str) -> Tuple[str, Dict]:
    """
    Returns (text, metadata)
    Supports: .txt, .pdf, .docx
    """
    ext = os.path.splitext(file_path)[1].lower()
    meta = {
        "filename": os.path.basename(file_path),
        "path": os.path.abspath(file_path),
    }

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        meta["filetype"] = "txt"

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        meta["filetype"] = "pdf"
        meta["pages"] = len(reader.pages)

    elif ext == ".docx":
        text = docx2txt.process(file_path) or ""
        meta["filetype"] = "docx"

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt, .pdf, or .docx")

    # Normalize whitespace
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text, meta


def count_tokens(s: str, model: str = "text-embedding-3-small") -> int:
    """
    Approximate token count for chunking. Falls back to char-based approximation if tiktoken missing.
    """
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    # Fallback: ~4 chars per token (rough). Good enough for chunking.
    return max(1, len(s) // 4)


def smart_chunk(
    text: str,
    chunk_tokens: int = 800,
    overlap_tokens: int = 150
) -> List[str]:
    """
    Token-aware chunking with sentence boundaries where possible.
    If tiktoken not available, uses char approximation.
    """
    # light sentence split
    sentences = []
    buff = []
    for line in text.split("\n"):
        if not line:
            continue
        parts = [p.strip() for p in line.replace("?", "?.").replace("!", "!.").split(".")]
        for p in parts:
            if p:
                sentences.append(p.strip())

    chunks = []
    current = ""
    current_tokens = 0

    def add_sentence(to_add: str):
        nonlocal current, current_tokens, chunks
        st = to_add if to_add.endswith(".") else (to_add + ".")
        new_tokens = count_tokens(st)
        if current_tokens + new_tokens > chunk_tokens and current:
            chunks.append(current.strip())
            current = st
            current_tokens = new_tokens
        else:
            current = (current + " " + st).strip() if current else st
            current_tokens += new_tokens

    for sent in sentences:
        add_sentence(sent)

    if current:
        chunks.append(current.strip())

    # add overlap by re-stitching last ~overlap_tokens from previous
    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
                continue
            prev = overlapped[-1]
            # take last overlap_tokens worth from prev
            prev_tail = prev.split()
            # rough token ≈ word here
            tail = " ".join(prev_tail[-overlap_tokens:]) if len(prev_tail) > overlap_tokens else prev
            overlapped.append((tail + " " + ch).strip())
        chunks = overlapped

    # final trim
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


# ----------------------------
# Embeddings
# ----------------------------

def embed_chunks_openai(
    chunks: List[str],
    model: str,
    batch_size: int = 128
) -> List[List[float]]:
    """
    Embeds chunks via OpenAI in batches.
    """
    # New client style (picks up key automatically from env var if set)
    client = OpenAI()

    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            vectors.append(d.embedding)
        time.sleep(0.1)  # gentle pacing
    return vectors



# ----------------------------
# Pinecone
# ----------------------------

def ensure_pinecone_index(pc: Pinecone, name: str, dimension: int, metric: str = "cosine",
                          cloud: str = "aws", region: str = "us-east-1"):
    """
    Creates the index if it doesn't exist (serverless).
    """
    try:
        existing = [idx["name"] if isinstance(idx, dict) else getattr(idx, "name", None)
                    for idx in pc.list_indexes()]
        if name not in existing:
            pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            # wait a bit for index to be ready
            time.sleep(2)
    except Exception as e:
        # If it's a harmless "already exists" kind of error, continue.
        if "already exists" not in str(e).lower():
            raise


def upsert_to_pinecone(
    pc: Pinecone,
    index_name: str,
    chunks: List[str],
    vectors: List[List[float]],
    doc_id: str,
    namespace: str = "default",
    batch_size: int = 100
):
    index = pc.Index(index_name)
    items = []
    for i, (text, vec) in enumerate(zip(chunks, vectors)):
        items.append({
            "id": f"{doc_id}::chunk::{i}",
            "values": vec,
            "metadata": {
                "doc_id": doc_id,
                "chunk_index": i,
                "text": text[:4000]  # PC metadata limit is generous but keep tidy
            }
        })
        if len(items) == batch_size:
            index.upsert(vectors=items, namespace=namespace)
            items = []
    if items:
        index.upsert(vectors=items, namespace=namespace)


# ----------------------------
# Neo4j
# ----------------------------

def init_neo4j_constraints_and_index(session, dim: int):
    """
    Create uniqueness constraints and (if supported) a vector index.
    Neo4j 5.11+ supports native VECTOR INDEX. If not, this will just skip.
    """
    session.run("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;")
    session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;")
    # Try to create vector index (Neo4j 5.11+)
    try:
        session.run("""
        CREATE VECTOR INDEX chunk_vector IF NOT EXISTS
        FOR (c:Chunk) ON (c.vector)
        OPTIONS { indexConfig: { `vector.dimensions`: $dim, `vector.similarity_function`: 'cosine' } }
        """, {"dim": dim})
    except Exception:
        pass  # Older Neo4j will not support this; it's fine for upload-only phase.


def upsert_to_neo4j(
    driver: GraphDatabase.driver,
    chunks: List[str],
    vectors: List[List[float]],
    doc_meta: Dict,
    doc_id: str
):
    """
    Create (Document) and (Chunk) nodes with HAS_CHUNK relationships.
    """
    with driver.session() as session:
        init_neo4j_constraints_and_index(session, dim=len(vectors[0]))

        # MERGE Document
        session.run("""
            MERGE (d:Document {id: $id})
            SET d.filename = $filename,
                d.path = $path,
                d.filetype = $filetype,
                d.pages = $pages
        """, {
            "id": doc_id,
            "filename": doc_meta.get("filename"),
            "path": doc_meta.get("path"),
            "filetype": doc_meta.get("filetype"),
            "pages": doc_meta.get("pages", 0)
        })

        # Batch create chunks
        payload = []
        for i, (text, vec) in enumerate(zip(chunks, vectors)):
            payload.append({
                "cid": f"{doc_id}::chunk::{i}",
                "idx": i,
                "text": text,
                "vec": vec,
                "doc_id": doc_id
            })

        session.run("""
            UNWIND $rows AS row
            MERGE (c:Chunk {id: row.cid})
            SET c.chunk_index = row.idx,
                c.text = row.text,
                c.vector = row.vec
            WITH c, row
            MATCH (d:Document {id: row.doc_id})
            MERGE (d)-[:HAS_CHUNK]->(c)
        """, {"rows": payload})


# ----------------------------
# Main pipeline
# ----------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG Upload: chunk → embed → store (Pinecone + Neo4j)")
    parser.add_argument("--file", required=True, help="Path to document (.pdf/.txt/.docx)")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--chunk_tokens", type=int, default=800, help="approx tokens per chunk")
    parser.add_argument("--overlap_tokens", type=int, default=150, help="approx overlap tokens between chunks")
    parser.add_argument("--namespace", default="default", help="Pinecone namespace")

    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")

    pc_key = os.getenv("PINECONE_API_KEY")
    pc_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-demo-index")
    pc_cloud = os.getenv("PINECONE_CLOUD", "aws")
    pc_region = os.getenv("PINECONE_REGION", "us-east-1")

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not pc_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment")
    if not (neo4j_uri and neo4j_user and neo4j_password):
        print("[WARN] Neo4j env vars missing; skipping Neo4j step.")
        neo4j_uri = neo4j_user = neo4j_password = None

    # 1) Load
    text, meta = load_text(args.file)
    if not text.strip():
        raise RuntimeError("No text extracted from the file.")
    print(f"[OK] Loaded text from {meta['filename']} ({len(text)} chars)")

    # Create a deterministic doc_id (hash)
    doc_hash = hashlib.sha256((meta["filename"] + str(len(text))).encode("utf-8")).hexdigest()[:16]
    doc_id = f"{meta['filename']}::{doc_hash}"

    # 2) Chunk
    chunks = smart_chunk(text, chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens)
    print(f"[OK] Created {len(chunks)} chunks (avg ~{sum(len(c) for c in chunks)//max(1,len(chunks))} chars/chunk)")

    # 3) Embeddings
    dim = 1536 if "text-embedding-3-small" in args.model else 3072
    vectors = embed_chunks_openai(chunks, model=args.model, batch_size=128)
    assert len(vectors) == len(chunks), "Mismatch between chunks and vectors"
    print(f"[OK] Embedded {len(vectors)} chunks with model {args.model} (dim={len(vectors[0])})")

    # 4a) Pinecone
    pc = Pinecone(api_key=pc_key)
    ensure_pinecone_index(pc, name=pc_index_name, dimension=dim, cloud=pc_cloud, region=pc_region)
    upsert_to_pinecone(pc, pc_index_name, chunks, vectors, doc_id, namespace=args.namespace)
    # Quick PC stats
    try:
        stats = pc.Index(pc_index_name).describe_index_stats()
        total = stats.get("total_vector_count") or stats.get("vectors", {}).get("count")
        print(f"[OK] Pinecone upsert complete. Approx vectors in index: {total}")
    except Exception:
        print("[INFO] Pinecone stats unavailable (non-fatal).")

    # 4b) Neo4j
    if neo4j_uri:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        upsert_to_neo4j(driver, chunks, vectors, meta, doc_id)
        with driver.session() as session:
            cnt = session.run("""
                MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS n
            """, {"id": doc_id}).single()["n"]
        print(f"[OK] Neo4j upsert complete. Chunks linked to Document: {cnt}")

    print("\nAll done (Upload Phase).\n")


if __name__ == "__main__":
    main()
