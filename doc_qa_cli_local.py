import argparse
import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------
# LOAD .env FILE
# -------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

print(f"Loaded model: {GROQ_MODEL}")

# -------------------------
# GROQ API CALL
# -------------------------
def query_groq_api(prompt: str) -> str:
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions using only the provided context. If the answer is not in the context, say 'I could not find this information in the document.'"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        error_detail = response.json() if response.content else {}
        print(f"Groq API Error: {e}")
        print(f"   Details: {error_detail}")
        return "API request failed."

# -------------------------
# LOAD DOCUMENT
# -------------------------
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported format. Use PDF, TXT, or DOCX.")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path
    return docs

# -------------------------
# SPLIT TEXT
# -------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

# -------------------------
# VECTOR STORE
# -------------------------
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)

# -------------------------
# RAG QUERY
# -------------------------
def answer_question(query: str, vectorstore) -> dict:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    source_docs = retriever.invoke(query)

    # Limit context to avoid exceeding token limit
    context_parts = []
    total_chars = 0
    for doc in source_docs:
        if total_chars + len(doc.page_content) > 3000:
            break
        context_parts.append(doc.page_content)
        total_chars += len(doc.page_content)

    context = "\n\n".join(context_parts)

    prompt = f"""Context from document:
{context}

Question: {query}

Answer based only on the context above:"""

    answer = query_groq_api(prompt)
    return {"result": answer, "source_documents": source_docs}

# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG CLI Tool with Groq API")
    parser.add_argument("file", help="Path to PDF, TXT, or DOCX file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("File not found.")
        return

    print("Loading document...")
    docs = load_document(args.file)

    print("Splitting text...")
    chunks = split_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    print("Creating embeddings...")
    vectorstore = create_vectorstore(chunks)

    print(f"\nAsk questions (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            result = answer_question(query, vectorstore)
            print("\n Answer:")
            print(result["result"])
            print("\n Sources:\n")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                preview = doc.page_content[:200].replace("\n", " ")
                print(f"{i}.  {source}")
                print(f"    {preview}...\n")
        except Exception as e:
            print(f" Error: {e}\n")

if __name__ == "__main__":
    main()