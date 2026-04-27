📚 RAG Chatbot (Retrieval-Augmented Generation)

A Natural Language Processing (NLP) project that implements a RAG (Retrieval-Augmented Generation) chatbot.
The system takes a document as input and allows users to ask questions, returning accurate, context-aware answers by combining information retrieval with generative AI.

🚀 Features
📄 Upload and process documents (PDF/Text)
🔍 Retrieve relevant chunks from documents
🤖 Generate intelligent answers using LLMs
🧠 RAG pipeline (Retrieval + Generation)
💬 Interactive question-answering chatbot
⚡ Fast and context-aware responses
🏗️ Architecture

The system follows the RAG pipeline:

Document Loading – Input document is loaded and cleaned
Chunking – Text is split into smaller meaningful segments
Embedding – Chunks are converted into vector representations
Retrieval – Relevant chunks are fetched based on user query
Generation – LLM generates final response using retrieved context
