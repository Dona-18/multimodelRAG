# Local RAG architecture

Documents are split into overlapping chunks before they are embedded.
`nomic-embed-text:latest` creates local embeddings. ChromaDB stores the vectors
with cosine distance. Retrieval combines semantic vector search and BM25 keyword
search with Reciprocal Rank Fusion. `gemma4:e2b-mlx` writes grounded answers.
