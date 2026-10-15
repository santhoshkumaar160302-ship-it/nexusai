# Retrieval-Augmented Generation

The RAG pipeline answers questions from your own documents, with citations, through the same gateway that serves chat requests.

## Ingestion

POST /v1/rag/ingest accepts a list of documents, each with a title, the text and an optional source. Documents are split into chunks, embedded and stored in the chunks table in pgvector.

The chunker is structure-aware. It splits Markdown at headings and records each chunk's heading path, such as "Setup > Docker". It packs whole paragraphs into chunks of up to RAG_CHUNK_SIZE characters, 800 by default, and starts each chunk with the last RAG_CHUNK_OVERLAP characters of the previous chunk, 150 by default, aligned to a sentence boundary. Chunks never cross a section boundary.

Before a chunk is embedded, the document title and heading path are prepended to it. A chunk that only says "set it to 30 seconds" becomes retrievable for questions about timeouts once its heading is attached.

Ingestion is idempotent. Each document has a content hash, and re-ingesting unchanged content is skipped without calling the embedding API.

## Retrieval

Retrieval is hybrid. The retriever first fetches four times as many chunks as requested by vector similarity. It then re-ranks them with a keyword score weighted by inverse document frequency, which rescues exact terms such as configuration keys and error codes. The final score is 0.75 times the vector score plus 0.25 times the keyword score. Chunks below RAG_MIN_SCORE vector similarity (0.25 with OpenAI embeddings) are dropped, and at most three chunks come from any one document so a single long document cannot crowd out the others.

## Answer generation

The retrieved chunks are numbered and placed in the prompt, and the model is instructed to cite every claim with the source number in brackets and to say "I don't know based on the provided sources." when the sources do not contain the answer. The answer request is sent through the gateway with task "rag", so it is routed to the standard tier by default and gets failover, caching and cost tracking.

If no chunk passes the relevance threshold, the pipeline answers "I don't know" without calling an LLM at all. This saves cost and prevents ungrounded answers.

## Evaluation

scripts/eval_rag.py measures retrieval quality on a labeled question set: hit@k, recall@k and mean reciprocal rank (MRR). With the --answers flag it also generates answers and reports citation precision, keyword accuracy and abstention accuracy on questions the corpus cannot answer.
