# Response Caching

NexusAI caches responses at two levels to avoid paying for the same answer twice.

## Exact cache

The exact cache key is a SHA-256 hash of the normalized request: the messages with whitespace collapsed, the model, the tier, the task, the temperature and max_tokens. Entries live in Redis and expire after NEXUS_EXACT_CACHE_TTL_S seconds, which defaults to 3600 (one hour).

## Semantic cache

The semantic cache catches rephrasings of a question that was already answered. The gateway embeds the last user message and searches previous prompts in pgvector. If the closest previous prompt has a cosine similarity of at least NEXUS_SEMANTIC_THRESHOLD, the cached response is returned. The default threshold is 0.95 with OpenAI embeddings and 0.97 with the offline hashing embedder. Semantic cache entries expire after NEXUS_SEMANTIC_CACHE_TTL_S seconds, 86400 by default (one day).

A semantic hit also writes the response into the exact cache, so the next identical request is served from Redis without computing an embedding.

## Cache namespaces

A cached answer must never be reused when the context differs. The semantic cache therefore scopes every entry to a namespace: a hash of the system prompt, the earlier conversation turns, the pinned model or tier, and the task. Two identical questions asked with different system prompts never share a cache entry.

## What is not cached

Requests with a temperature above NEXUS_CACHE_MAX_TEMPERATURE (0.3 by default) are never cached, because the caller asked for varied output. Callers can also send "cache": false to skip caching for a single request.

Prompts longer than NEXUS_SEMANTIC_MAX_CHARS (2,000 characters by default) skip the semantic cache. Long prompts are dominated by shared context such as pasted logs or documents, so two different questions with the same context would embed almost identically.

RAG requests skip the semantic cache and use only the exact cache. A RAG prompt contains the retrieved passages, so two different questions that retrieve the same passages would look almost identical to the embedding model and could wrongly share an answer.

## Choosing a threshold

A higher threshold returns fewer but safer hits. Lowering the threshold raises the hit rate but risks answering a different question with a cached response, for example the same question asked for a beginner and for a senior engineer. Run scripts/benchmark.py with --thresholds to compare hit rate and wrong hits at several thresholds, and check a sample of semantic hits from real traffic by hand.
