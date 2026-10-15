# Troubleshooting

## HTTP 401 invalid or missing API key

The request did not include the key set in NEXUS_API_KEY, or an admin endpoint was called without NEXUS_ADMIN_API_KEY. Send the key in the X-API-Key header.

## HTTP 422 from PUT /admin/api/rules

The rule set failed validation. Common causes are a duplicate rule name, a rule with both tier and model set, an unknown match key, or a rule set without a catch-all rule that has an empty match.

## HTTP 503 No healthy provider available

Every provider's circuit breaker is open, or no provider is configured. Check GET /admin/api/providers for breaker states. Breakers move to half-open after NEXUS_BREAKER_RESET_S seconds and retry automatically.

## HTTP 429 Rate limited by every available model

Every candidate model hit its quota, even after one retry. On Gemini's free tier this usually means the per-minute or per-day limit for those models. Check your project's limits in Google AI Studio, slow the client down, or add another provider key.

## HTTP 502 All providers failed

Every candidate model failed after its retries. The response body lists each attempt with the model, error and status code, which usually shows whether the cause is rate limiting (429) or an outage (5xx).

## Semantic cache returns wrong answers

The similarity threshold is too low for your traffic. Raise NEXUS_SEMANTIC_THRESHOLD toward 0.97, and check that requests with different system prompts are not being sent with identical context.

## RAG says "I don't know" for questions the documents answer

Check GET /v1/rag/documents to confirm the document was ingested. Use POST /v1/rag/search to see which chunks the question retrieves and their scores. If the right chunk scores below RAG_MIN_SCORE, lower the threshold or rewrite the question with the document's own terms.

## Gateway refuses to start: embedding dimension mismatch

The vector columns were created for a different embedding model. Drop the semantic_cache and chunks tables, or the database, and re-ingest documents.
