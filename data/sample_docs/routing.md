# Cost-Aware Routing

The router sends each request to the cheapest model that is good enough for it. Models are grouped into three tiers: economy, standard and premium.

## Model tiers

The economy tier holds gpt-4o-mini, gemini-3.1-flash-lite, gemini-3.5-flash-lite and claude-haiku-4-5. The standard tier holds gemini-3.6-flash, gpt-4.1 and claude-sonnet-4-5. The premium tier holds gemini-3.8-flash, gpt-5 and claude-opus-4-1. The router only uses models whose provider is configured, so a Gemini-only deployment routes entirely to Gemini models. If no configured model exists in the chosen tier or the one above it, the router falls back to the nearest tier that has one. Within a tier, models are sorted by a blended price that weights output tokens three times as heavily as input tokens, since answers are usually more expensive than prompts.

## Routing rules

Rules are evaluated in priority order, lowest number first, and the first rule that matches decides the tier. A rule can match on the task name, on keywords in the last user message, or on the estimated prompt size with min_prompt_tokens and max_prompt_tokens.

The default rules are:

- rag-answers (priority 10): requests with task "rag" go to the standard tier.
- complex-reasoning (priority 20): prompts containing phrases such as "step by step", "prove", "debug", "architecture", "trade-off" or "refactor" go to the premium tier.
- long-context (priority 30): prompts estimated at 2,000 tokens or more go to the standard tier.
- default-economy (priority 100): everything else goes to the economy tier.

Every rule set must include a catch-all rule with an empty match, so that every request is routed. The admin API rejects rule sets without one.

## Escalation

The candidate list contains the chosen tier's models, cheapest first, followed by the models of the next tier up. If every model in the economy tier fails, the request escalates to the standard tier instead of failing. If the chosen tier and the tier above it both fail, the gateway degrades to lower tiers as a last resort, because a cheaper model's answer is better than an error.

## Overriding the router

A caller can bypass the rules by setting "model" to a specific model name or "tier" to a tier name in the request body. Setting model to "auto", the default, lets the router decide.

## Editing rules

Rules can be edited at runtime from the admin UI at /admin or with PUT /admin/api/rules. Saved rules are stored in the routing_rules table and survive restarts.
