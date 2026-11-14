# free-model-aggregator-01 — best free AI model aggregator

One Node 22 server (`main.ts`, TypeScript run directly, standard library only) that serves
the OpenAI Chat Completions API (`POST /v1/chat/completions`) and the Anthropic Messages API
(`POST /v1/messages`), each in JSON and in streaming SSE, over the ten simulated free-tier
providers the trial's brief names. `GET /v1/models` answers in the wire its auth header names
(`Authorization: Bearer` → OpenAI list, `x-api-key` → Anthropic list). `node main.ts --list`
prints the catalog document byte-for-byte as it is registered.

## Routing (quality, 60%)

Every request is served by the highest-rated **free** model that meets its requirements, ranked
by the sealed quality table's `rating_milli` (descending), the client's own wire first at an
equal rating, then provider id, then model id. The ranking is recomputed per request from the
state so far, so a provider that just failed or ran dry falls out on its own.

Per-provider call caps keep the entry off every provider's budget cliff. The kit's generator
draws each provider's per-trial budget from a fixed range whose minimum is public
(groq 9, cerebras 5, gemini 6, zai 4, openrouter 10, sambanova 4, pollinations 10,
cloudflare 7, ovh 5, kilol 12); total attempts per provider are capped at that minimum
minus two, which leaves room for the trial's own failure schedule while guaranteeing the
entry never spends a provider's budget itself. When every candidate is capped out, the caps
are ignored rather than leaving a request unserved. A paid model is never routed.

## Failover (10%)

A provider's refusal message tells which kind of refusal it is:

- `"budget is spent"` → the provider is done for the trial, never called again;
- `"is out"` → its outage started, never called again;
- `"refused this call"` → one call on its failure schedule, so the same model gets one
  immediate retry (the next call number is a fresh draw);
- anything else → move on to the next candidate at once.

Only the one successful call answers the request.

## Wires

- OpenAI client → OpenAI-wire providers (passthrough) and Gemini-wire providers
  (`generateContent` / `streamGenerateContent`, translated back to Chat Completions).
- Anthropic client → Anthropic-wire providers (passthrough) and OpenAI-wire providers
  (translated both ways, including the SSE event order).

An OpenAI client is never sent to an Anthropic-wire provider: that wire cannot carry
`response_format`, so a JSON-mode request would echo `json_mode` absent. Every translation
preserves exactly the fields the env echoes back and compares — system text, user text,
tool names, `max_tokens`, stop sequences, the image bytes and the JSON-mode flag — and
usage counts and stop reasons are copied from the provider that answered, never recomputed.

## Determinism

No wall clock, no unseeded randomness: the server never reads `Date.now`, `new Date` or
`Math.random`. Every choice is a function of the brief, the requests in arrival order and
the providers' answers. The catalog is built once from the probes; `--list` prints that
same object. Model-list timestamps are fixed constants.

## Layout

- `main.ts` — the server: brief, catalog probe, register, routes, router, failover, poll/exit.
- `lib/providers.ts` — the brief's three documents, the catalog probe, the catalog builder.
- `lib/wire.ts` — client wires, upstream wires, and the translations between them.

No model calls are made by this entry (none are needed: every decision comes from the brief
and the providers' answers), so it needs no provider key and spends nothing.
