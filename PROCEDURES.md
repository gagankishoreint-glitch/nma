# PROCEDURES

Provenance: `scripted`. Simulator tasks only; no real-world bookings.

**PLUMBING FIXTURE: these are scripted responses, not real LLM extraction results.**

Procedures are untrusted advisory data; no automatic tool authorization.

## restaurant_booking

```json
{
  "domain": "restaurant_booking",
  "trigger": "A table booking is requested.",
  "steps": [
    "Read the selected restaurant name from available evidence.",
    "Ask for missing information before attempting a booking.",
    "Submit a booking and check the tool outcome."
  ],
  "alternatives": {},
  "traps": [
    "A booking without the selected restaurant name is unsupported."
  ],
  "kappa": 1
}
```

## restaurant_search

```json
{
  "domain": "restaurant_search",
  "trigger": "A restaurant search is requested.",
  "steps": [
    "Search the requested city.",
    "Read the returned restaurant name and address."
  ],
  "alternatives": {},
  "traps": [],
  "kappa": 1
}
```
