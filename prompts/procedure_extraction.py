"""Isolated extraction with explicit outcome gating and strict validation."""
import copy
import json
import re

PROCEDURE_EXTRACTION_SYSTEM = '''You are a procedure extractor. Given a
completed task log, extract a reusable procedure in strict JSON.

OUTPUT FORMAT:
{
  "domain": "<short domain name, e.g., flight_booking>",
  "trigger": "<condition that makes this procedure applicable>",
  "steps": ["<step 1>", "<step 2>", ...],
  "alternatives": {"<step>": ["<alt 1>", "<alt 2>"]},
  "traps": ["<specific failure mode>", ...],
  "kappa": 1
}

RULES:
1. Only extract if the task succeeded.
2. Steps must be concrete and actionable. Not vague.
3. Traps must be specific failure modes actually observed or strongly implied.
4. Generalize. Do not include user-specific data (names, dates, IDs).
5. Output JSON only. No preamble. No explanation. No markdown fences.
'''
PROCEDURE_EXTRACTION_USER = 'TASK LOG:\n{log}\n\nExtract the procedure.'
DOMAINS = {'restaurant_search', 'restaurant_booking', 'address_sharing'}
FIELDS = {'domain', 'trigger', 'steps', 'alternatives', 'traps', 'kappa'}


def validate_procedure(proc, forbidden_values=()):
    if not isinstance(proc, dict) or set(proc) != FIELDS:
        raise ValueError('Procedure schema mismatch')
    if proc['domain'] not in DOMAINS:
        raise ValueError('Unsupported procedure domain')
    if type(proc['kappa']) is not int or proc['kappa'] != 1:
        raise ValueError('Extractor may not set consolidation count')
    def text(value):
        return isinstance(value, str) and 0 < len(value) <= 200 and not any(ord(c) < 32 for c in value)
    if not text(proc['trigger']):
        raise ValueError('Invalid trigger')
    if not isinstance(proc['steps'], list) or not 1 <= len(proc['steps']) <= 8 or not all(text(x) for x in proc['steps']):
        raise ValueError('Invalid steps')
    if not isinstance(proc['traps'], list) or len(proc['traps']) > 8 or not all(text(x) for x in proc['traps']):
        raise ValueError('Invalid traps')
    alternatives = proc['alternatives']
    if not isinstance(alternatives, dict) or len(alternatives) > 8:
        raise ValueError('Invalid alternatives')
    for key, values in alternatives.items():
        if key not in proc['steps'] or not isinstance(values, list) or len(values) > 4 or not all(text(v) for v in values):
            raise ValueError('Invalid alternative step')
    serialized = json.dumps(proc, ensure_ascii=False).casefold()
    if any(value and str(value).casefold() in serialized for value in forbidden_values):
        raise ValueError('Procedure contains episode-specific data')
    if re.search(r'\b\d{2,}\b|[\w.+-]+@[\w.-]+\.[a-z]+', serialized):
        raise ValueError('Possible personal data in procedure')
    return copy.deepcopy(proc)


def extract_procedure(llm, log_text, *, succeeded, forbidden_values=(), expected_domain=None):
    if not succeeded:
        return {'error': 'task_not_successful'}
    try:
        # Exactly two fresh text messages, never the main-loop conversation.
        user = PROCEDURE_EXTRACTION_USER.format(log=log_text)
        if expected_domain is not None:
            if expected_domain not in DOMAINS:
                raise ValueError('Unknown required domain')
            user += f'\nRequired domain for this simulator: {expected_domain}.'
        raw = llm.generate(system=PROCEDURE_EXTRACTION_SYSTEM,
                           user=user,
                           max_tokens=512, temperature=0.)
        cleaned = raw.strip()
        fenced = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, flags=re.S | re.I)
        if fenced:
            cleaned = fenced.group(1).strip()
        parsed = validate_procedure(json.loads(cleaned), forbidden_values)
        if expected_domain is not None and parsed['domain'] != expected_domain:
            raise ValueError('Extracted domain does not match completed task')
        return parsed
    except (ValueError, TypeError) as error:
        return {'error': 'invalid_procedure', 'detail': str(error)}
