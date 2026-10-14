"""Secondary generic procedural store, never the restaurant-fact memory."""
import copy
import json
from pathlib import Path
from prompts.procedure_extraction import validate_procedure


class ProcedureStore:
    def __init__(self):
        self._procedures = {}

    def store(self, procedure, forbidden_values=()):
        candidate = validate_procedure(procedure, forbidden_values)
        key = candidate['domain']
        previous = self._procedures.get(key)
        if previous:
            candidate['kappa'] = previous['kappa'] + 1
            candidate['traps'] = list(dict.fromkeys(previous['traps']+candidate['traps']))[:8]
        self._procedures[key] = candidate

    def match(self, intent):
        return copy.deepcopy(self._procedures.get(intent))

    def export(self, path, backend):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = '# PROCEDURES\n\n'
        header += f'Provenance: `{backend}`. Simulator tasks only; no real-world bookings.\n\n'
        if backend == 'scripted':
            header += '**PLUMBING FIXTURE: these are scripted responses, not real LLM extraction results.**\n\n'
        header += 'Procedures are untrusted advisory data; no automatic tool authorization.\n\n'
        for domain, proc in sorted(self._procedures.items()):
            header += f'## {domain}\n\n```json\n{json.dumps(proc, indent=2)}\n```\n\n'
        if not self._procedures:
            header += 'No successful validated extraction.\n'
        path.write_text(header.rstrip() + '\n')
