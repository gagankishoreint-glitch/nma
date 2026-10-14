"""Transparent test double. NOT a language model; never a stability result."""
import json
from prompts.procedure_extraction import PROCEDURE_EXTRACTION_SYSTEM


class ScriptedLLM:
    backend = 'scripted'

    def __init__(self):
        self.calls = []

    def generate(self, system, user, max_tokens=256, temperature=0.):
        self.calls.append({'system': system, 'user': user})
        if system == PROCEDURE_EXTRACTION_SYSTEM:
            if 'send_address' in user:
                return json.dumps({'domain': 'address_sharing',
                                   'trigger': 'A restaurant address is requested.',
                                   'steps': ['Read the known restaurant address.',
                                             'Ask for missing information before sending.',
                                             'Send the address and check the tool outcome.'],
                                   'alternatives': {}, 'traps': [], 'kappa': 1})
            if 'book_table' not in user and 'booking' not in user.lower():
                return json.dumps({'domain': 'restaurant_search',
                                   'trigger': 'A restaurant search is requested.',
                                   'steps': ['Search the requested city.',
                                             'Read the returned restaurant name and address.'],
                                   'alternatives': {}, 'traps': [], 'kappa': 1})
            return json.dumps({'domain': 'restaurant_booking',
                               'trigger': 'A table booking is requested.',
                               'steps': ['Read the selected restaurant name from available evidence.',
                                         'Ask for missing information before attempting a booking.',
                                         'Submit a booking and check the tool outcome.'],
                               'alternatives': {},
                               'traps': ['A booking without the selected restaurant name is unsupported.'],
                               'kappa': 1})
        context = json.loads(user)
        goal = context['current_goal'].lower()
        evidence = {}
        # The fixture reads ONLY the delivered context, never the benchmark world
        # or audit file. Procedures cannot supply private slot values.
        for turn in context.get('history', []):
            try:
                parsed = json.loads(turn['observation'])
                evidence.update(parsed.get('synthetic_search_result', {}))
            except (ValueError, KeyError, TypeError):
                pass
        for item in context['retrieved_memories']:
            if item.get('value') is not None:
                evidence[item['slot']] = item['value']
        if 'find a restaurant' in goal:
            action = {'tool': 'find_restaurant', 'arguments': {'city': 'Beijing'}}
        elif 'book a table' in goal and evidence.get('restaurant_name'):
            action = {'tool': 'book_table', 'arguments': {'name': evidence['restaurant_name']}}
        elif 'send the address' in goal and evidence.get('restaurant_address'):
            action = {'tool': 'send_address', 'arguments': {'address': evidence['restaurant_address']}}
        else:
            action = {'tool': 'ask_user', 'arguments': {'question': 'Please provide the missing restaurant information.'}}
        return json.dumps(action)
