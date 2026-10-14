"""Constrained synthetic world. Only the executor can observe hidden truth.

No network, booking, or messaging side effects. A generated [DONE] never decides
success. Each method gets an independent world with the same seeded values.
"""
from dataclasses import dataclass, field
import json
import random

GOALS = [
    'Find a restaurant in Beijing.',
    'Book a table at the restaurant you found.',
    'Send the address to my colleague.',
]
INTENTS = ['restaurant_search', 'restaurant_booking', 'address_sharing']
MISSING_SLOTS = [[], ['restaurant_name'], ['restaurant_address']]
ACTION_ARGS = {'find_restaurant': {'city'}, 'book_table': {'name'},
               'send_address': {'address'}, 'ask_user': {'question'}}


def parse_action(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {'tool', 'arguments'}:
        raise ValueError('Expected tool and arguments only')
    tool, args = value['tool'], value['arguments']
    if not isinstance(tool, str) or tool not in ACTION_ARGS or not isinstance(args, dict) or set(args) != ACTION_ARGS[tool]:
        raise ValueError('Tool not allowlisted or argument schema mismatch')
    if not all(isinstance(v, str) and 0 < len(v) <= 200 and not any(ord(c)<32 for c in v) for v in args.values()):
        raise ValueError('Invalid tool argument')
    return value


@dataclass(frozen=True)
class Outcome:
    observation: str
    succeeded: bool = False
    terminal: bool = False
    slots: dict = field(default_factory=dict)


class RestaurantWorld:
    def __init__(self, seed=0):
        rng = random.Random(seed)
        self.name = f'Synthetic Lantern {rng.randrange(1000,9999)}'
        self.address = f'{rng.randrange(100,999)} Example Lane, Beijing'

    def execute(self, session, action):
        tool, args = action['tool'], action['arguments']
        if tool == 'ask_user':
            return Outcome('The user has no additional information in this benchmark.', terminal=True)
        if session == 0 and tool == 'find_restaurant' and args['city'].casefold() == 'beijing':
            values = {'restaurant_name': self.name, 'restaurant_address': self.address}
            return Outcome(json.dumps({'synthetic_search_result': values}), True, True, values)
        if session == 1 and tool == 'book_table' and args['name'] == self.name:
            return Outcome('Synthetic booking confirmed.', True, True, {'booking_status': 'confirmed'})
        if session == 2 and tool == 'send_address' and args['address'] == self.address:
            return Outcome('Synthetic address delivery confirmed.', True, True)
        # Do not leak missing ground-truth values through failure feedback.
        return Outcome('Tool rejected: wrong tool or incorrect/missing arguments.')
