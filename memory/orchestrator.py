"""Goal/slot-directed neural reads. Audit text is NEVER searched for recall.

A finite slot schema is public task structure, not a hidden value dictionary.
Arbitrary values only persist in neural parameters. No vector index/codebook.
"""
import torch
from .titans_memory import NeuralLongTermMemory

SLOT_QUERIES = {
    'restaurant_name': 'What restaurant did we select?',
    'restaurant_address': 'What is the address of the selected restaurant?',
    'booking_status': 'What is the restaurant table booking status?',
}


class MemoryOrchestrator:
    def __init__(self, encoder, memory=None, audit_sink=None, update_enabled=True):
        self.encoder = encoder
        self.memory = memory or NeuralLongTermMemory(input_dim=encoder.dim)
        self.audit_sink = audit_sink
        self.update_enabled = update_enabled

    def observe_slots(self, values):
        if not values or any(slot not in SLOT_QUERIES or not isinstance(value, str)
                             for slot, value in values.items()):
            raise ValueError('Unrecognized slot or value')
        query = torch.stack([self.encoder.encode(SLOT_QUERIES[slot]) for slot in values])
        targets = list(values.values())
        surprise = self.memory.compute_surprise(query, targets)
        if self.update_enabled:
            # Batch all slots in ONE step for this observed tool outcome.
            surprise = self.memory.test_time_update(query, targets)
        if self.audit_sink is not None:
            self.audit_sink({'slots': values, 'surprise': surprise})
        return surprise

    def observe_slot(self, slot, value):
        return self.observe_slots({slot: value})

    def observe(self, text, kind='observation'):
        # Unstructured observations train reconstruction only, never falsely
        # promise a textual memory can be recovered from embedding autoencoding.
        x = self.encoder.encode(text)
        surprise = self.memory.compute_surprise(x)
        if self.update_enabled:
            surprise = self.memory.test_time_update(x)
        if self.audit_sink is not None:
            self.audit_sink({'text': text, 'kind': kind, 'surprise': surprise})
        return surprise

    def recall(self, slots):
        results = []
        for slot in slots:
            if slot not in SLOT_QUERIES:
                raise ValueError('Unknown query slot')
            decoded = self.memory.recall_text(self.encoder.encode(SLOT_QUERIES[slot]))
            results.append({'slot': slot, 'value': decoded.text,
                            'confidence': decoded.confidence, 'reason': decoded.reason,
                            'source': 'neural_parameters'})
        return results
