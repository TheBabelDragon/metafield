"""
sonify — MetaField → AXIOM bridge

Decouples lattice QCD observables from MIDI / AXIOM via a plain JSONL contract.
metafield never imports anything musical; AXIOM never imports anything lattice.
"""

from .mapping import map_record_to_notes
from .export import export_jsonl_to_axiom

__all__ = ["map_record_to_notes", "export_jsonl_to_axiom"]
