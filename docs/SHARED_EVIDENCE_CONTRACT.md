# Shared Evidence Contract — MetaField (Phase 3)

**Status:** Implemented — semantic mapping of the CANtheon / CANgate evidence envelope into `FieldObservation`.

## Ownership

| Layer | Owns |
|-------|------|
| CANtheon | transport, dual timestamps, sequence, quality, provenance, boundary envelope |
| CANgate | adapter boundary (inside CANtheon) |
| **MetaField** | field semantics, `FieldObservation`, memory, geometry, prediction |
| TensorGate | numerical / tensor lineage (next) |

## What changed

- `schemas/evidence.py` — `EvidenceKind`, `EvidenceQuality`, `EvidenceLineage`
- `FieldObservation.evidence` — optional lineage; existing consumers ignore it
- `adapters/cantheon_boundary.py` — pure dict conversion, no CANtheon import
- Body types extended: `can`, `can_fd`, `cantheon`
- `FieldRegion.unit` optional

## Mapping rules

- Boundary `payload.value` → `FieldRegion.observed`
- Boundary `quality` → `EvidenceQuality` (never rewritten to VALID)
- Dual timestamps → `EvidenceLineage.source_timestamp_ns` / `ingest_timestamp_ns`
- Provenance parent → `parent_event_ids`
- RAW vs diagnostic/health → `EvidenceKind.RAW` / `ASSERTION`

## Failure behavior

- Unsupported boundary major version → `BoundaryAdapterError`
- Missing `node_id` on observation → `BoundaryAdapterError`
- Unknown quality string → `EvidenceQuality.UNKNOWN` (not VALID)

## Verification

```bash
PYTHONPATH=. python -m pytest tests/test_evidence_contract.py -q
```

## Next

TensorGate — computational lineage (input identities, transform version, output identity).
