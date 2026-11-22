# ADR 0008: Read Excel with the calamine engine; keep OpenPyXL for writing

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D1

## Context

The MVP must read `.xlsx` and, if practical, `.xls`. OpenPyXL reads only `.xlsx` and is slow on large workbooks. pandas supports `engine="calamine"` (python-calamine), which reads `.xlsx`, `.xls`, `.xlsb` and `.ods`.

## Decision

Read all Excel files with `pandas.read_excel(..., engine="calamine")`. Before parsing, check the zip's total uncompressed size to block zip bombs. Keep OpenPyXL as a dependency for Excel export in Phase 2. Install defusedxml so any OpenPyXL XML parsing is protected.

## Alternatives considered

- OpenPyXL only, dropping `.xls`: simpler, but excludes older workbooks that are common in environmental and business data.
- OpenPyXL plus xlrd for `.xls`: two readers with different behaviour.

## Consequences

- One reader for all Excel formats, and faster reads.
- python-calamine is a compiled dependency; wheel availability on Railway is confirmed in spike T0.4 (d).
