# Week 2 — Synthetic Anatomy

This directory is the independent Week 2 source tree.  It tests whether the
analysis methods behave as expected before any real-data anatomy claim is made.
The four required synthetic tasks are:

1. `class-only` — same-class support rows are feature-identical and therefore
   exchangeable; class-residual relation `beta` should be uniform within class.
2. `prototype` — each class has two separable prototypes; a query should favor
   its matching same-class prototype.
3. `boundary` — labels are separated by a known boundary; the report records
   how decoder relation mass varies with distance to that boundary.
4. `context-competition` — query and anchors remain fixed while the remaining
   support context changes; alpha and pre-softmax score stability are reported
   separately.

## Frozen server target

Week 2 runs on the same Windows OpenSSH host as Week 1:

```text
<SSH_USER>@<SERVER_HOST>:<SSH_PORT>
<REMOTE_ROOT>\Week 2
conda environment: base
```

The server was re-probed on 2026-09-11.  Its base environment has Python
3.13.13, CUDA-enabled `torch==2.6.0+cu126`, `tabpfn==8.5.0`, and
`tabpfn-extensions==0.6.2`; `pip check` passed.  The server owns the
`TABPFN_TOKEN` value.  Never place it in this project, a source archive, or an
evidence return.

## Teacher dependency

Week 2 intentionally does not copy Teacher extraction code.  It imports the
exact, immutable Week 1 source snapshot named in
`provenance/week01_teacher_dependency.json`.  Before touching a model, the
runner verifies that snapshot's manifest hash.  This makes a Week 2 result
traceable to the Teacher whose alpha/raw-score parity was already validated.

## Local structural checks

The synthetic generators and analysis metrics have no TabPFN dependency.  From
this directory, run:

```powershell
python -m unittest discover -s tests -v
python scripts\source_manifest.py write --root .
python scripts\source_manifest.py verify --root .
```

## Remote workflow

All transfer commands use key-based OpenSSH with strict host-key checking and
are dry runs until `-Execute` is supplied.

```powershell
# 1. Build a content-addressed Week 2 snapshot and upload it.
.\scripts\Sync-Week02Source.ps1
.\scripts\Sync-Week02Source.ps1 -Execute

# 2. Verify the actual server environment and frozen Week 1 dependency.
.\scripts\Prepare-ServerWeek02.ps1 -Execute

# 3. Run one minimal real-Teacher smoke extraction.
.\scripts\Invoke-Week02SyntheticAnatomy.ps1 -Mode smoke -Execute

# 4. After the smoke report is PASS, run every required synthetic task.
.\scripts\Invoke-Week02SyntheticAnatomy.ps1 -Mode full -RunId week02-full-<source_sha_prefix> -Execute

# 5. Return only evidence for a completed run.
.\scripts\Receive-Week02Evidence.ps1 -RunId <run_id> -Execute
```

`full` is intentionally an experiment, not a claim of success.  It writes
metrics and a gate-oriented report; the execution agent must preserve outcomes
that contradict the expected behavior and fix the experimental construction
before moving to Week 3.

