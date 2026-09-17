# Week 1 — TabPFN v3 Teacher Extraction

This directory is the only source tree for Week 1. It implements a reproducible
TabPFN v3 teacher-extraction prototype and keeps deployment artifacts separate
from source, checkpoints, caches, and credentials.

The runtime target is the Windows OpenSSH host `<SSH_USER>@<SERVER_HOST>` on port
`<SSH_PORT>`. Native server paths are rooted at `<REMOTE_ROOT>`; SCP uses the
equivalent `<REMOTE_ROOT_POSIX>/...` path only at transport time.

## Source snapshot workflow

Run commands from this directory in PowerShell. The transfer scripts are safe
by default: without `-Execute` they only print their intended actions.

```powershell
# 1. Regenerate this after every source change.
.\scripts\New-SourceManifest.ps1
.\scripts\Test-SourceManifest.ps1

# 2. Review the immutable snapshot target, then explicitly upload it.
.\scripts\Sync-SourceSnapshot.ps1
.\scripts\Sync-SourceSnapshot.ps1 -Execute
```

`source_sha256` is a deterministic hash of normalized relative paths, file
hashes, and byte counts. `git_commit` is always `null`: this project purposely
does not initialize Git. The manifest excludes generated evidence, virtual
environments, caches/checkpoints, and credential material; more importantly, it
has an explicit source allowlist (`src`, `tests`, `scripts`, `configs`, and the
reviewed root/provenance/report files). A likely inline credential fails the
manifest as well. The server expands a staging archive, reruns the manifest
validator, marks its files read-only, and only then moves it into
`snapshots/<source_sha256>`. The runner verifies the exact snapshot file tree
and SHA before and after execution. An existing snapshot is never overwritten.

The scripts require key-based OpenSSH authentication by default
(`BatchMode=yes`, `StrictHostKeyChecking=yes`). Accept the host key out of band
before an automated run; do not weaken host-key checking or put a password/token
in a script, YAML file, manifest, or report.

## Server environment evidence

`Prepare-ServerTeacher.ps1` captures the existing `base` environment, dry-runs
binary-only dependency resolution, checks CUDA/imports and new `pip check`
failures, then captures the post-state. The current server has no PyTorch, so a
CUDA wheel install is deliberately an additional explicit switch rather than an
implicit CPU-Torch dependency resolution. The script never selects a different
environment. Its pip-map rollback is fail-closed: a restoration failure stops
the workflow and must be repaired before any Teacher run.

```powershell
.\scripts\Prepare-ServerTeacher.ps1
# Existing compatible CUDA PyTorch:
.\scripts\Prepare-ServerTeacher.ps1 -AllowBaseMutation -Execute
# Current server: explicitly allow the pinned 2.6.0+cu126 PyTorch installation:
.\scripts\Prepare-ServerTeacher.ps1 -InstallCudaTorch -AllowBaseMutation -Execute
```

The capture includes a credential-redacted full `pip freeze`, `pip check`,
Python, PyTorch/CUDA, and `nvidia-smi` output. Existing evidence is never
overwritten without an explicit `-Overwrite` flag.

## Evidence return workflow

Run validation only after a source snapshot has been created. The runner checks
the snapshot SHA locally, executes from that exact snapshot, and writes only
returnable evidence under `<REMOTE_ROOT>\runs\<run_id>\evidence\`.
Download it into `reports/evidence/<run_id>/` as follows:

```powershell
.\scripts\Receive-Evidence.ps1 -RunId example-run
.\scripts\Invoke-TeacherValidation.ps1 -Execute
.\scripts\Receive-Evidence.ps1 -RunId example-run -ExpectedSourceSha <source_sha256> -Execute
```

Required run matrix after the environment gate passes:

- `synthetic`, `n_estimators=1`: all correctness checks.
- `breast-cancer`, `n_estimators=1`: all correctness checks.
- `synthetic`, `n_estimators=8`, `-SkipRelationalChecks`: ensemble provenance
  and raw-score smoke run; the strict relational checks are already covered by
  the one-estimator runs.

The return script refuses existing local run directories, accepts only a strict
evidence allowlist, and requires NPZ-level validation for a claimed successful
run. It preserves each returned per-run report rather than overwriting the
Week-1 summary. Generate that summary only after all three required runs are
received:

```powershell
.\scripts\New-Week01Summary.ps1 `
  -SyntheticSingleRunId <synthetic-e1-run-id> `
  -BreastCancerSingleRunId <breast-e1-run-id> `
  -SyntheticEnsembleRunId <synthetic-e8-run-id> `
  -ExpectedSourceSha <source_sha256>
```

Its remote archive is retained for audit; neither transfer script deletes a
remote snapshot.

## Runtime configuration and completion gate

- `configs/teacher_v3.yaml` fixes the v3 Teacher settings, fixtures, tensor
  layout, and validation tolerances.
- `configs/server_windows.example.yaml` records only non-secret connection and storage
  conventions.
- `reports/week01_teacher_validation.template.md` is the required final report
  shape. Do not mark Week 1 complete if raw-score reconstruction or any ID
  alignment/parity test fails.

The server must have its PriorLabs authorization configured securely (for
example, `TABPFN_TOKEN` in its own environment) before a checkpoint download.
No credential is part of this source tree or a source snapshot.
