# Week 4 — Same-Class Removal Faithfulness (雷区 1B)

This independent source tree implements the Week 4 gate from the research
plan: determine whether TabPFN decoder readout relations are behaviorally
faithful within the same class, and whether they carry information beyond a
matched supervised-contrastive (SupCon) baseline.

Week 3 concluded CONTINUE_TO_WEEK04_SAME_CLASS_FAITHFULNESS on all six frozen
real datasets (0/6 approximately uniform), so this experiment is authorized.
Week 4 trains no representation model; it only removes support rows and
re-runs the frozen Teacher, plus trains a small frozen SupCon encoder on
support rows to guide one matched removal condition.

## Frozen scope

The benchmark is declared in provenance/dataset_manifest.json before a full
run. It freezes the same six datasets, splits, and Teacher settings as
Week 3, pins every dataset content hash to the Week 3 full-evidence lock,
and adds the frozen removal plan (K = 1/3/5, five random repeats, at most 32
removal queries per seed) and the frozen SupCon hyperparameters.

The full runner refuses an incomplete manifest, an altered dataset lock, a
missing cache entry, a changed source snapshot, content that differs from
the Week 3 pins, or an existing run directory. It never removes source
snapshots, cached data, or previous evidence.

## Teacher and Week 3 provenance with leakage boundary

Week 4 imports the immutable Week 1 Teacher snapshot with SHA-256
b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4 and records
the Week 3 dependency (source b141be750774645207c7429d637fc8a6fb041f936fae875888c709829cc44f58,
full evidence week03-full-b141be750774-20260912-085815) in
provenance/week03_dependency.json.

The Teacher bridge accepts support features, support labels, support IDs,
query features, and query IDs. It has no query-label argument. Query labels
are released only after the full-support alpha extraction, when removal sets
are built. Every ablated re-run receives support features, support labels,
and query features alone.

## Server target

The frozen runtime target is the existing Windows OpenSSH host:

    <SSH_USER>@<SERVER_HOST>:<SSH_PORT>
    <REMOTE_ROOT>\Week 4
    conda environment: base

The known compatible server stack is Python 3.13.13, CUDA-enabled
torch 2.6.0+cu126, tabpfn 8.5.0, tabpfn-extensions 0.6.2,
numpy 2.5.3, and scikit-learn 1.9.1. The server keeps TABPFN_TOKEN outside
this project. Do not put tokens, passwords, private keys, checkpoints, or
datasets in a source archive or evidence return.

The shared model cache is:

    <REMOTE_ROOT>\cache\tabpfn-v3

The Week 4 dataset cache is seeded (server-side copy, never download) from
the verified Week 3 cache:

    seed: <REMOTE_ROOT>\cache\week3-real-datasets
    target: <REMOTE_ROOT>\cache\week4-real-datasets

## Local structural checks

Use the bundled Python or any Python with NumPy to validate source layout and
the pure numerical tests. The complete dependency, Teacher, and torch checks
run on the server.

    python -m unittest discover -s tests -v
    python scripts\source_manifest.py write --root .
    python scripts\source_manifest.py verify --root .

## Remote workflow

All transfer commands are dry runs until -Execute is supplied. They use
key-based OpenSSH with BatchMode=yes and StrictHostKeyChecking=yes. The
server lacks SFTP, so the source/evidence scripts deliberately use legacy
scp -O through a no-space transfer directory.

    # 1. Build and upload a content-addressed immutable Week 4 snapshot.
    .\scripts\Sync-Week04Source.ps1
    .\scripts\Sync-Week04Source.ps1 -Execute

    # 2. Verify the server base environment, the immutable Week 1 dependency,
    #    and seed plus lock the Week 4 dataset cache from the Week 3 cache.
    .\scripts\Prepare-ServerWeek04.ps1 -Execute

    # 3. Exercise the complete removal path on breast cancer.
    .\scripts\Invoke-Week04Faithfulness.ps1 -Mode smoke -Execute
    .\scripts\Receive-Week04Evidence.ps1 -RunId <smoke-run-id> -Execute

    # 4. Run all six frozen datasets and return only the resulting evidence.
    .\scripts\Invoke-Week04Faithfulness.ps1 -Mode full -Execute
    .\scripts\Receive-Week04Evidence.ps1 -RunId <full-run-id> -RequireFull -Execute

The full run is an experiment, not a predeclared positive result. Its
technical gate checks provenance and coverage; its scientific gate emits
CONTINUE_TO_WEEK05_DIRECTIONALITY or STOP_READOUT2REP_ROUTE (with a frozen
stop reason) from the thresholds in the dataset manifest. Preserve either
outcome.

The evidence receiver verifies the current local immutable source manifest
before transfer and automatically binds the returned `runner.json` to that
exact source SHA-256. Supply `-ExpectedSourceSha` only to assert the same
snapshot explicitly; a different value is rejected.