# Week 3 — Beyond Labels on Frozen Real Data

This independent source tree implements the Week 3 gate from the research
plan: determine whether TabPFN decoder readout contains instance-level
information beyond class labels on six frozen, real classification datasets.

The tree deliberately does not train a representation model. Its only purpose
is to make the Go/Pivot/Stop decision for the later Readout2Rep route
auditable.

## Frozen scope

The benchmark is declared in provenance/dataset_manifest.json before a full
run. It fixes:

- six datasets: three scikit-learn built-ins and three version-pinned OpenML
  datasets;
- exact source identifiers, expected shape, class count, and OpenML source
  checksum;
- three stratified, disjoint support/query splits per dataset;
- the class-only null, entropy, top-beta, and residual-profile metrics;
- numerical conditions for the Week 3 STOP recommendation.

The full runner refuses an incomplete manifest, an altered dataset lock, a
missing cache entry, a changed source snapshot, or an existing run directory.
It never removes source snapshots, cached data, or previous evidence.

## Teacher provenance and leakage boundary

Week 3 imports the immutable Week 1 Teacher snapshot with SHA-256
b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4. This is
the Teacher validated in Week 1 and reused by Week 2; it is not the mutable
local Week 1 directory.

The Teacher bridge accepts support features, support labels, support IDs,
query features, and query IDs. It has no query-label argument. Query labels
are released only after alpha extraction, when class-residual metrics are
computed.

## Server target

The frozen runtime target is the existing Windows OpenSSH host:

    <SSH_USER>@<SERVER_HOST>:<SSH_PORT>
    <REMOTE_ROOT>\Week 3
    conda environment: base

The known compatible server stack is Python 3.13.13, CUDA-enabled
torch 2.6.0+cu126, tabpfn 8.5.0, tabpfn-extensions 0.6.2,
numpy 2.5.3, and scikit-learn 1.9.1. The server keeps TABPFN_TOKEN outside
this project. Do not put tokens, passwords, private keys, checkpoints, or
datasets in a source archive or evidence return.

The shared model cache is:

    <REMOTE_ROOT>\cache\tabpfn-v3

The Week 3 dataset cache is intentionally separate:

    <REMOTE_ROOT>\cache\week3-real-datasets

## Local structural checks

Use the bundled Python or any Python with NumPy to validate source layout and
the pure numerical tests. The complete dependency and Teacher checks run on
the server.

    python -m unittest discover -s tests -v
    python scripts\source_manifest.py write --root .
    python scripts\source_manifest.py verify --root .

## Remote workflow

All transfer commands are dry runs until -Execute is supplied. They use
key-based OpenSSH with BatchMode=yes and StrictHostKeyChecking=yes. The
server lacks SFTP, so the source/evidence scripts deliberately use legacy
scp -O through a no-space transfer directory.

    # 1. Build and upload a content-addressed immutable Week 3 snapshot.
    .\scripts\Sync-Week03Source.ps1
    .\scripts\Sync-Week03Source.ps1 -Execute

    # 2. Verify the server base environment and immutable Week 1 dependency.
    .\scripts\Prepare-ServerWeek03.ps1 -Execute

    # 3. Exercise the complete Teacher-to-evidence path on breast cancer.
    .\scripts\Invoke-Week03BeyondLabels.ps1 -Mode smoke -Execute
    .\scripts\Receive-Week03Evidence.ps1 -RunId <smoke-run-id> -Execute

    # 4. Explicitly fetch and lock the three pinned OpenML datasets.
    .\scripts\Prepare-ServerWeek03.ps1 -BootstrapDatasetCache -Execute

    # 5. Run all six frozen datasets and return only the resulting evidence.
    .\scripts\Invoke-Week03BeyondLabels.ps1 -Mode full -Execute
    .\scripts\Receive-Week03Evidence.ps1 -RunId <full-run-id> -RequireFull -Execute

The full run is an experiment, not a predeclared positive result. Its
technical gate checks provenance and coverage; its scientific gate emits
CONTINUE_TO_WEEK04 or STOP_READOUT2REP_ROUTE from the frozen threshold in the
dataset manifest. Preserve either outcome.

The evidence receiver verifies the current local immutable source manifest
before transfer and automatically binds the returned `runner.json` to that
exact source SHA-256. Supply `-ExpectedSourceSha` only to assert the same
snapshot explicitly; a different value is rejected.
