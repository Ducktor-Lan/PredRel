# Week 5 - Directionality (Thunder Zone 2)

This independent source tree implements the Week 5 gate from the research
plan: determine whether the TabPFN decoder readout relation is fundamentally
directed, and whether a static symmetric geometry can express it.

Week 4 ended with a frozen STOP_READOUT2REP_ROUTE (readout_captured_by_supcon:
6/6 faithful, 1/6 beyond-SupCon at the 0.02 margin) plus a documented
investigator override to continue (18/18 seed-level increments positive,
sign-test p ~ 7.6e-06). The override conditions (Week 4 conclusion Section 6)
are binding on this tree: pre-registered weak/strong asymmetry boundaries, a
2-3 dataset extension arm frozen before any Week 5 readout, a falsification
clause, and at least one non-removal probe. All four are frozen in
provenance/dataset_manifest.json before any Teacher fit.

## Frozen scope

The benchmark declares nine datasets before a full run: the six frozen real
datasets from Weeks 3-4 (identical IDs, splits, Teacher settings, and content
pins) plus three pre-registered extension datasets (high-dimensional,
multi-cluster, and imbalanced arms). No dataset may be added, removed, or
replaced after inspecting Week 5 readout results.

The directionality protocol per (dataset, seed):

- Take a deterministic stratified probe pool P of at most M support rows
  (M = 48 full, M = 16 smoke; nested; label round-robin over frozen support
  order).
- For each pool row i, hold it out as a single query and keep P minus i as
  support. One frozen Teacher fit per held-out row yields row i of the
  directed matrix R (rows sum to 1 by the alpha simplex).
- Query labels are never supplied to the Teacher. Labels are released only
  after all R rows are extracted, for stratified analysis.
- Train the matched Week-4 SupCon encoder on support rows only and build the
  symmetric cosine matrix S on the same pool P as the symmetric reference.
- Non-removal probe: Top-K retrieval agreement between R and S (Jaccard,
  recall both directions, flattened Spearman) at K = 1/5/10.
- Extension-arm recheck: the exact Week-4 removal protocol at K = 3
  (top_beta / random_same_class / bottom_beta / supcon_top, five random
  repeats, at most 32 queries) on the three extension datasets only, with
  the same 0.02 TV margin. This feeds the falsification clause directly.

The full runner refuses an incomplete manifest, an altered dataset lock, a
missing cache entry, a changed source snapshot, content that differs from
the frozen pins, or an existing run directory. It never removes source
snapshots, cached data, or previous evidence.

## Teacher and Week 4 provenance with leakage boundary

Week 5 imports the immutable Week 1 Teacher snapshot with SHA-256
b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4 and records
the Week 4 dependency (source 3612a8d9..., full evidence
week04-full-3612a8d9e3b8-19198ac8, frozen gate stop_readout2rep_route) in
provenance/week04_dependency.json.

The Teacher bridge accepts support features, support labels, support IDs,
query features, and query IDs. It has no query-label argument. Query labels
are released only after the directed-row extractions, when metrics are
computed. Every Teacher fit receives support features, support labels, and
query features alone.

## Pre-registered decision rule (frozen before any Week 5 readout)

Per dataset (median over seeds 17/29/43):

- STRONG when median A_F >= 0.45, or (median reciprocity@5 <= 0.35 and
  median rank-reversal >= 0.55).
- WEAK when median A_F <= 0.30 and median reciprocity@5 >= 0.60 and median
  rank-reversal <= 0.45.
- Otherwise MODERATE.

where A_F = ||R - R^T||_F / ||R||_F, reciprocity@5 is the mean outgoing vs
incoming Top-5 overlap, and rank-reversal is the mean pairwise order
disagreement between each row and its column.

Global gate over all nine frozen datasets:

- STRONG on strictly more than half (>= 5/9) -> CONTINUE_TO_WEEK06_CONTEXT
  with the dual-space (Query/Key) representation as the main-route
  candidate (structural beyond-symmetric evidence).
- WEAK on strictly more than half AND 0/3 extension datasets beyond-SupCon
  at the 0.02 TV margin -> STOP_READOUT2REP_ROUTE with reason
  weak_directionality_with_no_extension_gain (falsification confirmed;
  execute the pivot to TabPFN Decoder Retrieval Anatomy).
- Otherwise -> CONTINUE_TO_WEEK06_CONTEXT (moderate/mixed evidence; keep
  both symmetric and directed candidates).

Metrics, thresholds, and the falsification clause live in
provenance/dataset_manifest.json. Preserve either outcome.

## Server target

The frozen runtime target is the existing Windows OpenSSH host:

    <SSH_USER>@<SERVER_HOST>:<SSH_PORT>
    <REMOTE_ROOT>\Week 5
    conda environment: base

The known compatible server stack is Python 3.13.13, CUDA-enabled
torch 2.6.0+cu126, tabpfn 8.5.0, tabpfn-extensions 0.6.2,
numpy 2.5.3, and scikit-learn 1.9.1. The server keeps TABPFN_TOKEN outside
this project. Do not put tokens, passwords, private keys, checkpoints, or
datasets in a source archive or evidence return.

The shared model cache is:

    <REMOTE_ROOT>\cache\tabpfn-v3

The Week 5 dataset cache is seeded (server-side copy, never re-download the
six frozen entries) from the verified Week 4 cache, then explicitly locked;
the three extension entries are bootstrapped once with an explicit download
flag before any Teacher run and pinned thereafter:

    seed:   <REMOTE_ROOT>\cache\week4-real-datasets
    target: <REMOTE_ROOT>\cache\week5-real-datasets

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

    # 1. Build and upload a content-addressed immutable Week 5 snapshot.
    .\scripts\Sync-Week05Source.ps1
    .\scripts\Sync-Week05Source.ps1 -Execute

    # 2. Verify the server base environment, the immutable Week 1 dependency,
    #    and seed plus lock the Week 5 dataset cache from the Week 4 cache.
    .\scripts\Prepare-ServerWeek05.ps1 -Execute

    # 3. One-time extension bootstrap (downloads ONLY the three new entries,
    #    then locks; must precede any Teacher run).
    .\scripts\Prepare-ServerWeek05.ps1 -BootstrapExtensionCache -Execute

    # 4. Exercise the complete directed-row path on breast cancer.
    .\scripts\Invoke-Week05Directionality.ps1 -Mode smoke -Execute
    .\scripts\Receive-Week05Evidence.ps1 -RunId <smoke-run-id> -Execute

    # 5. Run all nine frozen datasets and return only the resulting evidence.
    .\scripts\Invoke-Week05Directionality.ps1 -Mode full -Execute
    .\scripts\Receive-Week05Evidence.ps1 -RunId <full-run-id> -RequireFull -Execute

The full run is an experiment, not a predeclared positive result. Its
technical gate checks provenance and coverage; its scientific gate emits
CONTINUE_TO_WEEK06_CONTEXT* or STOP_READOUT2REP_ROUTE from the frozen
thresholds in the dataset manifest. Preserve either outcome.

The evidence receiver verifies the current local immutable source manifest
before transfer and automatically binds the returned runner.json to that
exact source SHA-256. Supply -ExpectedSourceSha only to assert the same
snapshot explicitly; a different value is rejected.
