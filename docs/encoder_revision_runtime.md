# Runtime encoder revisions

The released V3 and opt-in V4 heads already recorded immutable CLIP/DINO snapshots.
The runtime now passes those SHAs to both model and processor `from_pretrained`
calls. Previously it recorded the snapshots but loaded each model ID's default Hub
revision. The loader cache keys contain model ID, model revision and processor
revision; a differently pinned or unpinned request cannot reuse the old instance.

`infer.py`, the Gradio app and bundle-backed robustness/error/reference scoring pass
bundle metadata through the shared score boundary. The same context-local revision
scope reaches V4 global/local branches and dual feature extraction without changing
feature mathematics. Nested calls restore their prior scope; concurrent calls do
not overwrite each other's selected snapshots.

An existing snapshot map must contain the needed model IDs and full lowercase
40-hex commit SHAs. Empty maps, symbolic branches, abbreviated SHAs and missing
required encoders fail before scoring. Processor and model use the same SHA.
An unavailable shared encoder raises an error and leaves existing CLI output
untouched, rather than returning a successful set of neutral predictions. A broken
individual image still receives the documented 0.5 fallback row.

## Legacy and future training

A bundle with **no** `encoder_snapshots` retains its old default-revision loading
policy with `LegacyEncoderWarning`. That run is explicitly unpinned, not a
reproduction of a recorded checkpoint. An invalid/partial map is never treated as
legacy. Direct historical helpers that do not pass metadata also warn; they are
not a pinned reproduction route.

The default extraction config pins the SHAs recorded by the released V3/V4 heads.
New generic/V3/V4 caches and generic shard/extra-view archives carry a canonical
`encoder_snapshots` JSON field. New V3/V4 cache fingerprints include that map and
the current scoring/resolver source in a distinct versioned namespace. Resume and
merge do not combine different checkpoint identities. Future head fitting reads
the cached identities instead of guessing from this machine's `refs/main` cache.
Missing or mixed cache snapshots fail before fitting. V4 also requires the cached
snapshots to match its retained V3 classifier.

**Old unpublished caches are not retroactively labelled or certified.** To use
the updated fitting paths, re-extract the required caches with known pinned
checkpoints. A partial generic cache with missing/different identity is discarded
for an explicitly logged fresh extraction; complete/merged caches are rejected.
This change does not reconstruct the uncommitted original data/caches or make the
full historical training procedure reproducible from a clean clone. The archived
corner migration and unversioned diagnostic helpers remain historical tools.

## Separate historical evidence from current source

All existing heads, reports, pretest/evaluation locks and metric bytes are retained.
The original source at commit
`1249bc1ecdbd050a03ec6071dcb7708adf683a4f` is retained byte-for-byte as `.py.txt` audit
data under `docs/frozen-source/`. Those files are never imported or executed.
`index.json` records original and current source hashes separately. The historical
receipt resolver checks the original digest and the current runtime digest; it
rejects corruption of either source instead of changing an old receipt.

```text
python scripts/check_encoder_receipts.py
python -m pytest tests/test_encoder_revisions.py tests/test_frozen_source.py tests/test_v3_evidence.py tests/test_v4_evidence.py tests/test_v3_contract.py tests/test_v4.py tests/test_extra_views.py tests/test_feature_variants.py -q
```

The new validation is offline, using fake encoder factories and synthetic feature
caches. It proves requested revision binding, version-aware caching, failure
behavior and receipt consistency. It includes no checkpoint download, actual
encoder forward, retraining, GPU run or new detector-quality/performance result.
Published V3/V4 metrics continue to describe their frozen historical source and
measurements; these checks do not remeasure or certify the new runtime's outputs.

Offline local outcome: **74 passed**, no skips; **45** receipt checks pass in normal
and optimized Python, including **9** original source copies independently compared
to the recorded Git commit. [Bounded validation receipt](../results/runtime_integrity/encoder_revision_checks.json)
records the checked source hashes and the explicit absence of new measurements.
