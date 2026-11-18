# Test plan

## Test levels

| Level | Location | Checks |
| --- | --- | --- |
| Unit | `tests/unit` | one function or class, with fakes for networks and services |
| Leakage | `tests/leakage` | no feature or graph edge uses data from after its snapshot |
| Integration | `tests/integration` | all stages in DAG order on the offline fixture, then the API |
| End-to-end | `tests/e2e` | the running stack: every `/health`, both serving paths, a project scan |

Unit, leakage and integration tests run in the `test` stage of `dvc repro`. End-to-end tests run with `pytest -m e2e`.

## Acceptance criteria

| ID | Criterion |
| --- | --- |
| AC1 | all unit tests pass |
| AC2 | all leakage tests pass |
| AC3 | the DVC DAG runs end to end on the fixture |
| AC4 | every service is healthy and both serving paths answer |
| AC5 | test ROC-AUC >= 0.80 |
| AC6 | test PR-AUC above the "no release in a year" heuristic |

## Test cases

| ID | File :: test | Expected |
| --- | --- | --- |
| TC01 | `test_ingest_infrastructure` :: offline cache never calls the network | `CacheMissError` |
| TC02 | `test_ingest_infrastructure` :: token bucket spaces requests | waits 0, 0.5, 1.0 s at 2/s |
| TC03 | `test_ingest_sources` :: repository URLs are normalised | non-code URLs rejected |
| TC04 | `test_ingest_sources` :: issue fetching stops at the history start | one request |
| TC05 | `test_ingest_sources` :: API error objects are rejected | `ValueError` |
| TC06 | `test_ingest_tables` :: shared repositories are flagged | `monorepo` true |
| TC07 | `test_validate` :: bad dates and unknown packages rejected | reason per row |
| TC08 | `test_graph_features_labels` :: snapshot edges use the release current then | later dependencies ignored |
| TC09 | `test_graph_features_labels` :: bus factor | 2 for a 4/3/3 split |
| TC10 | `test_graph_features_labels` :: missing commit data | NaN, not zero |
| TC11 | `test_graph_features_labels` :: label rules | label per definition |
| TC12 | `test_models` :: SHAP values sum to the log-odds | tolerance 1e-6 |
| TC13 | `test_models` :: calibration with one class | skipped, recorded |
| TC14 | `test_models` :: promotion gate | promote or reject as specified |
| TC15 | `test_operations` :: rollback | previous champion restored |
| TC16 | `test_operations` :: drift of a 3-std shift | 3.0 |
| TC17 | `test_operations` :: project risk | weakest link |
| TC18 | `test_operations` :: model server waits for a model | retries, then loads |
| TC19 | `test_operations` :: CI stages stop the pipeline on failure | `CheckFailedError` |
| TC20 | `test_operations` :: stage runs are logged | run log with traceback |
| TC21 | `test_api` :: batch and live risk, 404, 502 | status codes |
| TC22 | `test_api` :: project scan propagates risk | weakest-link path |
| TC23 | `test_api` :: pipeline DAG and runs | console data |
| TC24 | `test_api` :: metrics endpoint | drift and route metrics |
| TC25 | `test_frontend` :: name validation, friendly errors | invalid names refused |
| TC26 | `test_no_future_data` :: future events never change features | identical frames |
| TC27 | `test_no_future_data` :: later dependencies never change graphs | identical edges |
| TC28 | `test_no_future_data` :: coverage flags are not model inputs | not in feature list |
| TC29 | `test_pipeline_on_fixture` :: all stages succeed, API serves output | all succeed |
| TC30 | `test_smoke` :: stack healthy, live predictions on both paths | all healthy |

## Test report

Run on 2026-11-18. Unit, leakage and integration counts are from `reports/tests.json`; the end-to-end count is from `pytest -m e2e` against the Docker Compose stack.

| Suite | Passed | Failed |
| --- | --- | --- |
| unit | 140 | 0 |
| leakage | 5 | 0 |
| integration | 8 | 0 |
| e2e | 11 | 0 |
| total | 164 | 0 |

| Criterion | Result | Status |
| --- | --- | --- |
| AC1 | 140 / 140 | PASS |
| AC2 | 5 / 5 | PASS |
| AC3 | 8 / 8 | PASS |
| AC4 | 11 / 11 | PASS |
| AC5 | ROC-AUC 0.839 (pilot data) | PASS |
| AC6 | PR-AUC 0.128 vs 0.065 (pilot data) | PASS |
