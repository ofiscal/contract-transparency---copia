# Dependencies: how to upgrade without fear

The project must keep working as pandas, scikit-learn, XGBoost, Streamlit,
etc. release new versions. The design:

1. **`pyproject.toml` declares only minimum versions, never upper caps.**
   Nothing in this project blocks you from installing a newer release. A test
   (`tests/test_packaging.py`) fails if someone adds `<`, `<=`, `==` or `~=`.
2. **The minimums are real.** `make test-lowest` installs the *oldest* allowed
   version of every dependency (on Python 3.10) and runs the tests. Two
   declared minimums turned out to be false and were corrected:
   - `xgboost 2.0.0`–`2.0.2` crash with the pseudo-Huber loss, so the minimum is 2.0.3;
   - `typer < 0.15.4` crashes on `--help` with current `click`.
3. **`uv.lock` records one known-good set** of exact versions, for
   reproducibility. It is a convenience, not a constraint: regenerate it
   whenever you want newer packages.
4. **Saved models survive upgrades.** Models contain no pickles:
   - preprocessing is stored as JSON + NumPy arrays (`features/portable.py`);
   - the booster is stored in XGBoost's native format.

   A model trained with scikit-learn 1.4 / NumPy 1.26 / XGBoost 2.0 /
   Python 3.10 gives **identical** predictions with scikit-learn 1.9 /
   NumPy 2.5 / XGBoost 3.4 / Python 3.14 (`make portability-check`).
   Before this design, the same upgrade crashed every saved model.
5. **`ct doctor`** checks the environment after any change:
   - Python and package versions against the declared minimums;
   - writable folders and the pseudonymisation key;
   - that the champion model loads and predicts.

## Upgrading

```bash
make upgrade                       # everything to the newest versions, then tests + doctor
make upgrade-package PKG=pandas    # a single package
# without make:
uv lock --upgrade-package pandas && uv sync && uv run pytest -m "not slow" && uv run ct doctor
```

* All green → commit `uv.lock`.
* Something fails → `make rollback-deps` (restores `uv.lock` from git and
  re-syncs). Then read the failing test: it names the module that needs
  attention (see the table below).
* Add a package: `uv add "package>=X.Y"` (always with a minimum, never a cap).
* Raise a minimum: edit `pyproject.toml`, run `uv lock`, and run `make test-lowest`.

## What protects you (CI: `.github/workflows/ci.yml`)

| Job | What it proves |
|---|---|
| `test` | The locked versions work on Python 3.10, 3.12 and 3.14 |
| `lowest` | The minimum versions in `pyproject.toml` really work |
| `latest` | The newest releases of everything work, with `FutureWarning` as errors. Non-blocking, and runs **weekly**, so you hear about a breaking upstream release before you upgrade |
| `portability` | A model trained with the oldest libraries scores identically with the newest |
| `lint` | Code style, plus `uv lock --check` (lock in sync with `pyproject.toml`) |

Locally: `make test`, `make test-lowest`, `make test-latest`, `make portability-check`.

## Verified span (refactoring environment)

| | Oldest tested | Newest tested |
|---|---|---|
| Python | 3.10 | 3.14 |
| numpy | 1.26 | 2.5 |
| pandas | 2.1 | 3.1 |
| scikit-learn | 1.4 | 1.9 |
| scipy | 1.11 | 1.18 |
| xgboost | 2.0.3 | 3.4 |
| pyarrow | 14 | 25 |
| pydantic | 2.5 | 2.14 |
| typer | 0.15.4 | 0.29 |
| streamlit | 1.50 | 1.67 |
| truststore | 0.9 | 0.10 |

## Where library-specific code lives

If an upgrade breaks something, it will almost certainly be in one of these
places. Each is isolated and covered by tests.

| Library | Module | Used for |
|---|---|---|
| scikit-learn | `features/builder.py` | *Fitting* the preprocessing (OneHotEncoder, TfidfVectorizer, TruncatedSVD, SimpleImputer) |
| scikit-learn | `features/portable.py` | Exporting fitted parameters. Applying them only uses `CountVectorizer(vocabulary=…)` and `normalize`; `tests/test_portable.py` checks equality with scikit-learn |
| xgboost | `models/price_model.py` | `xgb.train`, `DMatrix`, `Booster.save_model/load_model` |
| pandas | `data/cleaning.py` | `to_datetime(format="mixed")`, string handling (tested on pandas 2.x and 3.x) |
| streamlit | `app/streamlit_app.py` | Dashboard (`tests/test_app.py` renders it headless) |
| typer | `cli.py` | Command line |
| pydantic | `config.py` | Config validation |
| urllib (stdlib) + truststore | `data/download.py` | datos.gov.co API with the operating-system certificate store; no third-party HTTP client |

Optional extras (`mlflow`, `nlp`) are imported lazily. If they are missing
or broken, only their feature is affected.

## Models from before v1.1

Registries created by v1.0 stored a pickled scikit-learn preprocessor. They
still load with the same scikit-learn version. After an upgrade, `ct doctor`
flags them and suggests `ct train` to re-create them in the portable format
(the old version stays in the registry for reference).
