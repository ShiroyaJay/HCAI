# Project 3: Active Learning for Learning-to-Defer — Implementation Plan

## Context

Project 3 of the Human-Centric AI course (TUHH) is the learning-to-defer brief
(`project3/HCAI-project_03.pdf`). It works on the AG News dataset
(https://huggingface.co/datasets/fancyzhx/ag_news — 120k train / 7.6k test news
articles, 4 topics: World, Sports, Business, Sci/Tech) and asks for five linked
tasks:

1. Train a classifier on the full labeled training set; report test accuracy
   (the baseline the human-AI team should match or surpass).
2. Design ≥ 1 **simulated expert** — imperfect, with expertise in specific
   regions of the input space; analyze strengths/weaknesses, report test accuracy.
3. **Learning-to-defer**: per input, the system predicts itself or defers to
   the expert; evaluate classification accuracy *and* deferral-decision quality.
4. **Active learning** with no expert labels upfront: query the expert on
   chosen examples to efficiently learn when deferral is beneficial; justify
   the strategy, report results.
5. (Optional, included) An interactive interface where the user plays the expert.

Deliverable: a **PDF report** (experiments, design justifications, results)
**downloadable from the project interface**.

## Decisions locked with the user

- Classifier: **TF-IDF + linear model** (sklearn only; no torch/transformers).
- Task 5 interactive mode: **included**.
- Report: generated programmatically from the real experiment results, served
  via a download button.
- Architecture: experiments run **offline as scripts**; the Django app is a
  read-only dashboard over the stored artifacts + the interactive mode.

## Architecture

Two layers, mirroring project2's conventions (Django-free logic modules,
`@lru_cache`, media PNGs, empty models.py):

1. **Offline pipeline** — `python -m project3.experiments.run_all`
   (`--stages data,task1,task2,task3,task4,figures,report` to run a subset).
   Plain `python -m`, not a management command: the repo has none, and the
   scientific code stays Django-free. Outputs (all gitignored):
   - `project3/data/` — AG News parquet (~30 MB), downloaded once from the HF
     resolve endpoint with stdlib urllib (fallback URL + clear manual
     instructions on failure); row counts verified (120,000 / 7,600).
   - `project3/artifacts/models/` — joblib classifier + rejector (new pattern
     for this repo, justified: refitting 110k docs per request is not viable).
   - `project3/artifacts/metrics/task{1..4}.json` — every number the web page
     and the report consume.
   - `project3/artifacts/figures/*.png` + copies at `media/project3_<name>.png`.
   - `project3/artifacts/report.pdf` — built with fpdf2 (pure Python; no
     pandoc/LaTeX on the machine); all dynamic strings latin-1-sanitized.
2. **Django app `project3`** — dashboard (per-task result cards, tables,
   figures), report download button (`FileResponse(as_attachment=True)`,
   404 + hidden button while missing), and the Task 5 loop. Every page
   degrades gracefully with a "run the pipeline first" notice on a fresh clone.

Data protocol: the official 120k train set is split once (stratified, seed 42)
into 110k train / 10k validation. All tuning (deferral threshold τ, active-
learning monitoring) uses validation only; **the official test set is used
exactly once per reported number**.

## Files

New app `project3/`: `data.py` (download/load/split), `ml.py` (Task 1),
`experts.py` (Task 2), `l2d.py` (Task 3), `active.py` (Task 4), `plots.py`,
`report.py`, `interactive.py` (Task 5), `experiments/run_all.py`,
`views.py`, `urls.py`, `templates/project3/{index,interactive}.html`,
`static/project3/style.css` (`.p3-*` prefix), `tests.py`.

Edits to existing files: `pbl/settings.py` (INSTALLED_APPS), `pbl/urls.py`,
`templates/base.html` (nav), `home/views.py` (projects list), `.gitignore`
(`project3/data/`, `project3/artifacts/`), `requirements.txt`
(+ `pyarrow==16.1.0`, `fpdf2==2.7.9`), `README.md` (status table).

## Task → implementation mapping

- **Task 1** → `TfidfVectorizer(1-2 grams, sublinear_tf, 300k features)` +
  multinomial `LogisticRegression(C=4)`. Chosen for CPU practicality (~33 s
  training) and because its `predict_proba` supplies the confidence scores the
  deferral rule needs. **Result: 92.41% test accuracy.**
- **Task 2** → deterministic per-example oracles: each answer derives from
  `sha1(seed:row_key)`, so every task and test sees one consistent expert.
  - `ClassConditionalExpert` (primary): 97% correct on Business/Sci-Tech, 55%
    elsewhere — a business/tech specialist strong *exactly where the Task 1
    classifier is weakest* (design choice: an earlier World/Sports-strong
    expert gave a flat deferral story because the classifier is already strong
    there). Wrong answers prefer plausible confusions (Business↔Sci/Tech).
    **75.9% test accuracy.**
  - `LengthConditionalExpert` (analysis contrast): 93% on articles ≥ 215 chars,
    62% below — competence orthogonal to class boundaries. **81.9%.**
- **Task 3** → staged post-hoc rejector: train `g(x) = P(expert correct | x)`
  (logreg on the *same fitted* TF-IDF features, labels = expert==gold on the
  110k train set); defer iff `g(x) − max_y p_clf(y|x) > τ`, τ tuned on
  validation (τ = 0.04). The Mozannar–Sontag joint surrogate needs a custom
  weighted (K+1)-way loss sklearn cannot express; the staged g(x) is also an
  explicit, inspectable competence model that Task 4 then learns actively.
  Evaluation: team accuracy, coverage, kept/deferred accuracy, deferral
  precision/recall vs the oracle should-defer set {clf wrong ∧ expert right},
  per-class deferral rates, coverage-accuracy curve; baselines = classifier
  alone / expert alone / random deferral at the same rate / oracle deferral.
  **Result: team 95.17% at 20.1% deferral (clf 92.41%, random 89.14%, oracle
  98.57%); deferral concentrates on Business 31% / Sci-Tech 34% vs Sports 5%.**
- **Task 4** → pool-based AL (30k pool, precomputed sparse TF-IDF + clf
  probabilities); classifier fixed, only the rejector learns from queried
  expert answers; refit after each batch (100, clamped so every budget
  checkpoint 25→5000 is hit exactly); 3 seeds. Strategies: random /
  classifier-uncertainty / **deferral-boundary sampling** (smallest
  `|g − max p|` — the answer there is exactly the information that flips a
  defer/keep decision; first batch random, ε=0.1 exploration). Fixed rule
  τ=0 during AL; upper bound = Task 3 rejector under the same rule.
  **Honest finding: ~25–100 queries already recover most of the value of 110k
  expert labels, and random matches/beats the targeted strategies — the
  competence signal is class-determined and linearly recoverable, and biased
  sampling slightly miscalibrates g. Reported as such in the report.**
- **Task 5** → "Be the expert" page: warm-up random queries, then
  deferral-boundary selection; the user labels articles (4 topic buttons,
  POST-redirect-GET); g is refit on their answers (<100 ms; heavy assets in a
  process-level `@lru_cache`; session stores only indices + labels). Sidebar:
  queries used, live per-topic accuracy vs gold, estimated team accuracy on a
  held-out slice (deferred share uses ĝ — labeled as an estimate in the UI).

## Build order (each step independently verifiable)

1. App scaffold + 4-point wiring → `/project3/` placeholder loads, nav works.
2. Dependencies (`pyarrow`, `fpdf2`) + `.gitignore` entries.
3. `data.py` + data stage → 120,000 / 7,600 rows verified.
4. `ml.py` + task1 stage → test accuracy ≥ 0.90.
5. `experts.py` + task2 stage → profiles match the configured rates.
6. `l2d.py` + task3 stage → oracle ≥ team > max(clf alone, random at rate).
7. `active.py` + task4 stage → curves approach the Task 3 bound.
8. `plots.py` + figures stage (dataviz palette validated; figures eyeballed
   and label collisions fixed).
9. Dashboard views + download button.
10. `report.py` + report stage (prose interpolates the metrics JSONs, so the
    report can never drift from the experiments).
11. Task 5 interactive mode.
12. `tests.py` + clean full-pipeline rerun to prove reproducibility.

## Verification (end-to-end)

```bash
source venv/bin/activate && pip install -r requirements.txt
python -m project3.experiments.run_all      # full pipeline, ~3 min after download
python manage.py test                        # 34 tests (12 new), all green
python manage.py runserver                   # then in the browser:
```

- `/home/` and the nav show Project 3; `/project3/` renders all four task
  sections with tables + figures; the report button downloads a valid PDF.
- `/project3/interactive/` → label ~15 articles; competence profile and
  estimated team accuracy update live after the 10-answer warm-up.
- Graceful degradation: with `project3/artifacts/` absent the dashboard still
  loads with run instructions and `/project3/report/` returns 404.
- Reproducibility: wiping `artifacts/` and rerunning the pipeline reproduces
  every reported number exactly (fixed seeds + hash-based experts).

## Open notes / assumptions

- Expert strength/weakness rates (0.97/0.55) are a documented design choice
  under Task 2's "design" framing, tuned so the team has real complementarity
  to exploit; the report states this explicitly.
- The rejector reuses the classifier's TF-IDF representation — competence
  patterns invisible in bag-of-words space cannot be learned (noted under
  Limitations in the report).
- Deferral is evaluated at zero query cost; a per-query cost would move the
  operating point along the coverage-accuracy curve.
- `media/project3_*.png` copies exist only for web display; the report embeds
  the canonical copies from `artifacts/figures/`.
