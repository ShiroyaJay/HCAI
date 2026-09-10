Project 2: Explainability — Implementation Plan                                                    ↑

ntext                                                                                            ↑

oject 2 of the Human-Centric AI course (TUHH) is the Explainability brief                        ↑
roject2/HCAI-project_02.pdf). It works on the Palmer Penguins dataset
   (target = species: Adelie / Gentoo / Chinstrap) and asks for one interactive                       ↑
   web interface that grows across five linked tasks:
                                                                                                      ↑
   1. Fit a decision tree; show the tree, its test accuracy, and number of leaves.
   2. A λ slider that selects, among trees of varying regularization, the maximizer                   ↑
   of acc_test − λ·Ω(f) where Ω = number of leaves.
   3. Same idea for logistic regression with a suitable complexity measure Ω.                         ↑
   4. A Counterfactuals region (pick an example x + target class → nearest
   counterfactuals via local sampling + MAD-weighted L1 distance), linked to the                      ↑
   selected model class and λ.
 A Feature Effect Plots region (hand-written PDP + ALE for each of the                           ↑
ur numerical features, three species curves per plot), linked to model + λ.
                                                                                                 ↑
is is a graded submission and must integrate into the existing Django site exactly
e way project1 does (one Django app behind the shared home/launch page).                         ↑

cisions locked with the user                                                                     ↑

   - Data: vendor a penguins.csv in project2/data/ (no new dependency), mirroring                     ↑
   how project1 ships iris.csv.
   - Interactivity: minimal-JS — controls auto-submit on change (this.form.submit()),                 ↑
   state carried in the URL via GET params; rendering stays fully server-side
   (matplotlib → PNG in media/), matching project1's plotting approach.                               ↑
   - Conceptual notes: include concise written answers to the PDF's conceptual
   questions (ALE exact-vs-discretized, categorical noising, logreg Ω choice) in a                    ↑
   project2/README.md and surfaced briefly on the page.
                                                                                                      ↑
   Conventions to mirror (from project1 exploration)
                                                                                                      ↑
   - Function-based views; per-request training (dataset is tiny — ~333 rows — so cheap).
   - matplotlib.use("Agg"); save PNG to MEDIA_ROOT; embed as <img src=MEDIA_URL+name>.                ↑
   - Templates extend templates/base.html; per-app CSS at static/project2/style.css
   with a .p2-* class prefix; cache-bust with ?v=.                                                    ↑
   - App wiring: app_name = 'project2', mounted in pbl/urls.py, listed in
   home/views.py projects, linked in base.html nav.                                                   ↑
   - media/* is git-ignored (keeps .gitkeep), so generated PNGs need no gitignore edit.
                                                                                                      ↑
   ---
   Architecture                                                                                       ↑

   One page, one main view. All UI state lives in GET query params so every control is a              ↑
   small form that auto-submits and preserves the others via hidden inputs:
                                                                                                      ↑
   /project2/?model=tree&lam=0.02&cf_index=12&cf_target=Gentoo&feature=bill_length_mm
                                                                                                      ↑
   model ∈ {tree, logreg}; lam = slider value; cf_index/cf_target drive the
   counterfactual region; feature drives the feature-effect region. Changing λ or model               ↑
   re-selects the active model and so correctly refreshes all regions (the model
   changed). Generated PNG filenames are keyed by session + region + a short hash of the              ↑
   relevant params, with a ?v=<hash> cache-buster on the <img>.
                                                                                                      ↑
   New files (under project2/)
                                                                                                      ↑
   - data.py — load/clean penguins CSV (drop NaN rows), constants:
   TARGET, NUMERIC_FEATURES (bill_length_mm, bill_depth_mm, flipper_length_mm,                        ↑
   body_mass_g), CATEGORICAL_FEATURES (island, sex, year), train/test split
   (random_state fixed, 80/20, stratified), per-feature MAD helper.                                   ↑
   - ml.py — pipelines + candidate generation + selection + model display:
     - tree: Pipeline(OrdinalEncoder(categoricals) + passthrough numerics → DecisionTreeClassifier);  ↑
   candidates over max_leaf_nodes ∈ {2..M} (M = leaves of unconstrained tree, capped ~30);
   record (acc_test, Ω=n_leaves).                                                                     ↑
     - logreg: Pipeline(OneHot(categoricals) + StandardScaler(numerics) → LogisticRegression(penalty='l1', solver='saga'/'liblinear', multi_class));                         ↑
   candidates over C ∈ logspace; Ω = number of non-zero coefficients;
   record (acc_test, Ω).                                                                              ↑
     - select_model(candidates, lam) → argmax of acc_test − λ·Ω (note in code: this λ
   is the selection meta-parameter, distinct from the fitting param max_leaf_nodes / C).              ↑
     - plot_tree_png() via sklearn.tree.plot_tree (allowed — visualization only);
   logreg_coef_table() / coefficient bar chart for the logreg view.                                   ↑
   - counterfactuals.py — generate_counterfactuals(predict_fn, x_row, target, mad, k):
   sample N around x — numeric: Gaussian noise scaled by feature MAD/std; categorical &               ↑
   year: with prob p resample a different category uniformly (the PDF's
   binary/categorical-noising requirement); keep rows predicted as target; rank by                    ↑
   MAD-weighted L1 distance (numeric: |Δ|/MAD; categorical: indicator mismatch);
   return best k. Iterative fallback: if < k found, grow N and inflate variance.                      ↑
   - feature_effects.py — hand-written (numpy/pandas only; no PDP/ALE library):
     - pdp(predict_proba_fn, df, feature, grid) → for each grid value, set feature =                  ↑
   value for all rows, average predicted prob per class → 3 curves.
     - ale(model, df, feature, bins) → centered accumulated local effects per class:                  ↑
         - logreg → exact partial derivatives (closed-form gradient of predicted
   probability w.r.t. the standardized numeric feature).                                              ↑
       - tree → discretization: bin the feature; per bin take mean difference of
   predictions when the feature is moved from the lower to the upper bin edge;                        ↑
   accumulate and center (finite-difference ALE).
     - plotting helpers → two PNGs (PDP, ALE), each with three species curves.                        ↑
   - views.py — single index(request) reading the GET params above; builds candidates,
   selects the active model, renders model/accuracy/Ω, optional CF table, optional                    ↑
   feature plots into project2/index.html. Thin; heavy lifting in the modules above.
   - urls.py — app_name = 'project2', path('', views.index, name='index').                            ↑
   - templates/project2/index.html — extends base.html; control bar (model radios +
   λ <input type=range> auto-submitting) + three region cards; conceptual-notes blurb.                ↑
   - static/project2/style.css — .p2-* styles consistent with the site.
   - data/penguins.csv — vendored canonical dataset (columns: species, island,                        ↑
   bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year). Fetched
   once from the palmerpenguins source (allisonhorst GitHub raw CSV) and committed.                   ↑
   - README.md — run notes + the conceptual answers.
                                                                                                      ↑
   Edits to existing files
                                                                                                      ↑
   - pbl/settings.py — add "project2" to INSTALLED_APPS.
   - pbl/urls.py — add path("project2/", include("project2.urls")).                                   ↑
   - home/views.py — append {"name": "Project 2", "url_name": "project2:index"}.
   - templates/base.html — add a Project 2 nav link.                                                  ↑
   - (project2/ is created via python manage.py startapp project2; empty
   models.py/admin.py/tests.py left as generated.)                                                    ↑

   ---                                                                                                ↑
   Task → implementation mapping
                                                                                                      ↑
   - Task 1 → tree pipeline + plot_tree_png + show acc_test and clf.get_n_leaves().
   - Task 2 → tree candidates + λ slider + select_model (Ω = leaves).                                 ↑
   - Task 3 → logreg candidates + same slider/selection (Ω = non-zero coeffs); model
   view shows coefficients instead of a tree.                                                         ↑
   - Task 4 → Counterfactuals region using the active model's predict.
   - Task 5 → Feature Effect Plots region: PDP + ALE (exact for logreg, discretized                   ↑
   for tree) using the active model.
                                                                                                      ↑
   Build order (each step independently verifiable)
                                                                                                      ↑
   1. App scaffold + wiring (startapp, settings, urls, home link, nav) → verify: /project2/
   loads a placeholder page; /home/ shows the Project 2 button. → python manage.py check.             ↑
   2. Vendor penguins.csv + data.py → verify: a one-off script/shell loads it, prints
   shape (~333×8) and the feature lists; no NaNs after cleaning.                                      ↑
   3. ml.py tree path + Task 1 view/template (tree image, acc, leaves) → verify in browser.
   4. Tree candidates + λ slider (Task 2) → verify: sliding λ changes the displayed tree              ↑
   and its leaf count monotonically (more λ → fewer leaves).
 Logreg path (Task 3): model radio switches tree↔logreg; coefficients + Ω shown;                 ↑
ider re-selects → verify in browser.
 Counterfactuals (Task 4) → verify: pick an example whose class ≠ target, get k valid
unterfactuals all predicted as the target; switching model/λ changes them.
 Feature effects (Task 5) → verify: each of the 4 numeric features yields a PDP and an
E plot with 3 species curves; logreg uses exact derivative, tree uses binning.
 README.md conceptual notes + final styling pass + requirements.txt unchanged
   (CSV route adds no dependency).

 Verification (end-to-end)

   - source venv/bin/activate && python manage.py check → no errors.
   - python manage.py runserver, open http://127.0.0.1:8000/project2/:
   - Toggle model tree↔logreg; move λ slider → model view, CF region, and feature plots
   all update consistently.
     - Counterfactuals: choose x and a different target → table of k counterfactuals,
ch verified (re-predicted) as the target class; sanity-check that only a few
   features changed and changes are plausibly small (MAD-weighted).
   - Feature effects: PDP curves lie in [0,1] and (per grid point) sum ≈ 1 across the
 three species; ALE curves are centered near 0.
 - Lightweight numeric sanity checks (in project2/tests.py or a scratch script):
 selection returns the leaf-minimal model at large λ and the most-accurate at λ=0;
 PDP class probabilities sum to ~1; counterfactual predictions all equal the target.

 Open notes / assumptions

 - λ slider range fixed to a small interval (≈ [0, 0.1]) chosen so the slider sweeps the
 full accuracy↔simplicity spectrum for this dataset; tunable constant in ml.py.
 - Rows with missing values are dropped (standard for this dataset; ~333 rows remain).
 - year treated as a categorical feature (only 2007–2009); the four Task-5 numeric
 features are the body-measurement columns, as the PDF specifies ("four numerical features").
 - plot_tree and LogisticRegression/DecisionTreeClassifier are sklearn-provided and
 allowed; the only hand-written-from-scratch requirement (PDP/ALE) is honored.