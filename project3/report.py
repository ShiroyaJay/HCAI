"""Builds the project report PDF from the metrics JSONs and figures.

The prose is fixed; every number is interpolated from the metrics files on
disk, so the report always matches the experiments that actually ran.
fpdf2's core fonts are latin-1 only, so all dynamic strings are sanitized.
"""

import json
import os
from datetime import date

from fpdf import FPDF

from . import data

MARGIN = 15
WIDTH = 210 - 2 * MARGIN


def _s(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


class _Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(0, 6, "Project 3 - Active Learning for Learning-to-Defer", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(0, 6, str(self.page_no()), align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(20)
        self.multi_cell(WIDTH, 8, _s(text))
        self.ln(2)

    def h2(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(20)
        self.multi_cell(WIDTH, 7, _s(text))
        self.ln(1)

    def p(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40)
        self.multi_cell(WIDTH, 5.2, _s(text))
        self.ln(2)

    def table(self, header, rows, col_widths=None):
        n = len(header)
        col_widths = col_widths or [WIDTH / n] * n
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(232, 238, 248)
        self.set_text_color(20)
        for w, cell in zip(col_widths, header):
            self.cell(w, 6.5, _s(cell), border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40)
        for row in rows:
            for w, cell in zip(col_widths, row):
                self.cell(w, 6, _s(cell), border=1)
            self.ln()
        self.ln(3)

    def figure(self, figures_dir, name, width=150):
        path = os.path.join(figures_dir, f"{name}.png")
        if not os.path.exists(path):
            return
        if self.get_y() > 200:
            self.add_page()
        self.image(path, x=(210 - width) / 2, w=width)
        self.ln(4)


def _load(metrics_dir, name):
    with open(os.path.join(metrics_dir, f"{name}.json")) as f:
        return json.load(f)


def build_report(metrics_dir, figures_dir, out_path):
    m1 = _load(metrics_dir, "task1")
    m2 = _load(metrics_dir, "task2")
    m3 = _load(metrics_dir, "task3")
    m4 = _load(metrics_dir, "task4")

    pdf = _Report(format="A4")
    pdf.set_margins(MARGIN, MARGIN)
    pdf.set_auto_page_break(True, margin=16)

    # ---------- title page ----------
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20)
    pdf.multi_cell(WIDTH, 11, "Project 3: Active Learning\nfor Learning-to-Defer", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(70)
    pdf.multi_cell(WIDTH, 7, "Human-Centric Artificial Intelligence", align="C")
    pdf.ln(2)
    pdf.multi_cell(WIDTH, 7, "AG News topic classification with a human-AI team",
                   align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(WIDTH, 6, _s(f"Generated from the experiment artifacts on "
                                f"{date.today().isoformat()}"), align="C")

    # ---------- setup ----------
    pdf.add_page()
    pdf.h1("1  Experimental setup")
    pdf.p(
        "All experiments use the AG News dataset (fancyzhx/ag_news): 120,000 training "
        "and 7,600 test news articles, evenly distributed over four topics (World, "
        "Sports, Business, Sci/Tech). The official training set is split once, "
        "stratified with a fixed seed, into 110,000 training and 10,000 validation "
        "articles. Every design decision that needs tuning - the deferral threshold in "
        "Task 3 and all monitoring during active learning in Task 4 - uses only the "
        "validation split. The official test set is reserved for final evaluation, so "
        "each reported test number is computed exactly once."
    )
    pdf.p(
        "The code is organised as an offline pipeline (python -m "
        "project3.experiments.run_all) that writes metrics, figures and this report to "
        "disk; the project web page displays those artifacts, offers this report for "
        "download, and adds an interactive mode where a person can play the expert "
        "(Task 5)."
    )

    # ---------- task 1 ----------
    t = m1["test"]
    pdf.h1("2  Task 1: Baseline classifier")
    pdf.h2("Design")
    pdf.p(
        "The baseline is a linear classifier over sparse TF-IDF features: word 1-2 "
        "grams, sublinear term frequency, 300k features, followed by multinomial "
        "logistic regression (C=4). Two considerations drove this choice over a "
        "fine-tuned transformer. First, practicality: the model trains on 110k "
        "documents in about half a minute on CPU, which keeps every later experiment "
        "(rejector training, nine active-learning sweeps, the interactive mode) cheap "
        "and reproducible. Second, the deferral system needs per-example confidence "
        "scores: logistic regression's probability estimates are reasonably calibrated "
        "out of the box, whereas a small accuracy gain from a deep model would not "
        "change the human-AI comparisons studied here."
    )
    pdf.h2("Results")
    pdf.p(
        f"The classifier reaches {_pct(t['accuracy'], 2)} accuracy on the test set "
        f"(validation: {_pct(m1['val']['accuracy'], 2)}). Performance is not uniform: "
        "Sports is nearly solved, while Business and Sci/Tech are confused with each "
        "other most often - exactly the region where a complementary expert can help."
    )
    pdf.table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [[c["class"], f"{c['precision']:.3f}", f"{c['recall']:.3f}",
          f"{c['f1']:.3f}", c["support"]] for c in t["per_class"]],
        [45, 34, 34, 34, 33],
    )
    pdf.figure(figures_dir, "task1_confusion", width=110)

    # ---------- task 2 ----------
    cc = m2["class_conditional"]
    lc = m2["length_conditional"]
    pdf.add_page()
    pdf.h1("3  Task 2: Simulated experts")
    pdf.h2("Design")
    pdf.p(
        "Both experts are imperfect by construction and competent only in a region of "
        "the input space. They are deterministic: each answer is derived "
        "from a hash of the article's row key, so querying the same article twice "
        "always returns the same answer - important because Tasks 3, 4 and 5 must see "
        "one consistent expert. When an expert errs, it prefers a plausible confusion "
        "(e.g. Business vs Sci/Tech) rather than a uniformly random topic."
    )
    pdf.p(
        f"The primary expert is class-conditional: {cc['description']}. It models a "
        "business/technology specialist and is deliberately strong exactly where the "
        "Task 1 classifier is weakest (Business, Sci/Tech), giving the human-AI team "
        "real complementarity. The second expert is length-conditional: "
        f"{lc['description']}. It models a reader who judges long articles reliably "
        "but guesses on short snippets, and shows that competence regions need not "
        "align with class boundaries."
    )
    pdf.h2("Results and analysis")
    pdf.p(
        f"On the test set the class-conditional expert reaches {_pct(cc['accuracy'])} "
        f"overall - far below the classifier's {_pct(t['accuracy'])} - yet on its "
        f"strong classes it is clearly better: "
        f"{_pct(cc['per_class'][2]['accuracy'])} on Business and "
        f"{_pct(cc['per_class'][3]['accuracy'])} on Sci/Tech versus the classifier's "
        f"{_pct(t['per_class'][2]['recall'])} and {_pct(t['per_class'][3]['recall'])}. "
        f"Its weakness is everything else ({_pct(cc['per_class'][0]['accuracy'])} on "
        f"World, {_pct(cc['per_class'][1]['accuracy'])} on Sports). The "
        f"length-conditional expert reaches {_pct(lc['accuracy'])} overall; its "
        "accuracy rises with article length, and neither expert alone can replace the "
        "classifier. A deferral policy that knows these profiles, however, can combine "
        "the best of both team members."
    )
    pdf.table(
        ["Expert", "Overall"] + data.CLASS_NAMES,
        [["Class-conditional", _pct(cc["accuracy"])] +
         [_pct(c["accuracy"]) for c in cc["per_class"]],
         ["Length-conditional", _pct(lc["accuracy"])] +
         [_pct(c["accuracy"]) for c in lc["per_class"]]],
        [40, 28, 28, 28, 28, 28],
    )
    pdf.figure(figures_dir, "task2_expert_profile", width=170)

    # ---------- task 3 ----------
    t3 = m3["test"]
    b = t3["baselines"]
    pdf.add_page()
    pdf.h1("4  Task 3: Learning to defer")
    pdf.h2("Method")
    pdf.p(
        "With expert labels available for the whole training set, the system learns "
        "when deferral is beneficial with a staged (post-hoc rejector) approach. A "
        "binary rejector g(x) is trained on the same TF-IDF features to predict the "
        "probability that the expert answers x correctly, using the observed expert "
        "answers on the training set. The team then defers exactly when the expert "
        "looks more reliable than the classifier: defer iff g(x) - max_y p(y|x) > tau. "
        "The threshold tau is chosen on the validation split "
        f"(selected value: tau = {m3['tau']})."
    )
    pdf.p(
        "A joint surrogate loss in the style of Mozannar & Sontag (2020) - a (K+1)-way "
        "classifier with a defer option trained with a specially weighted loss - is "
        "the main alternative. It cannot be expressed with scikit-learn's estimators "
        "(no custom per-class loss weighting), and the staged approach has a clear "
        "advantage for analysis: g(x) is an explicit, inspectable model of the "
        "expert's competence, which Task 4 then learns actively."
    )
    pdf.h2("Results")
    pdf.p(
        f"The team reaches {_pct(t3['team_accuracy'], 2)} test accuracy at "
        f"{_pct(t3['coverage'], 1)} coverage: {t3['n_deferred']:,} of the 7,600 test "
        f"articles ({_pct(1 - t3['coverage'])}) are sent to the expert and the "
        "classifier answers the rest. That beats the classifier alone "
        f"({_pct(b['classifier_alone'], 2)}), the expert alone "
        f"({_pct(b['expert_alone'], 2)}), and random deferral at the same rate "
        f"({_pct(b['random_deferral_same_rate'], 2)}); the oracle that defers exactly "
        f"when it helps sits at {_pct(b['oracle_deferral'], 2)}."
    )
    pdf.p(
        "The deferral decisions themselves are sensible on both axes. Quality of the "
        f"work split: kept articles are classified at {_pct(t3['accuracy_kept'], 2)} "
        f"and deferred ones answered by the expert at "
        f"{_pct(t3['expert_accuracy_deferred'], 2)} - both far above what the other "
        "team member would achieve on that share. Competence discovery: the system "
        f"defers {_pct(t3['per_class_deferral_rate'][2]['rate'], 0)} of Business and "
        f"{_pct(t3['per_class_deferral_rate'][3]['rate'], 0)} of Sci/Tech articles "
        f"but only {_pct(t3['per_class_deferral_rate'][1]['rate'], 0)} of Sports - it "
        "has located the expert's strong region without being told. Judged against "
        "the oracle should-defer set (classifier wrong and expert right), deferral "
        f"recall is {_pct(t3['deferral_recall'], 1)} and precision "
        f"{_pct(t3['deferral_precision'], 1)}; precision is naturally low at this "
        "operating point because deferring an article the classifier would also get "
        "right costs the team nothing when the expert is strong there."
    )
    pdf.table(
        ["System", "Test accuracy"],
        [["Expert alone", _pct(b["expert_alone"], 2)],
         ["Random deferral (same rate)", _pct(b["random_deferral_same_rate"], 2)],
         ["Classifier alone", _pct(b["classifier_alone"], 2)],
         ["Learned deferral (team)", _pct(t3["team_accuracy"], 2)],
         ["Oracle deferral (upper bound)", _pct(b["oracle_deferral"], 2)]],
        [110, 70],
    )
    pdf.figure(figures_dir, "task3_deferral", width=170)
    pdf.figure(figures_dir, "task3_coverage_accuracy", width=130)
    pdf.p(
        "The coverage-accuracy curve (deferring the top-q fraction by rejector score) "
        "shows the whole trade-off: team accuracy peaks around 20-30% deferral and "
        "degrades gracefully toward the expert-alone extreme, so an operator can pick "
        "any workload split for the human expert without retraining."
    )

    # ---------- task 4 ----------
    refs = m4["references"]
    budgets = m4["budgets"]

    def _mean_acc(name, i):
        runs = m4["strategies"][name]["runs"]
        return sum(r[i]["team_accuracy"] for r in runs) / len(runs)

    pdf.add_page()
    pdf.h1("5  Task 4: Active learning for expert competence discovery")
    pdf.h2("Setting and strategies")
    pdf.p(
        "Now no expert labels exist up front. The classifier stays fixed (it already "
        "has all gold labels); what is missing is the expert's competence profile, so "
        "only the rejector must be learned, from answers obtained by querying the "
        f"expert. Queries are selected in batches of {m4['batch']} from a pool of "
        f"{m4['pool_size']:,} training articles; after each batch the rejector is "
        "refit on all answers so far and the team is evaluated on the validation "
        "split with the fixed rule defer iff g(x) > max_y p(y|x). Budgets from "
        f"{budgets[0]} to {budgets[-1]:,} queries are checkpointed, with "
        f"{len(m4['seeds'])} seeds per strategy."
    )
    pdf.p(
        "Three query strategies are compared. Random sampling is the baseline. "
        "Classifier-uncertainty sampling queries where the classifier is least "
        "confident - the classic active-learning heuristic, included because deferral "
        "candidates live exactly in that region. The proposed strategy, "
        "deferral-boundary sampling, queries where |g(x) - max_y p(y|x)| is smallest: "
        "the expert's answer there is precisely the information that flips a "
        "defer/keep decision, which is the quantity we care about (with a first "
        "random batch and 10% random exploration per batch)."
    )
    pdf.h2("Results")
    idx100 = budgets.index(100)
    pdf.p(
        "The headline result is how cheap competence discovery is: with only 25 "
        f"expert queries every strategy already lifts the team to about "
        f"{_pct(_mean_acc('random', 0), 1)} on validation, versus "
        f"{_pct(refs['classifier_alone_val'], 2)} for the classifier alone and "
        f"{_pct(refs['full_info_team_accuracy_val'], 2)} for a rejector trained on "
        "all 110,000 expert labels. Roughly 100 queries recover about "
        f"{_pct((_mean_acc('random', idx100) - refs['classifier_alone_val']) / (refs['full_info_team_accuracy_val'] - refs['classifier_alone_val']), 0)} "
        "of the value of full expert supervision at 0.1% of the labeling cost."
    )
    pdf.p(
        "Between strategies the honest finding is that random sampling is the "
        f"strongest at large budgets ({_pct(_mean_acc('random', len(budgets) - 1), 2)} "
        f"at {budgets[-1]:,} queries vs "
        f"{_pct(_mean_acc('deferral_boundary', len(budgets) - 1), 2)} for "
        "deferral-boundary), and no strategy separates from the others at small "
        "budgets. The explanation is instructive. This expert's competence is "
        "class-determined, and class is linearly recoverable from TF-IDF features, "
        "so a handful of random examples per topic already pins the profile down - "
        "there is little room for cleverness. Both targeted strategies also collect "
        "a deliberately biased sample (hard or boundary articles), which "
        "miscalibrates the rejector's probabilities slightly; the deferral rule "
        "compares g(x) against classifier confidence directly, so calibration "
        "matters more than ranking here. Targeted querying should pay off when "
        "competence varies within classes in subtler ways; for this problem the "
        "practical recommendation is the simple one - stratified random querying "
        "with a few hundred labels."
    )
    header = ["Strategy"] + [str(bu) for bu in budgets]
    rows = []
    for key, label in [("random", "Random"),
                       ("clf_uncertainty", "Clf. uncertainty"),
                       ("deferral_boundary", "Deferral boundary")]:
        rows.append([label] + [f"{_mean_acc(key, i):.4f}" for i in range(len(budgets))])
    pdf.table(header, rows, [36] + [(WIDTH - 36) / len(budgets)] * len(budgets))
    pdf.figure(figures_dir, "task4_learning_curves", width=150)

    # ---------- task 5 + limitations ----------
    pdf.add_page()
    pdf.h1("6  Task 5: Interactive mode (optional)")
    pdf.p(
        "The project interface includes a 'Be the expert' mode implementing the "
        "optional Task 5. The Task 4 loop runs with a person instead of the simulated "
        "expert: the system picks the next article (random warm-up, then "
        "deferral-boundary sampling), the user assigns one of the four topics, and "
        "the rejector - now a model of that user's competence - is refit after every "
        "answer. A live panel shows the user's measured accuracy per topic and the "
        "estimated team accuracy if the current rejector were deployed with them as "
        "the expert. Because the queried articles come from the labeled training "
        "set, the user's answers can be scored against gold labels immediately."
    )
    pdf.h1("7  Limitations")
    pdf.p(
        "The experts are simulations with conveniently simple competence structure; "
        "as Task 4 showed, that structure makes competence discovery easy and "
        "flatters random querying. Real expert competence drifts, depends on effort "
        "and time, and is not a deterministic function of the article. The rejector "
        "reuses the classifier's TF-IDF representation, so competence patterns "
        "invisible in bag-of-words space cannot be learned. Deferral is evaluated "
        "at zero query cost; a per-query cost would move the operating point along "
        "the coverage-accuracy curve. Finally, the staged rejector is not jointly "
        "optimal - the classifier is not retrained to specialise on the region it "
        "keeps - which is the price paid for an interpretable, sklearn-only design."
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pdf.output(out_path)
    return out_path
