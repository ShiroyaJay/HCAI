"""The Task 1-3 report, built on demand as PDF bytes.

Hybrid by design: the figures are pre-rendered PNGs committed under assets/
(so matplotlib -- a 1-3s cold import -- never touches the request path), while
the prose, tables and numbers are assembled here and cached as immutable bytes.

The bytes are returned to an HttpResponse rather than a FileResponse over a
shared BytesIO, whose read position is stateful and would hand a second
concurrent request an empty file.

fpdf2's core fonts are latin-1 only, so every dynamic string goes through _s().
"""

import json
import os
from functools import lru_cache

from fpdf import FPDF

from . import features

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
METRICS = os.path.join(ASSETS, "sim_metrics.json")

MARGIN = 16
WIDTH = 210 - 2 * MARGIN


def _s(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(0, 6, "Project 4 - Preference Elicitation", align="R")
        self.ln(9)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(0, 6, str(self.page_no()), align="C")

    def h1(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(20)
        self.multi_cell(WIDTH, 8, _s(text))
        self.ln(2)

    def h2(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(20)
        self.multi_cell(WIDTH, 6.5, _s(text))
        self.ln(1)

    def p(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40)
        self.multi_cell(WIDTH, 5.1, _s(" ".join(str(text).split())))
        self.ln(2)

    def bullets(self, items):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40)
        for it in items:
            x = self.get_x()
            self.cell(4, 5.1, "-")
            self.multi_cell(WIDTH - 4, 5.1, _s(" ".join(str(it).split())))
            self.set_x(x)
        self.ln(2)

    def formula(self, text):
        self.set_font("Courier", "", 9.5)
        self.set_text_color(20)
        self.multi_cell(WIDTH, 5, _s(text))
        self.ln(2)

    def table(self, headers, rows, widths):
        """A table whose final column wraps; earlier columns are single-line."""
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(60)
        for h, w in zip(headers, widths):
            self.cell(w, 6, _s(h), border="B")
        self.ln(6)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(40)
        last_w = widths[-1]
        for row in rows:
            text = _s(" ".join(str(row[-1]).split()))
            lines = len(self.multi_cell(last_w, 4.6, text, dry_run=True, output="LINES"))
            height = max(5.4, lines * 4.6)
            if self.get_y() + height > 268:
                self.add_page()
            y0, x = self.get_y(), self.l_margin
            for value, w in list(zip(row, widths))[:-1]:
                self.set_xy(x, y0)
                self.cell(w, height, _s(value))
                x += w
            self.set_xy(x, y0)
            if lines == 1:                      # keep single-line rows aligned
                self.cell(last_w, height, text)
            else:
                self.multi_cell(last_w, 4.6, text)
            self.set_y(y0 + height)
        self.ln(3)

    def maybe_break(self, threshold=120):
        """Start a new page only if the current one is already well used."""
        if self.get_y() > threshold:
            self.add_page()

    def figure(self, name, caption):
        path = os.path.join(ASSETS, f"{name}.png")
        if not os.path.exists(path):
            return
        if self.get_y() > 180:
            self.add_page()
        self.image(path, w=WIDTH)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130)
        self.multi_cell(WIDTH, 4.6, _s(caption))
        self.ln(3)


def _n_for_dz(dz, alpha=0.05, power=0.80):
    """Participants needed for a paired test, by normal approximation.

    Duplicated from experiments/simulate.py rather than imported: the report is
    built inside a web request and must not reach into the offline experiment
    package, which exists to be run from a shell.
    """
    from math import ceil
    from statistics import NormalDist
    nd = NormalDist()
    return int(ceil(((nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)) / abs(dz)) ** 2)) + 1


def _metrics():
    if os.path.exists(METRICS):
        with open(METRICS) as f:
            return json.load(f)
    return None


FEATURE_TABLE = [
    ("genre_* (12)", "12", "genres", "Raw 0/1 for the 12 commonest genres. The single strongest axis of film taste, and directly nameable by a user."),
    ("year", "1", "title_year", "Standardised release year: era preference."),
    ("duration", "1", "duration", "Winsorised then standardised; tolerance for long films."),
    ("rating_*", "3", "content_rating", "18 messy levels collapsed to family / teen / unrated, with adult (R, NC-17) as the dropped reference."),
    ("imdb_score", "1", "imdb_score", "Standardised; taste for critically acclaimed vs. guilty-pleasure films."),
    ("popularity", "1", "num_voted_users", "log10 vote count: mainstream vs. niche."),
    ("non_english", "1", "language", "Appetite for subtitled cinema."),
    ("black_and_white", "1", "color", "Appetite for classic-era film."),
]

REJECTED = [
    ("plot_keywords TF-IDF", "Thousands of dimensions. Unidentifiable from ~50 comparisons (see the separability figure)."),
    ("director / actor one-hots", "~4,000 levels, almost all seen once or twice. Extreme sparsity, no generalisation."),
    ("budget", "Denominated in unlabelled local currency (Korean and Japanese films run to the billions) and ~10% missing. Not comparable across rows."),
    ("gross", "US domestic only, not inflation-adjusted, ~17% missing. Also not something a viewer has a preference over."),
    ("movie_facebook_likes", "Zero-inflated by release year, so it acts as a covert recency proxy that would fight the era feature."),
]


def _build():
    m = _metrics()
    pdf = _Report()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 21)
    pdf.set_text_color(20)
    pdf.multi_cell(WIDTH, 10, "Preference Elicitation for a Movie Recommender")
    pdf.set_x(MARGIN)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90)
    pdf.multi_cell(WIDTH, 6, "Human-Centric Artificial Intelligence - Project 4\n"
                             "Feature design, a ranking preference model, and the design of a "
                             "user study comparing two elicitation interfaces")
    pdf.ln(4)

    pdf.p("This report covers Tasks 1 to 3. Task 4 (the participant-facing interface) "
          "is the running application this document was downloaded from; the 'Start the "
          "study' button on the project page begins a complete session. Everything "
          "described here as part of the protocol is implemented there: where the two "
          "would disagree, the report has been corrected to match the interface.")
    pdf.p("The setting: the IMDB 5000 dataset describes about 5,000 films but contains no "
          "user ratings at all. A recommender therefore cannot be bootstrapped from "
          "history; it has to ask. Each user is modelled by a latent preference vector w, "
          "with the utility of a film x given by U(x) = w'x, and the whole design problem "
          "is how to estimate w from as few interactions as a person will tolerate.")

    # ---------------- Task 1 ----------------
    pdf.h1("Task 1 - Feature representation")
    pdf.p("Requirements, in priority order. (a) Interpretability: w is shown back to the "
          "user in plain language and they can override any component, so every dimension "
          "must be something a person could name as a taste. (b) Low dimension, argued "
          "quantitatively below. (c) Available for every film without any ratings. "
          "(d) Comparable scale, so the components of w can be read against each other.")

    pdf.h2("How large can the representation be?")
    pdf.p("Only utility differences within a choice set enter the likelihood, so what a "
          "few dozen comparisons must pin down is a direction in R^d. For random labels "
          "on n difference vectors in general position, Cover's counting theorem gives "
          "the probability that they are linearly separable:")
    pdf.formula("P(separable) = 2^-(n-1) * SUM_{k<d} C(n-1, k)")
    pdf.p("Where this probability approaches 1 the responses do not determine a "
          "direction at all: the unpenalised maximum-likelihood estimate runs off to "
          "infinity, and what a regularised fit returns in the unidentified subspace is "
          "the prior rather than the participant. At d = 34 and 50 comparisons that "
          "probability is 0.995; at d = %d it is 0.13. This bounds how far the "
          "representation could sensibly be pushed." % features.dimension())
    pdf.p("The estimator used here is a MAP fit under a Gaussian prior, which stays "
          "finite and unique under separation by construction: separability does not "
          "break it, it means some directions of w are reported at their prior value. "
          "This is an argument against representations of the size that was actually "
          "rejected: keyword TF-IDF and cast one-hots, thousands of columns.")
    if m:
        s = m["separability"]
        pdf.table(["d"] + [f"n = {n}" for n in s["ns"]],
                  [[str(d)] + [f"{v:.3f}" for v in s["table"][str(d)]]
                   for d in sorted(int(k) for k in s["table"])],
                  [20] + [26] * len(s["ns"]))
    pdf.figure("separability", "Figure 1. Probability that the collected comparisons are "
                               "linearly separable, by feature dimension. Above the dotted "
                               "line the estimate is dominated by the prior.")

    pdf.h2("The chosen features")
    pdf.table(["Feature", "Dims", "Source column", "Rationale"],
              [[a, b, c, d] for a, b, c, d in FEATURE_TABLE], [26, 11, 30, 111])
    pdf.p("Total: %d dimensions over %s films. The extractor is "
          "project4/features.py:build_matrix(), which cleans, encodes, winsorises and "
          "standardises in one pass, and is cached per process."
          % (features.dimension(), f"{m['assumptions']['n_movies']:,}" if m else "~4,800"))

    pdf.h2("Two identifiability rules, enforced rather than assumed")
    pdf.bullets([
        "Genres stay raw 0/1 and are never row-normalised. Normalising rows so that the "
        "genre block sums to 1 makes the all-ones genre direction constant across all "
        "films. Since only differences within a choice set enter the likelihood, that "
        "direction never appears in the data; yet its posterior variance is maximal, so "
        "an adaptive question-selection rule would preferentially chase a direction that "
        "is provably unlearnable. build_matrix() asserts the centred matrix is full rank.",
        "The content-rating one-hot drops a reference level (adult), for exactly the same "
        "reason.",
        "Rare binary flags are centred but not divided by their standard deviation. "
        "Scaling a 4%-prevalence flag by an sd near 0.2 turns a handful of films into "
        "high-leverage points that dominate the fit.",
    ])

    pdf.h2("Rejected representations")
    pdf.table(["Candidate", "Why not"], [[a, b] for a, b in REJECTED], [40, 138])

    pdf.h2("Confirming the choice")
    if m and "dimension" in m:
        dim = m["dimension"]
        gap = dim["wide"]["ranking"] - dim["narrow"]["ranking"]
        pdf.p("A wider representation (+ %d rare genres, d = %d) was fitted on the same "
              "synthetic responses and scored against held-out films: held-out rho "
              "changes by only %.3f, confirming that the chosen, smaller representation "
              "is not leaving predictive power on the table. It is kept small for the "
              "interpretability reason above, not because a larger one would perform "
              "worse."
              % (len(dim["extra_features"]), dim["wide"]["d"], gap))

    # ---------------- Task 2 ----------------
    pdf.maybe_break()
    pdf.h1("Task 2 - From Bradley-Terry to rankings")
    pdf.p("Bradley-Terry models a single pairwise comparison: the probability that film i "
          "beats film j is the logistic function of their utility difference.")
    pdf.formula("P(i > j | w) = exp(w'x_i) / (exp(w'x_i) + exp(w'x_j))\n"
                "            = sigmoid( w'(x_i - x_j) )")
    pdf.p("Design 2 collects a full ordering of ten films, so the model must assign a "
          "probability to an entire permutation. The proposed extension is the "
          "Plackett-Luce model:")
    pdf.formula("P(i_1 > i_2 > ... > i_n | w) = PROD_{k=1..n-1}  exp(w'x_{i_k})\n"
                "                                              ----------------------\n"
                "                                              SUM_{l=k..n} exp(w'x_{i_l})")
    pdf.p("Read generatively, this is exactly 'pick the best of what is left, then repeat': "
          "the first factor is a softmax choice over all ten films, the second a softmax "
          "over the nine that remain, and so on. The last factor is 1, so the product runs "
          "to n-1.")

    pdf.h2("Justification")
    pdf.bullets([
        "It reduces exactly to Bradley-Terry at n = 2, so a pairwise choice and a ranking "
        "are observations of the same model. This matters for the study: both designs are "
        "fitted with one estimator, so the headline comparison cannot be confounded by two "
        "different likelihoods or optimisers.",
        "It is the unique model consistent with Luce's choice axiom (independence of "
        "irrelevant alternatives): the relative odds of preferring i over j do not depend "
        "on which other films are in the set.",
        "It is the random-utility model obtained by adding i.i.d. Gumbel noise to each "
        "film's utility and taking the induced ordering (McFadden). Note this is Gumbel, "
        "not Thurstonian: Thurstone's Case V uses Gaussian noise and yields no closed form "
        "for n > 2, which is a further practical argument for Plackett-Luce.",
        "Its log-likelihood is concave in w, and strictly concave once a Gaussian prior is "
        "added, so the MAP estimate is unique and computable by Newton's method in well "
        "under ten iterations (fast enough to refit inside a web request).",
        "Truncating the product after K factors gives the top-K partial-ranking "
        "likelihood. This is not a footnote: a ranking task interrupted by the block's "
        "time limit yields a partial order, and it is used rather than discarded.",
    ])

    pdf.h2("Honest limitation: IIA")
    pdf.p("The axiom that makes Plackett-Luce clean is also its main weakness. If a set "
          "contains two near-identical films (two entries in the same superhero franchise), "
          "they are substitutes, and a real person's choice probabilities violate "
          "independence of irrelevant alternatives. Plackett-Luce cannot represent that. "
          "A mixed-logit or nested model would, at the cost of far more parameters than "
          "50 comparisons can support. The trade-off is made deliberately in favour of "
          "identifiability.")

    pdf.h2("Rejected alternatives")
    pdf.bullets([
        "Decomposing each ranking into all C(10,2) = 45 pairwise Bradley-Terry terms. This "
        "is not a normalised likelihood over permutations, and it treats 45 strongly "
        "dependent comparisons as independent evidence, overstating the information in a "
        "ranking by roughly a factor of five.",
        "Mallows' model, which is defined by distance to a central permutation and so "
        "cannot use features: it could not generalise to a film the user has not seen.",
    ])

    pdf.h2("Estimation")
    pdf.p("w is estimated by MAP under a Gaussian prior: maximise the sum of ranking "
          "log-likelihoods minus ||w||^2 / 2*sigma^2. The gradient and Hessian are "
          "available in closed form, and the Hessian is bounded below by I/sigma^2, so "
          "damped Newton converges quickly and without line-search heroics. Inverting the "
          "same Hessian gives the Laplace posterior covariance, which is used twice over: "
          "for the uncertainty attached to each learned weight, and for the adaptive "
          "question-selection rule described at the end of this report. "
          "sigma = %s is pre-registered and identical in both conditions."
          % (m["assumptions"]["sigma"] if m else "1.0"))

    # ---------------- Task 3 ----------------
    pdf.maybe_break()
    pdf.h1("Task 3 - Design of the user study")
    pdf.p("The study compares two interfaces for eliciting w. Design 1 presents two films "
          "and asks which the participant would rather watch. Design 2 presents ten films "
          "and asks for a full ranking. This study was not run; what follows is the "
          "protocol that the implemented interface would execute.")

    pdf.h2("Research question and hypotheses")
    pdf.p("Which interface recovers more of a user's preferences per unit of their time, "
          "and which do users prefer to use? Answering both matters: an interface that "
          "extracts marginally more signal but that people find tedious will not survive "
          "contact with a real product.")
    pdf.bullets([
        "H1 (primary, two-sided): the two designs differ in how well the fitted w predicts "
        "held-out preferences, at an equal elicitation time budget.",
        "H2: they differ in held-out pairwise accuracy (a second, coarser operationalisation).",
        "H3: ranking imposes higher perceived workload (Raw NASA-TLX).",
        "H4: the designs differ in how participants judge them, on both halves of that "
        "judgement: the process (a forced choice between the two interfaces) and the "
        "output (a blind rating of the two recommendation lists, one fitted from each "
        "block, presented unlabelled).",
        "H5 (exploratory): later positions in a ranking are noisier than early ones, "
        "tested by refitting each ranking as a top-3 partial ranking and comparing "
        "predictive performance against the full-ranking fit. Only the top-K truncation "
        "is used, because that is the likelihood the model actually supports; a "
        "bottom-K constraint would need a different, unimplemented likelihood.",
    ])

    pdf.h2("Design: within-subjects, counterbalanced")
    pdf.p("Every participant uses both interfaces, in an order randomised across "
          "participants (AB / BA). Taste in film varies enormously between people; that "
          "between-person variance is the dominant noise term, and a within-subjects "
          "design removes it entirely by comparing each participant against themselves. "
          "The cost is carryover and fatigue, mitigated by counterbalancing, by drawing "
          "each block's films from disjoint pools, and by a short break between blocks.")

    pdf.h2("The measurement problem, and how it is solved")
    pdf.p("There is no ground-truth w to compare an estimate against. The protocol "
          "therefore ends with a validation block: 16 held-out films, none of them seen "
          "during either elicitation block, rated on a 7-point 'how much would you want to "
          "watch this' scale. The w fitted from block A and the w fitted from block B are "
          "each scored against those same ratings. The primary dependent variable is "
          "Spearman's rho between predicted utility and stated rating.")
    pdf.p("Three details make this fair. First, the validation format is neutral: it is "
          "neither a pairwise choice nor a ranking. A pairwise validation block would "
          "structurally favour Design 1, whose model is trained and tested in the same "
          "format and so also learns any format-specific response bias, a confound "
          "plausibly larger than the effect being measured. Second, the elicitation blocks "
          "are time-boxed at five minutes rather than fixed at a number of tasks, because "
          "the question is about information per unit of the participant's time. Note "
          "that once time is fixed by construction, the dependent variable is simply the "
          "held-out score; dividing by a constant duration would add nothing.")
    pdf.p("Third, the time box is enforced strictly, which is less obvious than it "
          "sounds. The naive implementation checks the clock only before handing out the "
          "next task, so a ten-film ranking served at t = 299 s runs to completion and "
          "the ranking block quietly receives up to forty seconds more elicitation time "
          "than the pairwise block at the same nominal budget, a bias of the same order "
          "as the effect being measured, pointing in the direction of the hypothesis. The "
          "interface therefore cuts the block where the clock says, and keeps whatever "
          "ordering the participant had established as a top-K partial ranking. Nothing "
          "the participant did is discarded, and neither condition is given extra time.")
    pdf.p("The 16 films are split over two pages of ten, with four of the first page's "
          "films repeated, unannounced, on the second: twenty ratings in total. This "
          "costs about ninety seconds and buys three things: an attention check, an "
          "estimate of each participant's own self-consistency (the ceiling any model "
          "could reach for them) and a measure of position bias. Reported effects are "
          "interpreted against that ceiling, not against 1.0.")
    pdf.p("Two baselines are reported alongside: w = 0 (chance), and a leave-one-"
          "participant-out population-average w. If neither design beats the population "
          "average, personalisation is not doing any work and the comparison is moot.")

    pdf.h2("What the simulations say to expect")
    if m:
        r = m["recovery"]
        a = m["assumptions"]
        i5 = r["budgets"].index(300) if 300 in r["budgets"] else -1
        pdf.p("Before recruiting anyone, the protocol was simulated with synthetic users "
              "whose true preference vectors are known, answering according to the "
              "Plackett-Luce model and drawing films from the same familiarity pool the "
              "study uses. Timing is assumed at %.0f s per pairwise click and %.0f s for "
              "a full ten-film ranking. Each simulated participant completes BOTH designs, "
              "exactly as in the real within-subjects protocol, so the difference between "
              "them is a paired quantity."
              % (a["sec_per_pair"], a["sec_per_rank_set"]))
        pdf.table(["Budget", "Pairwise tasks", "Rank tasks", "Design 1 rho", "Design 2 rho",
                   "Paired diff", "SD of diff", "d_z"],
                  [[f"{b // 60} min", str(r["n_pairs"][i]), str(r["n_sets"][i]),
                    f"{r['pairwise'][i]:.3f}", f"{r['ranking'][i]:.3f}",
                    f"{r['diff_mean'][i]:+.3f}", f"{r['diff_sd'][i]:.3f}",
                    f"{r['dz'][i]:+.2f}"]
                   for i, b in enumerate(r["budgets"])],
                  [20, 27, 21, 22, 22, 22, 22, 22])
        naive = abs(r["diff_mean"][i5]) / r["ranking_sd"][i5]
        pdf.p("The right-hand columns are the ones that matter. The effect size "
              "is the mean paired difference over the standard deviation OF THAT "
              "DIFFERENCE (d_z), never over the between-participant standard deviation of "
              "either condition. Which way the mistake cuts is not fixed: a paired design "
              "helps only to the extent that the two conditions are correlated across "
              "participants, and here they are not correlated enough for the pairing to "
              "pay off. At five minutes the conditions have SDs of %.3f and %.3f while "
              "their paired difference has an SD of %.3f, larger than either. Dividing "
              "by a condition SD would give d_z = %.2f and N = %d, understating the "
              "sample needed by a third."
              % (r["pairwise_sd"][i5], r["ranking_sd"][i5], r["diff_sd"][i5],
                 naive, _n_for_dz(naive)))
        pdf.p("This is a property of this study rather than an accident of the "
              "simulation. Each block draws its own random films, so two blocks by the "
              "same participant "
              "are two independent noisy measurements of the same taste vector; the "
              "shared term they have in common is the participant's w, which the "
              "held-out score is largely insensitive to. Within-subjects is still the "
              "right design (it controls order, fatigue and the participant's own "
              "consistency ceiling) but it should not be assumed to deliver a variance "
              "reduction it does not, and the sample size is calculated accordingly.")
        pdf.p("At the five-minute budget the simulated advantage of ranking is %+.3f in "
              "rho, giving d_z = %.2f. The advantage is stable in size across budgets "
              "from two minutes upward, but d_z grows with the budget because the "
              "difference gets less noisy, not larger."
              % (r["diff_mean"][i5], r["dz"][i5]))
    pdf.figure("recovery", "Figure 2. Simulated recovery of a known preference vector under "
                          "each design at matched time budgets. Bands are +/- 1 SD across "
                          "synthetic participants.")

    pdf.h2("The assumption the whole comparison rests on")
    if m and "sensitivity" in m:
        sens = m["sensitivity"]
        pdf.p("Nothing in the dataset says how long a person takes to rank ten films. "
              "That single constant fixes how many ranking tasks fit in the budget, and "
              "the comparison turns on it, so it is varied rather than asserted. The "
              "pairwise cost is held at %.0f s throughout and the budget at %d minutes."
              % (m["assumptions"]["sec_per_pair"], sens["budget"] // 60))
        pdf.table(["Seconds per ranking task", "Tasks in budget", "Design 1 rho",
                   "Design 2 rho", "Paired diff", "d_z", "N for 80% power"],
                  [[f"{row['sec_per_rank_set']:.0f} s", str(row["n_sets"]),
                    f"{row['pairwise']:.3f}", f"{row['ranking']:.3f}",
                    f"{row['diff_mean']:+.3f}", f"{row['dz']:+.2f}",
                    str(row["n_required"]) if row["n_required"] else "no effect"]
                   for row in sens["rows"]],
                  [42, 26, 22, 22, 22, 16, 28])
        flip = [row for row in sens["rows"] if row["dz"] < 0]
        pdf.p("The predicted effect does not merely shrink as ranking "
              "gets more expensive: it CHANGES SIGN. If a ten-film ranking costs a "
              "person about a minute, the two designs are indistinguishable; if it costs "
              "ninety seconds, pairwise choice wins by a margin comparable to the one "
              "ranking wins by at thirty seconds. The honest conclusion is that this "
              "study cannot be powered, or even given a directional hypothesis, until "
              "that constant is measured. That is precisely what the pilot is for, and "
              "it is why H1 is stated as two-sided.")
        if flip:
            pdf.p("Pre-registration therefore commits to the SESOI and the analysis, and "
                  "fixes the final sample size only after the pilot has measured the "
                  "per-task costs, with the pilot data excluded from the main analysis, "
                  "so this does not become an optional-stopping problem.")

    pdf.h2("Sample size")
    if m:
        p_ = m["power"]
        r = m["recovery"]
        i5 = r["budgets"].index(300) if 300 in r["budgets"] else -1
        pdf.p("Two numbers bracket the answer. The first is a judgement: a difference "
              "smaller than d_z = 0.4 in held-out rho would not change which interface a "
              "product should ship, so d_z = 0.4 is adopted as the smallest effect of "
              "interest. The second is the simulation's own prediction under the assumed "
              "timings, d_z = %.2f. Powering for the smaller of the two is the "
              "conservative choice."
              % r["dz"][i5])
        pdf.table(["Smallest effect (d_z)"] + [str(e) for e in p_["effects"]],
                  [["Participants required"] + [str(n) for n in p_["n"]]],
                  [46] + [22] * len(p_["effects"]))
        pdf.p("Paired test, alpha = %.2f, power = %.2f, normal approximation. d_z = 0.4 "
              "requires N = %d; the simulated d_z = %.2f requires N = %d. The study "
              "recruits for the larger of the two and inflates for the exclusion criteria "
              "below: N = 70."
              % (p_["alpha"], p_["power"], p_["n"][p_["effects"].index(0.4)],
                 r["dz"][i5], r["n_required"][i5]))
    pdf.figure("power", "Figure 3. Participants required as a function of the smallest "
                        "effect worth detecting.")

    pdf.h2("Participants and recruitment")
    pdf.bullets([
        "Recruited through Prolific, screened for fluent English and watching at least one "
        "film a month; the task assumes some familiarity with mainstream cinema.",
        "Compensated at the platform's fair-pay rate for the full expected duration "
        "(about 15 minutes), independent of how they perform.",
        "A pilot of 5 participants runs first. Its primary job is not instruction "
        "wording but MEASURING the per-task costs, because the sensitivity table above "
        "shows the predicted effect changing sign across a plausible range of them. The "
        "final sample size is fixed from the pilot's measured timings before main "
        "recruitment opens; pilot data are not pooled into the main analysis.",
    ])

    pdf.h2("Procedure")
    pdf.p("The interface implements exactly this sequence, one screen per step: consent; "
          "a short demographic questionnaire; instructions for the first interface; one "
          "unrecorded practice task in that format; the five-minute elicitation block; "
          "the Raw NASA-TLX for that block; a break; instructions, practice, block and "
          "RTLX for the second interface; the validation ratings over two pages; the "
          "comparative questionnaire; a blind judgement of two recommendation lists; and "
          "finally the reveal and debrief, where the learned taste profile is shown in "
          "plain language, can be overridden, and the participant can download "
          "everything they produced.")
    pdf.bullets([
        "The practice task is the real task, run once with recording switched off, and "
        "its films are removed from the pool so they cannot reappear. Without it the "
        "first minute of each block measures learning the interface rather than "
        "preference, and in a within-subjects design that cost lands entirely on "
        "whichever interface the participant met first.",
        "Each RTLX comes IMMEDIATELY after its own block, not at the end. Workload is "
        "retrospective self-report: asking about block A after block B and two pages of "
        "ratings have intervened measures recall and contrast, not workload, and the "
        "damage would fall asymmetrically on the first block.",
        "The two recommendation lists are shown UNLABELLED, before the reveal discloses "
        "which came from which interface. A participant who knows which list came from "
        "the ranking task cannot rate it independently of how they felt about ranking. "
        "The process half of H4 is asked on the previous screen, before either list is "
        "seen, for the same reason.",
    ])
    pdf.p("Workload is measured with Raw NASA-TLX rather than the full instrument. The "
          "standard weighting procedure requires fifteen pairwise comparisons per "
          "condition, which would be a serious burden on top of the study's own tasks. "
          "RTLX drops the weighting, not the instrument: all six subscales are "
          "administered unweighted, "
          "since its agreement with the weighted score is a property of the complete "
          "six-item set.")

    pdf.h2("Sampling frame")
    pdf.p("Films shown to participants are drawn uniformly at random, as the brief "
          "specifies, but from a documented pool rather than the raw catalogue: films with "
          "at least 10,000 IMDB votes, about 3,500 of the roughly 4,800 usable titles. "
          "Uniform sampling over everything surfaces films almost nobody has heard of, at "
          "which point 'which would you rather watch' degenerates into 'which title sounds "
          "more appealing' and both interfaces measure the same noise. Restricting the "
          "frame is an ecological-validity control and is reported as such.")
    pdf.p("Recommendations at the end are drawn from a DIFFERENT and wider frame (at "
          "least 1,000 votes, about 4,450 titles) because a recommender that can only "
          "propose films the participant was already shown is not recommending anything. "
          "It is not the whole catalogue either, and the reason is worth recording. "
          "Utility here is linear, so maximising it over an unfiltered catalogue lands on "
          "whichever corner of the feature space is most extreme; since popularity is "
          "itself one of the features, any participant with a taste for the niche is "
          "handed films with single-digit vote counts. The threshold removes the "
          "343-film tail where this happens and "
          "still leaves roughly 950 titles the study never showed them. The top-5 is "
          "additionally de-duplicated by primary genre, for the same reason: an "
          "unconstrained argmax over a fixed catalogue returns five near-identical films.")

    pdf.h2("Exclusion criteria, fixed in advance")
    pdf.bullets([
        "Disagreement of 3 or more scale points on the repeated validation items, "
        "indicating inattentive responding.",
        "Median per-decision response time below 1.5 seconds: faster than the films can "
        "be read. Every decision is timed in both designs, including each individual pick "
        "inside a ranking task, so the rule applies equally to both blocks rather than "
        "only to the one that happens to record latencies.",
        "Submitting a ranking in exactly the order presented, in every ranking task "
        "(straight-lining).",
        "Not completing both blocks.",
    ])

    pdf.h2("Analysis plan")
    pdf.bullets([
        "Primary: paired t-test on the within-participant difference in Spearman rho "
        "between conditions, with Wilcoxon signed-rank as the pre-specified fallback if "
        "the differences are visibly non-normal.",
        "A mixed-effects model on item-level validation responses, with condition as a "
        "fixed effect and crossed random intercepts for participant and validation item, "
        "is reported alongside; it uses more of the data and is more powerful, but is more "
        "assumption-laden, so the simple paired test remains primary.",
        "Order (AB vs BA) is included as a covariate to check for carryover.",
        "H4 is tested as two pre-specified components: a binomial test on the "
        "forced-choice preference between the interfaces, and a paired test on the blind "
        "ratings of the two recommendation lists. They can disagree (a participant may "
        "prefer the interface that produced the worse list) and that disagreement is "
        "itself the interesting result.",
        "Holm correction across the secondary hypothesis family (H2-H4).",
        "Sensitivity analysis over the prior scale sigma in {0.5, 1, 2}, since the two "
        "designs yield different numbers of effective observations and therefore different "
        "amounts of shrinkage.",
        "Pre-registered on OSF before any data collection, including the exclusion rules, "
        "the primary DV, and this analysis plan.",
    ])

    pdf.h2("Ethics and data protection")
    pdf.bullets([
        "Approval from the TUHH ethics board before recruitment.",
        "Informed consent on the first screen, with the right to withdraw at any point "
        "without giving a reason and without loss of compensation.",
        "No personal data is collected: no name, no email, no IP address. Participants are "
        "identified only by the platform's pseudonymous ID, held separately from responses.",
        "GDPR lawful basis is consent; data are retained for five years in line with good "
        "scientific practice, then deleted. Anonymised responses are published with the paper.",
        "Film metadata only: no content that could plausibly distress a participant.",
    ])

    pdf.h2("Threats to validity")
    pdf.bullets([
        "Carryover and fatigue between blocks; addressed by counterbalancing, disjoint film "
        "pools, and a break. Order is checked as a covariate.",
        "Participants may not have seen the films. The task deliberately asks which they "
        "would rather *watch*, a forward-looking judgement that needs no prior viewing.",
        "Uniform random draws produce many trivially easy pairs, wasting participant "
        "effort; this is the motivation for the adaptive extension below.",
        "The simulations validate the estimator, not the human. Real participants are "
        "noisier and less internally consistent than the Plackett-Luce model assumes, so "
        "the simulated effect sizes should be read as an upper bound.",
        "Design 2 is implemented as repeated best-choice rather than drag-to-rank. This "
        "matches the Plackett-Luce generative process exactly and works without JavaScript "
        "or a mouse, but it is not cognitively identical to arranging ten items at once.",
        "That implementation costs one server round trip per pick, so a ten-film ranking "
        "is ten page loads against a single page load for a pairwise choice, and the "
        "primary DV is scored against block wall-clock time, which includes them. On a "
        "slow connection a measurable share of Design 2's budget would be spent waiting "
        "rather than deciding, and a result could then reflect the deployment rather than "
        "the interaction. Server-side processing time is logged per request so it can be "
        "subtracted, and the pilot checks whether it is large enough to matter; a "
        "single-page client-side implementation would remove the confound at the cost of "
        "requiring JavaScript.",
        "The simulated timings are assumptions, not measurements, and the sensitivity "
        "table above shows the predicted effect reversing across a plausible range of "
        "them. This is the largest single threat to the study's conclusion and the main "
        "thing the pilot exists to resolve.",
    ])

    pdf.h2("Extension - choosing more informative questions")
    pdf.p("The study asks questions uniformly at random, which is what keeps the two "
          "interfaces comparable. A deployed recommender would not. Given the Laplace "
          "posterior covariance S, a natural score for a candidate pair with difference "
          "vector d = x_a - x_b is:")
    pdf.formula("score(a, b) = log( 1 + p(1-p) * d' S d ),   p = sigmoid(w'd)")
    pdf.p("Both factors are necessary. The naive D-optimal choice (maximise d'Sd alone) "
          "systematically selects the pairs the user finds easiest, because a large "
          "predicted utility gap means a near-certain answer that carries almost no "
          "information. Weighting by the response entropy p(1-p) selects questions that are "
          "both uncertain under the current posterior and close to a coin flip for this "
          "particular user. Candidate pairs are subsampled, since the catalogue admits "
          "roughly eleven million of them. The project page includes a live demonstration; "
          "in simulation this reaches a given accuracy in substantially fewer questions "
          "than random selection.")

    return bytes(pdf.output())


@lru_cache(maxsize=1)
def build():
    """The report as immutable PDF bytes, built once per process."""
    return _build()
