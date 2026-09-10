"""Task 4: the participant protocol.

The protocol is linear: a tuple of steps and an index. No state machine.

Everything a participant produces is held in the Django session. The encoding is
deliberately terse: records are lists of integers, never dicts of titles, and the
"already seen" exclusion set is derived from the response log rather than stored.
"""

import random
import time

SESSION_KEY = "p4"

PAIRWISE, RANKING = "pairwise", "ranking"
RANK_SET_SIZE = 10

# The elicitation budget per block. Blocks are time-boxed rather than
# task-boxed: that is what makes "how much preference information does this
# interface extract per unit of the participant's time" a fair question.
#
# The box is enforced strictly. A rank set handed out at t = 299 s must not be
# allowed to run to completion, or Design 2 quietly receives up to a further
# ~40 s of elicitation time, a first-order confound on the primary dependent
# variable, which is held-out agreement at a fixed time budget. Instead the
# block is cut where the clock says, and whatever ordering the participant had
# established by then is kept as a top-K partial ranking (see preference.py).
BLOCK_SECONDS = 300

# Validation: 16 distinct held-out movies rated on a 7-point want-to-watch
# scale, split over two pages, with 4 items from page 1 repeated on page 2 to
# give a per-participant self-consistency ceiling and an attention check.
VALIDATION_UNIQUE = 16
VALIDATION_REPEATS = 4
VALIDATION_PAGE = 10

# One unrecorded practice task in each format, immediately before that format's
# timed block. Without it the first minute of block A measures learning the
# interface rather than preference, and in a within-subjects design that lands
# entirely on whichever design was assigned first.
STEPS = (
    "consent", "demographics",
    "intro_a", "practice_a", "block_a", "rtlx_a",
    "break",
    "intro_b", "practice_b", "block_b", "rtlx_b",
    "validation_1", "validation_2",
    "final", "recs",
    "reveal", "debrief",
)

# The one field in the protocol with no natural bound.
MAX_FREE_TEXT = 300

LIKERT_7 = ["1 - not at all", "2", "3", "4", "5", "6", "7 - very much"]
RTLX_SCALE = ["1 - very low", "2", "3", "4", "5", "6", "7 - very high"]

# Raw NASA-TLX (RTLX): all six subscales, unweighted. The full TLX weighting
# procedure is 15 pairwise comparisons per condition, which is a serious burden
# on participants, particularly in a study about the burden of pairwise
# comparison. Dropping a subscale instead would be worse: RTLX's agreement with
# the weighted score is a property of the complete six-item set.
RTLX_ITEMS = [
    ("mental", "Mental demand: how mentally demanding was the task?"),
    ("physical", "Physical demand: how physically demanding was the task?"),
    ("pace", "Temporal demand: how hurried or rushed did you feel?"),
    ("success", "Performance: how successful were you at expressing your taste?"),
    ("effort", "Effort: how hard did you have to work?"),
    ("frustration", "Frustration: how irritated or stressed did you feel?"),
]

QUESTIONS = {
    "demographics": [
        ("age", "Your age group", ["18-24", "25-34", "35-44", "45-54", "55+", "Prefer not to say"]),
        ("films", "How many films do you watch in a typical month?", ["0", "1-2", "3-5", "6-10", "More than 10"]),
        ("recsys", "How often do you use a recommender (Netflix, Spotify, ...)?",
         ["Never", "Rarely", "Sometimes", "Often", "Daily"]),
    ],
    "final": [
        ("preferred", "Which way of expressing your preferences did you prefer overall?",
         ["Choosing between two films", "Ranking ten films", "No preference"]),
        ("expressive", "Which felt better at capturing your actual taste?",
         ["Choosing between two films", "Ranking ten films", "No preference"]),
        ("tedious", "Which felt more tedious?",
         ["Choosing between two films", "Ranking ten films", "About the same"]),
        ("comments", "Anything else you would like to tell us? (optional)", None),
    ],
    # Asked on the `recs` step, against two unlabelled recommendation lists,
    # one fitted from each block. This is the half of H4 that is about the
    # output rather than the process, and it has to be blind: a participant who
    # knows which list came from the ranking task cannot rate it independently
    # of how they felt about ranking.
    "recs": [
        ("better", "Which list better matches your taste?",
         ["List 1", "List 2", "They are about the same"]),
        ("rate_1", "How appealing is List 1 overall?", LIKERT_7),
        ("rate_2", "How appealing is List 2 overall?", LIKERT_7),
    ],
}

# Every closed question in the protocol, keyed by field name, so a submitted
# value can be checked against the options actually offered rather than stored
# verbatim. Free-text fields (choices None) are truncated instead.
def _choice_index():
    allowed = {}
    for group in QUESTIONS.values():
        for key, _q, choices in group:
            allowed[key] = set(choices) if choices else None
    for key, _q in RTLX_ITEMS:
        allowed[key] = set(RTLX_SCALE)
    return allowed


ALLOWED_ANSWERS = _choice_index()


def new_state(seed=None):
    """A fresh participant record. `o` is the counterbalancing assignment."""
    rng = random.Random(seed)
    return {
        "s": 0,                       # step index
        "o": rng.randint(0, 1),       # 0 = pairwise block first, 1 = ranking first
        "sd": rng.randrange(1 << 30),  # seed for this participant's movie draws
        "g": {},                      # demographics
        "p": [],                      # pairwise: [a, b, chosen, ds]
        "r": [],                      # rankings: [n_ranked] + order(10) + pick ds
        "v": [],                      # validation: [movie_index, rating]
        "vlen": {},                   # len(v) before each validation page's last submit,
                                       # so resubmitting (via Back) replaces it instead of
                                       # stacking a second copy of the same items on top
        "pr": [],                     # movies burned on practice tasks
        "sk": [],                     # movies burned on skipped pairwise tasks
        "q": {},                      # questionnaires
        "t": {},                      # block wall-clock seconds
        "bs": None,                   # current block start (epoch seconds)
        "cur": None,                  # movies currently on screen
        "ct": None,                   # when the current decision was offered
        "picked": [],                 # ranking in progress, best first
        "pd": [],                     # deciseconds taken over each of those picks
        "w": None,                    # user-overridden weights on the reveal page
    }


def step(state):
    return STEPS[min(state["s"], len(STEPS) - 1)]


def advance(state):
    state["s"] = min(state["s"] + 1, len(STEPS) - 1)
    state["bs"] = None
    state["cur"] = None
    state["ct"] = None
    state["picked"] = []
    state["pd"] = []
    return state


# Steps a participant may step back from. block_*, practice_* and reveal are
# excluded: the blocks are time-boxed, and reveal is excluded specifically so
# recs' blind list comparison can't be redone after already having seen, on
# reveal, which list came from which design. That would un-blind it after
# the fact. rtlx_*/validation_* aren't in the list either (reached only as a
# Back *target*, from break/final), but resubmitting them is made idempotent
# below rather than blocked, so that doesn't limit which steps get a button.
SAFE_STEPS = frozenset({
    "consent", "demographics", "intro_a", "break", "intro_b",
    "final", "recs", "debrief",
})


def can_go_back(state):
    i = state["s"]
    return i > 0 and STEPS[i] in SAFE_STEPS


def go_back(state):
    state["s"] = max(0, state["s"] - 1)
    state["bs"] = None
    state["cur"] = None
    state["ct"] = None
    state["picked"] = []
    state["pd"] = []
    return state


def design_for(state, slot):
    """Which elicitation design runs in block A or block B for this participant."""
    first = PAIRWISE if state["o"] == 0 else RANKING
    second = RANKING if state["o"] == 0 else PAIRWISE
    return first if slot == "a" else second


def slot_of(st):
    """Which block a step belongs to, or None outside the elicitation phases."""
    if st in ("intro_a", "practice_a", "block_a", "rtlx_a"):
        return "a"
    if st in ("intro_b", "practice_b", "block_b", "rtlx_b"):
        return "b"
    return None


def current_design(state):
    slot = slot_of(step(state))
    return design_for(state, slot) if slot else None


# --------------------------------------------------------------------------
# Ranking record encoding: [n_ranked] + order(RANK_SET_SIZE) + pick deciseconds
# --------------------------------------------------------------------------

def rank_order(rec):
    return rec[1:1 + RANK_SET_SIZE]


def rank_times(rec):
    return rec[1 + RANK_SET_SIZE:]


def seen(state, include_validation=True):
    """Every catalogue index the participant has already been shown.

    `include_validation=False` gives everything except the validation films.
    The validation draw needs that: it must return the same 16 films on page 2
    as on page 1, and if the films already rated on page 1 were excluded, the
    second page would silently draw a fresh set and the repeated-item
    consistency check would never fire.
    """
    out = set(state.get("pr") or [])
    out.update(state.get("sk") or [])
    for a, b, _c, _ds in state["p"]:
        out.add(a); out.add(b)
    for rec in state["r"]:
        out.update(rank_order(rec))
    out.update(state.get("picked") or [])
    out.update(state.get("cur") or [])
    if include_validation:
        for idx, _rating in state["v"]:
            out.add(idx)
    return out


def _rng(state, salt):
    return random.Random(f"{state['sd']}:{salt}")


def draw(state, k, salt, include_validation=True):
    """k distinct unseen movies drawn uniformly from the study pool."""
    from . import data
    pool = [int(i) for i in data.study_pool_index()]
    used = seen(state, include_validation=include_validation)
    available = [i for i in pool if i not in used]
    return _rng(state, salt).sample(available, k)


def start_block(state):
    if state["bs"] is None:
        state["bs"] = time.time()
    return state


def remaining(state):
    if state["bs"] is None:
        return BLOCK_SECONDS
    return max(0, int(BLOCK_SECONDS - (time.time() - state["bs"])))


def finish_block(state, slot):
    """Close a block at the clock, keeping any ranking that was in progress."""
    # `cur` is what makes the record a full choice set; without it the ordering
    # cannot be stored in the fixed-width encoding rank_order() decodes.
    if state.get("picked") and state.get("cur"):
        record_ranking(state, list(state["picked"]), list(state["pd"]),
                       n_ranked=len(state["picked"]))
    elapsed = time.time() - state["bs"] if state["bs"] else 0.0
    state["t"][slot] = round(min(elapsed, float(BLOCK_SECONDS)), 1)
    return advance(state)


def next_task(state, design, phase="block"):
    """The movies to display now, drawn once and cached in the session."""
    if state["cur"]:
        return state["cur"]
    salt = f"{phase}:{design}:{len(state['p'])}:{len(state['r'])}"
    k = 2 if design == PAIRWISE else RANK_SET_SIZE
    state["cur"] = draw(state, k, salt)
    state["ct"] = time.time()
    if phase == "practice":
        # Practice films are burned: a participant must not meet the same film
        # again in the timed block.
        state["pr"].extend(state["cur"])
    return state["cur"]


def clear_task(state):
    state["cur"] = state["ct"] = None
    state["picked"] = []
    state["pd"] = []
    return state


def skip_pair(state):
    """Decline the pair on screen without recording a choice.

    Forced choice between two unseen or disliked films is a real burden, so a
    participant can decline. The pair is burned, excluded from later draws, so
    declining doesn't just hand the same pair straight back.
    """
    state["sk"] = (state.get("sk") or []) + [int(i) for i in (state.get("cur") or [])]
    return clear_task(state)


def stop_ranking(state):
    """Record the order established so far for the current set as a partial
    ranking, then let a fresh set be drawn.

    Same top-K partial-ranking path the block-time cutoff uses in
    finish_block, offered here as something the participant can choose for
    themselves, per set, instead of only the clock enforcing it.
    """
    return record_ranking(state, list(state["picked"]), list(state["pd"]),
                          n_ranked=len(state["picked"]))


def record_pairwise(state, chosen):
    """Store [movie_a, movie_b, chosen, deciseconds].

    Deciseconds rather than milliseconds, and per-task rather than elapsed into
    the block, purely to keep the integers short. Timing at this resolution is
    descriptive only; the primary measure is block wall-clock.
    """
    a, b = state["cur"]
    state["p"].append([a, b, int(chosen), _task_ds(state)])
    clear_task(state)
    return state


def _task_ds(state):
    return int((time.time() - state["ct"]) * 10) if state.get("ct") else 0


def take_ds(state):
    """Deciseconds since the current decision was offered, then restart the clock.

    Each pick inside a rank set is one decision, so a ranking task yields a
    latency per position, which is what the pre-registered response-time
    exclusion and the middle-ranks-are-noisier hypothesis both need.
    """
    ds = _task_ds(state)
    state["ct"] = time.time()
    return ds


def record_ranking(state, picked, pick_ds, n_ranked=None):
    """Store a (possibly partial) ranking, best first.

    `picked` is the prefix the participant actually established; the films they
    never got to are appended in presentation order so the choice set is
    recoverable, and `n_ranked` says how many positions are real. That is
    exactly the top-K partial-ranking likelihood, so a block cut off by the
    clock contributes its evidence instead of being thrown away.
    """
    picked = [int(i) for i in picked]
    rest = [int(i) for i in (state.get("cur") or []) if int(i) not in picked]
    order = picked + rest
    assert len(order) == RANK_SET_SIZE, f"choice set was {len(order)} films"
    n = len(order) - 1 if n_ranked is None else min(int(n_ranked), len(order) - 1)
    state["r"].append([n] + order + [int(d) for d in pick_ds[:n]])
    clear_task(state)
    return state


def validation_items(state):
    """The 20 rating items: 16 unique, with 4 of page 1 repeated on page 2."""
    unique = draw(state, VALIDATION_UNIQUE, "validation", include_validation=False)
    repeats = _rng(state, "repeats").sample(unique[:VALIDATION_PAGE], VALIDATION_REPEATS)
    page1 = unique[:VALIDATION_PAGE]
    page2 = unique[VALIDATION_PAGE:] + repeats
    _rng(state, "shuffle").shuffle(page2)
    return page1, page2


def prior_validation_ratings(state, st):
    """This page's most recently submitted ratings, keyed by movie index.

    Only meaningful for a page reached via Back (from `final`): before that
    page's first submission, `vlen` has no entry for it and this is empty, so
    a first-time visit renders blank as before.
    """
    start = (state.get("vlen") or {}).get(st)
    if start is None:
        return {}
    return {idx: rating for idx, rating in state["v"][start:]}


def recs_first_slot(state):
    """Which block's recommendations are shown as 'List 1'.

    Derived from the participant's own seed rather than stored: the masking has
    to be stable across a refresh.
    """
    return "a" if _rng(state, "recmask").random() < 0.5 else "b"


def observations(state, design):
    """The elicitation records for one design, as Plackett-Luce observations."""
    from .preference import ranking
    if design == PAIRWISE:
        return [ranking([a, b] if c == 0 else [b, a]) for a, b, c, _ds in state["p"]]
    return [ranking(rank_order(rec), n_ranked=rec[0]) for rec in state["r"] if rec[0] > 0]


def fitted_weights(state):
    """The preference vector each block produced, named feature by feature."""
    from . import features, preference
    out = {}
    for slot in ("a", "b"):
        design = design_for(state, slot)
        w = preference.fit_map(observations(state, design))
        out[design] = {n: round(float(v), 4)
                       for n, v in zip(features.FEATURE_NAMES, w)}
    return out


def export(state):
    """The full response record, for the download at debrief.

    Everything the study would analyse, plus everything the system concluded,
    including the corrections the participant made to it on the reveal screen,
    which are the only record of where a person disagreed with the model.
    """
    return {
        "condition_order": "pairwise_first" if state["o"] == 0 else "ranking_first",
        "demographics": state["g"],
        "pairwise_responses": [
            {"movie_a": a, "movie_b": b, "chosen": (a if c == 0 else b),
             "seconds": round(ds / 10, 1)}
            for a, b, c, ds in state["p"]
        ],
        "ranking_responses": [
            {"n_ranked": rec[0], "order_best_first": rank_order(rec),
             "pick_seconds": [round(d / 10, 1) for d in rank_times(rec)]}
            for rec in state["r"]
        ],
        "validation_ratings": [{"movie": i, "rating": r} for i, r in state["v"]],
        "questionnaires": state["q"],
        "block_seconds": state["t"],
        "recommendation_lists_masked": {"list_1": recs_first_slot(state)},
        "fitted_weights": fitted_weights(state),
        "user_overrides": state.get("w") or {},
    }
