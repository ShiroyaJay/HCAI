"""Task 2 simulated experts.

Experts are deliberately imperfect, with competence tied to regions of the
input space. Each expert is deterministic: the randomness for an example is
derived from ``sha1(f"{seed}:{key}")`` of its stable row key, so repeated
queries for the same article always return the same answer — Tasks 3, 4 and
the tests all see one consistent expert.
"""

import hashlib

import numpy as np

from . import data


def _u(seed, key, salt=""):
    """Deterministic uniform in [0, 1) for one example."""
    digest = hashlib.sha1(f"{seed}:{key}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# Plausible confusion partner for each class when an expert errs:
# World<->Business (politics/economy overlap), Sports->World,
# Business<->Sci/Tech (tech-company news), Sci/Tech->Business.
_CONFUSION = {0: 2, 1: 0, 2: 3, 3: 2}


class ClassConditionalExpert:
    """Expert whose competence depends on the article's true topic.

    Correct with probability ``p_strong`` on the strong classes and ``p_weak``
    elsewhere. Wrong answers go to the plausible confusion class with
    probability 0.7, otherwise to a random other class.
    """

    def __init__(self, strong_classes=(2, 3), p_strong=0.97, p_weak=0.55, seed=0):
        self.strong_classes = set(strong_classes)
        self.p_strong = p_strong
        self.p_weak = p_weak
        self.seed = seed

    def describe(self):
        strong = ", ".join(data.CLASS_NAMES[k] for k in sorted(self.strong_classes))
        return (f"Class-conditional expert: {self.p_strong:.0%} correct on "
                f"[{strong}], {self.p_weak:.0%} on the other topics")

    def _p_correct(self, gold, text):
        return self.p_strong if gold in self.strong_classes else self.p_weak

    def predict(self, df):
        """Expert labels for a DataFrame with ``label``, ``key`` (and ``text``)."""
        out = np.empty(len(df), dtype=int)
        for i, (gold, key, text) in enumerate(
            zip(df["label"].values, df["key"].values, df["text"].values)
        ):
            if _u(self.seed, key, "correct") < self._p_correct(gold, text):
                out[i] = gold
            elif _u(self.seed, key, "confuse") < 0.7:
                out[i] = _CONFUSION[gold]
            else:
                others = [k for k in range(data.N_CLASSES) if k != gold]
                out[i] = others[int(_u(self.seed, key, "other") * len(others))]
        return out


class LengthConditionalExpert(ClassConditionalExpert):
    """Expert whose competence depends on article length, not topic.

    Reliable on long articles (more context to judge), weak on short ones.
    The threshold is the median AG News text length (~215 characters).
    """

    def __init__(self, p_long=0.93, p_short=0.62, length_threshold=215, seed=1):
        super().__init__(strong_classes=(), p_strong=p_long, p_weak=p_short, seed=seed)
        self.length_threshold = length_threshold

    def describe(self):
        return (f"Length-conditional expert: {self.p_strong:.0%} correct on articles "
                f">= {self.length_threshold} chars, {self.p_weak:.0%} on shorter ones")

    def _p_correct(self, gold, text):
        return self.p_strong if len(text) >= self.length_threshold else self.p_weak


def default_experts():
    """The experts used throughout the project. The class-conditional one is
    the primary expert for Tasks 3-5: a business/technology specialist, strong
    exactly on the classes where the Task 1 classifier is weakest, so the
    human-AI team has real complementarity to exploit."""
    return {
        "class_conditional": ClassConditionalExpert(),
        "length_conditional": LengthConditionalExpert(),
    }


def expert_report(expert, df):
    """Overall, per-class and per-length-bin accuracy of an expert on ``df``."""
    pred = expert.predict(df)
    correct = pred == df["label"].values
    lengths = df["text"].str.len().values

    bins = [(0, 150), (150, 250), (250, 400), (400, 10_000)]
    by_length = []
    for lo, hi in bins:
        mask = (lengths >= lo) & (lengths < hi)
        if mask.sum() == 0:
            continue
        by_length.append({
            "bin": f"{lo}-{hi if hi < 10_000 else '+'} chars",
            "accuracy": float(correct[mask].mean()),
            "n": int(mask.sum()),
        })

    return {
        "description": expert.describe(),
        "accuracy": float(correct.mean()),
        "per_class": [
            {
                "class": data.CLASS_NAMES[k],
                "accuracy": float(correct[df["label"].values == k].mean()),
                "n": int((df["label"].values == k).sum()),
            }
            for k in range(data.N_CLASSES)
        ],
        "by_length": by_length,
    }
