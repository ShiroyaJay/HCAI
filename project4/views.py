"""Views for Project 4: landing page, the user-study protocol, and a demo of
adaptive question selection.

Participant state lives entirely in the session, so no request writes to disk.
Every step is POST-redirect-GET, so refreshing never double-records.
"""

import numpy as np
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render

from . import data, features, preference, selection, study
from .study import SESSION_KEY

# How many of the learned weights get a slider above the fold. The rest stay
# editable behind a disclosure.
SLIDERS_VISIBLE = 12

# The range the sliders span. Enforced on the way in as well as on the way out,
# so a stored override always equals the number the participant saw.
OVERRIDE_LIMIT = 3.0


def _state(request):
    return request.session.get(SESSION_KEY)


def _save(request, state):
    request.session[SESSION_KEY] = state
    request.session.modified = True


def index(request):
    return render(request, "project4/landing.html", {
        "n_movies": len(data.catalogue()),
        "n_pool": len(data.study_pool_index()),
        "n_features": features.dimension(),
        "in_progress": bool(_state(request)),
    })


def report_pdf(request):
    from . import report
    return HttpResponse(report.build(), content_type="application/pdf",
                        headers={"Content-Disposition":
                                 'attachment; filename="project4_report.pdf"'})


def study_start(request):
    if request.method == "POST":
        _save(request, study.new_state())
    return redirect("project4:study")


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------

def study_step(request):
    state = _state(request)
    if state is None:
        return redirect("project4:index")

    st = study.step(state)
    ctx = {"state": state, "step": st, "step_no": state["s"] + 1,
           "n_steps": len(study.STEPS), "design": study.current_design(state),
           "can_go_back": study.can_go_back(state)}

    if st in ("consent", "demographics", "final"):
        answers = state["g"] if st == "demographics" else state["q"].get(st, {})
        ctx["questions"] = _with_answers(study.QUESTIONS.get(st, []), answers)
        ctx["title"], ctx["intro"] = _form_copy(st)
        return render(request, "project4/form_step.html", ctx)

    if st in ("rtlx_a", "rtlx_b"):
        answers = state["q"].get(f"rtlx_{ctx['design']}", {})
        ctx["questions"] = _with_answers(
            [(k, q, study.RTLX_SCALE) for k, q in study.RTLX_ITEMS], answers)
        ctx["title"] = "How did that section feel?"
        ctx["intro"] = ("Thinking about the section you have just finished, in which you "
                        + ("chose between two films" if ctx["design"] == study.PAIRWISE
                           else "ranked ten films") + ", rate each statement.")
        return render(request, "project4/form_step.html", ctx)

    if st in ("intro_a", "intro_b", "break"):
        return render(request, "project4/interstitial.html", ctx)

    if st in ("practice_a", "practice_b"):
        return _task_page(request, state, ctx, practice=True)

    if st in ("block_a", "block_b"):
        return _task_page(request, state, ctx, practice=False)

    if st in ("validation_1", "validation_2"):
        page1, page2 = study.validation_items(state)
        prior = study.prior_validation_ratings(state, st)
        items = data.movies(page1 if st == "validation_1" else page2)
        for m in items:
            m["prior"] = prior.get(m["index"])
        ctx["items"] = items
        ctx["scale"] = study.LIKERT_7
        return render(request, "project4/validation.html", ctx)

    if st == "recs":
        return _recs(request, state, ctx)

    if st == "reveal":
        return _reveal(request, state, ctx)

    return render(request, "project4/debrief.html", ctx)


def _form_copy(st):
    return {
        "consent": ("Consent", "Please read and confirm before taking part."),
        "demographics": ("A few questions about you", "This helps us describe who took part."),
        "final": ("Comparing the two methods", "You have now used both. A few final questions."),
    }[st]


def _with_answers(questions, answers):
    """Pair each question's choices with whether that was already answered.

    A step revisited via Back should show what the participant chose, not a
    blank form, so form_step.html renders `checked`/prefilled from this
    instead of resubmitting looking like the answers were wiped.
    """
    out = []
    for key, question, choices in questions:
        prior = answers.get(key, "")
        opts = [(c, c == prior) for c in choices] if choices else None
        out.append((key, question, opts, prior))
    return out


def _task_page(request, state, ctx, practice):
    """One elicitation task, in practice or inside a timed block.

    The two share a template: a practice trial has to be the real task, or it
    does not do its job.
    """
    design = ctx["design"]
    if practice:
        left = None
    else:
        study.start_block(state)
        left = study.remaining(state)
        if left <= 0:
            # The clock, not the task, ends the block. Anything the participant
            # had ordered by now is kept as a partial ranking.
            _save(request, study.finish_block(state, study.slot_of(ctx["step"])))
            return redirect("project4:study")

    items = study.next_task(state, design, phase="practice" if practice else "block")
    _save(request, state)
    ctx.update({"movies": data.movies(items), "remaining": left, "practice": practice,
                "done": len(state["p"]) if design == study.PAIRWISE else len(state["r"])})
    if design == study.RANKING:
        picked = state.get("picked") or []
        ctx["picked"] = data.movies(picked)
        ctx["movies"] = data.movies([i for i in items if i not in picked])
    return render(request, "project4/pairwise.html" if design == study.PAIRWISE
                  else "project4/rank.html", ctx)


def _design_recommendations(state, X, n=5):
    """Top-n from each block's fitted w, in the masked order shown to the user."""
    pool = data.recommend_pool_index()
    excluded = study.seen(state)
    out = {}
    for slot in ("a", "b"):
        design = study.design_for(state, slot)
        w = preference.fit_map(study.observations(state, design), X)
        out[slot] = {"design": design, "w": w,
                     "movies": data.movies(preference.recommend(
                         w, pool, n=n, exclude=excluded, X=X))}
    first = study.recs_first_slot(state)
    second = "b" if first == "a" else "a"
    return [out[first], out[second]]


def _recs(request, state, ctx):
    """H4, output half: rate two unlabelled lists, one fitted from each block.

    Blind on purpose. A participant who knows which list came from the ranking
    task cannot rate it independently of how they felt about ranking, and the
    process half of H4 was already asked on the previous screen.
    """
    lists = _design_recommendations(state, features.build_matrix())
    ctx.update({"lists": lists, "questions": study.QUESTIONS["recs"]})
    return render(request, "project4/recs.html", ctx)


def _reveal(request, state, ctx):
    X = features.build_matrix()
    results = []
    for slot in ("a", "b"):
        design = study.design_for(state, slot)
        obs = study.observations(state, design)
        w = preference.fit_map(obs, X)
        cov = preference.laplace_cov(w, obs, X) if obs else None
        likes, dislikes = preference.explain(w, cov)
        results.append({
            "design": design,
            "label": "Choosing between two films" if design == study.PAIRWISE
                     else "Ranking ten films",
            "n_obs": len(obs),
            "seconds": state["t"].get(slot, 0),
            "likes": likes, "dislikes": dislikes,
            "spearman": _spearman(state, w, X),
        })

    # The combined estimate drives the recommendations and the override sliders.
    combined = preference.fit_map(
        study.observations(state, study.PAIRWISE) + study.observations(state, study.RANKING), X)
    w = _apply_overrides(state, combined)

    # Order the sliders by the model's weights, never by the overridden ones:
    # ordering by the live values makes a slider you have just dragged reshuffle
    # the panel, which is disorienting to use.
    table = preference.weight_table(w, order_by=combined)
    pool = data.recommend_pool_index()
    unseen = [int(i) for i in pool if int(i) not in study.seen(state)]
    top = preference.recommend(w, unseen, n=5, X=X)
    bottom = [int(i) for i in preference.rank_movies(w, unseen, X, best_first=False)[:3]]

    ctx.update({
        "results": results,
        "top": data.movies(top),
        "bottom": data.movies(bottom),
        "sliders": table[:SLIDERS_VISIBLE],
        "more_sliders": table[SLIDERS_VISIBLE:],
        "overridden": sorted((state.get("w") or {}).keys()),
        "lists": _design_recommendations(state, X),
        "recs_answers": state["q"].get("recs", {}),
        "consistency": _consistency(state),
    })
    return render(request, "project4/reveal.html", ctx)


def _apply_overrides(state, w):
    w = np.array(w, dtype=float)
    for name, value in (state.get("w") or {}).items():
        if name in features.FEATURE_NAMES:
            w[features.FEATURE_NAMES.index(name)] = _clamp(value)
    return w


def _clamp(value):
    return max(-OVERRIDE_LIMIT, min(OVERRIDE_LIMIT, float(value)))


def _spearman(state, w, X):
    """Rank correlation between predicted utility and the stated ratings."""
    seen_once = {}
    for idx, r in state["v"]:
        seen_once.setdefault(idx, []).append(r)
    items = [(i, np.mean(rs)) for i, rs in seen_once.items()]
    if len(items) < 3:
        return None
    idx = np.array([i for i, _ in items])
    stated = np.array([r for _, r in items], dtype=float)
    pred = X[idx] @ w
    return round(float(_rank_corr(pred, stated)), 3)


def _rank_corr(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return 0.0 if denom == 0 else float((ra * rb).sum() / denom)


def _ranks(v):
    order = np.argsort(v)
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    return r


def _consistency(state):
    """Mean absolute difference between repeated validation ratings."""
    by_movie = {}
    for idx, r in state["v"]:
        by_movie.setdefault(idx, []).append(r)
    diffs = [max(rs) - min(rs) for rs in by_movie.values() if len(rs) > 1]
    if not diffs:
        return None
    return {"n": len(diffs), "mean_gap": round(float(np.mean(diffs)), 2)}


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------

def _clean(key, value):
    """Keep a submitted answer only if it is one of the options offered."""
    allowed = study.ALLOWED_ANSWERS.get(key)
    if allowed is None:
        return str(value)[:study.MAX_FREE_TEXT]
    return value if value in allowed else ""


def study_submit(request):
    state = _state(request)
    if state is None or request.method != "POST":
        return redirect("project4:index")
    st = study.step(state)
    post = request.POST

    if post.get("back"):
        if study.can_go_back(state):
            study.go_back(state)
        _save(request, state)
        return redirect("project4:study")

    if st == "consent":
        # The `required` attribute on the checkbox is a browser hint, not a
        # gate. Consent is the one thing that must be checked server-side.
        if post.get("agree") == "yes":
            study.advance(state)

    elif st in ("demographics", "final", "recs"):
        target = state["g"] if st == "demographics" else state["q"].setdefault(st, {})
        for key, *_ in study.QUESTIONS[st]:
            target[key] = _clean(key, post.get(key, ""))
        study.advance(state)

    elif st in ("rtlx_a", "rtlx_b"):
        state["q"][f"rtlx_{study.current_design(state)}"] = {
            k: _clean(k, post.get(k, "")) for k, _q in study.RTLX_ITEMS}
        study.advance(state)

    elif st in ("intro_a", "intro_b", "break"):
        study.advance(state)

    elif st in ("practice_a", "practice_b"):
        _practice_submit(state, post)

    elif st in ("block_a", "block_b"):
        slot = study.slot_of(st)
        if post.get("skip"):
            study.finish_block(state, slot)
        elif study.current_design(state) == study.PAIRWISE:
            if post.get("skip_pair"):
                study.skip_pair(state)
            else:
                choice = post.get("choice")
                if choice in ("0", "1") and state.get("cur"):
                    study.record_pairwise(state, choice)
        elif post.get("stop_ranking") and state.get("picked"):
            study.stop_ranking(state)
        else:
            _rank_click(state, post, record=True)

    elif st in ("validation_1", "validation_2"):
        page1, page2 = study.validation_items(state)
        # Resubmitting this page (reached via Back) replaces its ratings rather
        # than appending a second copy: `v` is trimmed back to how it stood
        # before this page's last submit, first.
        vlen = state.setdefault("vlen", {})
        prev_len = vlen.get(st)
        if prev_len is not None:
            state["v"] = state["v"][:prev_len]
        vlen[st] = len(state["v"])
        for idx in (page1 if st == "validation_1" else page2):
            value = post.get(f"m{idx}", "")
            if value.isdigit() and 1 <= int(value) <= len(study.LIKERT_7):
                state["v"].append([int(idx), int(value)])
        study.advance(state)

    elif st == "reveal":
        overrides = {}
        for name in features.FEATURE_NAMES:
            if name in post:
                try:
                    overrides[name] = round(_clamp(post[name]), 3)
                except ValueError:
                    pass
        if post.get("next"):
            study.advance(state)
        elif post.get("reset"):
            state["w"] = None
        else:
            state["w"] = overrides

    _save(request, state)
    return redirect("project4:study")


def _practice_submit(state, post):
    """A practice task is the real task with the recording turned off."""
    if post.get("skip"):
        study.clear_task(state)
        study.advance(state)
        return
    if study.current_design(state) == study.PAIRWISE:
        if post.get("choice") in ("0", "1") and state.get("cur"):
            study.clear_task(state)
            study.advance(state)
        return
    if _rank_click(state, post, record=False):
        study.advance(state)


def _rank_click(state, post, record):
    """Design 2 is sequential best-choice: each click removes one film.

    This is exactly the Plackett-Luce generative process ("repeatedly pick the
    best of what is left"), so the interface matches the likelihood rather than
    approximating it, and it needs no JavaScript, which also means it works
    on touch devices.

    Returns True when the set has been fully ordered.
    """
    try:
        chosen = int(post.get("pick", ""))
    except ValueError:
        return False
    shown = state.get("cur") or []
    picked = state.get("picked") or []
    if chosen not in shown or chosen in picked:
        return False
    picked.append(chosen)
    state["picked"] = picked
    state["pd"] = (state.get("pd") or []) + [study.take_ds(state)]
    if len(picked) < len(shown) - 1:
        return False
    if record:
        study.record_ranking(state, picked, state["pd"])
    else:
        study.clear_task(state)
    return True


def export_json(request):
    state = _state(request)
    if state is None:
        return redirect("project4:index")
    return JsonResponse(study.export(state), json_dumps_params={"indent": 2},
                        headers={"Content-Disposition":
                                 'attachment; filename="project4_responses.json"'})


# --------------------------------------------------------------------------
# Demo: adaptive vs random question selection (outside the study path)
# --------------------------------------------------------------------------

def demo(request):
    mode = request.GET.get("mode", "random")
    if mode not in ("random", "adaptive"):
        mode = "random"
    try:
        n = int(request.GET.get("n") or 20)
    except (TypeError, ValueError):
        n = 20
    n = max(5, min(n, 60))
    X = features.build_matrix()
    trace = selection.simulate_trace(X, data.study_pool_index(), mode=mode, n=n)
    return render(request, "project4/demo.html", {
        "mode": mode, "n": n, "trace": trace,
        "modes": [("random", "Uniformly at random"), ("adaptive", "Adaptive (expected information)")],
    })
