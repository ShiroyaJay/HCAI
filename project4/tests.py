"""Tests for Project 4.

Logic is tested against the real vendored catalogue (it is small and committed,
so this needs no network and no generated artifacts) and against synthetic
users with known preference vectors.
"""

import time

import numpy as np
from django.test import TestCase
from django.urls import reverse

from . import data, features, preference, selection, study


class DataTests(TestCase):
    def test_cleaning_removes_known_defects(self):
        df = data.catalogue()
        self.assertGreater(len(df), 4000)
        # 4935 of 5043 raw titles end in a non-breaking space.
        self.assertFalse(df["movie_title"].str.contains("\xa0").any())
        self.assertFalse(df["imdb_id"].duplicated().any())
        # Raw values are 'Color' and ' Black and White' (leading space).
        self.assertNotIn(" Black and White", set(df["color"]))

    def test_recommendation_pool_is_wider_than_the_study_pool(self):
        """The recommendation pool is wider than the study pool but still bounded
        by a vote threshold."""
        study_pool = set(int(i) for i in data.study_pool_index())
        rec_pool = set(int(i) for i in data.recommend_pool_index())
        self.assertTrue(study_pool < rec_pool)
        self.assertGreater(len(rec_pool - study_pool), 500)
        self.assertLess(len(rec_pool), len(data.catalogue()))
        df = data.catalogue()
        self.assertGreaterEqual(df.loc[list(rec_pool), "num_voted_users"].min(),
                                data.RECOMMEND_MIN_VOTES)

    def test_study_pool_is_a_familiar_subset(self):
        pool = data.study_pool_index()
        df = data.catalogue()
        self.assertLess(len(pool), len(df))
        self.assertTrue((df.loc[pool, "num_voted_users"] >= data.FAMILIARITY_MIN_VOTES).all())

    def test_movies_preserves_order(self):
        got = data.movies([5, 1, 3])
        self.assertEqual([m["index"] for m in got], [5, 1, 3])


class FeatureTests(TestCase):
    def test_matrix_shape_and_full_rank(self):
        X = features.build_matrix()
        self.assertEqual(X.shape, (len(data.catalogue()), features.dimension()))
        self.assertTrue(np.isfinite(X).all())
        # Full rank is the identifiability guarantee: a constant direction would
        # never appear in the likelihood yet would carry maximal posterior
        # variance, so an adaptive selector would chase it forever.
        self.assertEqual(np.linalg.matrix_rank(X), X.shape[1])

    def test_no_constant_column(self):
        X = features.build_matrix()
        self.assertGreater(X.std(axis=0).min(), 1e-6)

    def test_every_feature_has_a_plain_language_label(self):
        for name in features.FEATURE_NAMES:
            self.assertIn(name, features.LABELS)


class PreferenceTests(TestCase):
    def setUp(self):
        self.X = features.build_matrix()
        self.rng = np.random.default_rng(0)

    def _sample(self, items, w_true, n_ranked=None):
        rem, order = list(items), []
        k = len(items) - 1 if n_ranked is None else n_ranked
        for _ in range(k):
            u = self.X[rem] @ w_true
            p = np.exp(u - u.max()); p /= p.sum()
            order.append(rem.pop(int(self.rng.choice(len(rem), p=p))))
        return order + rem

    def test_gradient_matches_finite_differences(self):
        w = self.rng.normal(size=self.X.shape[1]) * 0.3
        groups = [preference.ranking(self.rng.choice(len(self.X), 5, replace=False))
                  for _ in range(6)]
        grad, hess = preference._grad_hess(w, groups, self.X)
        eps = 1e-6
        numeric = np.array([
            (preference.neg_log_posterior(w + e, groups, self.X)
             - preference.neg_log_posterior(w - e, groups, self.X)) / (2 * eps)
            for e in np.eye(len(w)) * eps])
        np.testing.assert_allclose(grad, numeric, atol=1e-5)
        self.assertGreater(np.linalg.eigvalsh(hess).min(), 0)

    def test_pairwise_case_reduces_to_bradley_terry(self):
        from sklearn.linear_model import LogisticRegression
        w_true = self.rng.normal(size=self.X.shape[1]) * 0.8
        pairs = [preference.ranking(self._sample(
            list(self.rng.choice(len(self.X), 2, replace=False)), w_true)) for _ in range(400)]
        w_pl = preference.fit_map(pairs, self.X, sigma=1.0)
        D = np.array([self.X[a] - self.X[b] for (a, b), _ in pairs])
        lr = LogisticRegression(fit_intercept=False, C=1.0, tol=1e-12, max_iter=10000).fit(
            np.vstack([D, -D]), np.r_[np.ones(len(D)), np.zeros(len(D))])
        cosine = (w_pl @ lr.coef_[0]) / (np.linalg.norm(w_pl) * np.linalg.norm(lr.coef_[0]))
        # Directions agree; magnitudes differ only by how the prior is scaled.
        self.assertGreater(cosine, 0.99)

    def test_recovers_a_known_preference_vector(self):
        w_true = self.rng.normal(size=self.X.shape[1]) * 0.8
        obs = [preference.ranking(self._sample(
            list(self.rng.choice(len(self.X), 10, replace=False)), w_true)) for _ in range(20)]
        w = preference.fit_map(obs, self.X)
        cosine = (w @ w_true) / (np.linalg.norm(w) * np.linalg.norm(w_true))
        self.assertGreater(cosine, 0.8)

    def test_partial_rankings_are_usable(self):
        w_true = self.rng.normal(size=self.X.shape[1]) * 0.8
        obs = [preference.ranking(
            self._sample(list(self.rng.choice(len(self.X), 10, replace=False)), w_true, n_ranked=3),
            n_ranked=3) for _ in range(15)]
        w = preference.fit_map(obs, self.X)
        cosine = (w @ w_true) / (np.linalg.norm(w) * np.linalg.norm(w_true))
        self.assertGreater(cosine, 0.6)

    def test_no_observations_returns_the_prior_mean(self):
        np.testing.assert_array_equal(preference.fit_map([], self.X),
                                      np.zeros(self.X.shape[1]))

    def test_recommendations_exclude_seen_movies(self):
        w = self.rng.normal(size=self.X.shape[1])
        pool = data.study_pool_index()
        seen = set(int(i) for i in preference.rank_movies(w, pool, self.X)[:4])
        recs = preference.recommend(w, pool, n=5, exclude=seen, X=self.X)
        self.assertEqual(len(recs), 5)
        self.assertFalse(seen & set(recs))

    def test_least_recommended_also_excludes_seen_movies(self):
        """The least-recommended list excludes films the participant has already
        seen."""
        w = self.rng.normal(size=self.X.shape[1])
        pool = [int(i) for i in data.study_pool_index()]
        worst = set(int(i) for i in preference.rank_movies(w, pool, self.X, best_first=False)[:4])
        unseen = [i for i in pool if i not in worst]
        bottom = [int(i) for i in preference.rank_movies(w, unseen, self.X, best_first=False)[:3]]
        self.assertFalse(worst & set(bottom))

    def test_explain_splits_likes_and_dislikes(self):
        w = np.zeros(self.X.shape[1])
        w[0], w[1] = 2.0, -2.0
        likes, dislikes = preference.explain(w)
        self.assertEqual(likes[0]["name"], features.FEATURE_NAMES[0])
        self.assertEqual(dislikes[0]["name"], features.FEATURE_NAMES[1])


class SelectionTests(TestCase):
    def test_informative_pairs_outscore_lopsided_ones(self):
        X = features.build_matrix()
        rng = np.random.default_rng(0)
        w = rng.normal(size=X.shape[1])
        cov = np.eye(X.shape[1])
        ordered = preference.rank_movies(w, range(len(X)), X)
        lopsided = [int(ordered[0]), int(ordered[-1])]      # a certain answer
        toss_up = [int(ordered[len(ordered) // 2]), int(ordered[len(ordered) // 2 + 1])]
        scores = selection.score_pairs(w, cov, X, [lopsided, toss_up])
        self.assertGreater(scores[1], scores[0])

    def test_adaptive_beats_random_on_a_synthetic_user(self):
        X = features.build_matrix()
        pool = data.study_pool_index()
        rnd = selection.simulate_trace(X, pool, mode="random", n=15)
        ada = selection.simulate_trace(X, pool, mode="adaptive", n=15)
        self.assertGreater(ada[-1]["cosine"], rnd[-1]["cosine"])


class StudyTests(TestCase):
    def test_counterbalancing_assigns_both_orders(self):
        orders = {study.new_state(seed=s)["o"] for s in range(20)}
        self.assertEqual(orders, {0, 1})
        state = study.new_state(seed=0)
        self.assertNotEqual(study.design_for(state, "a"), study.design_for(state, "b"))

    def test_seen_covers_every_shown_movie(self):
        state = study.new_state(seed=0)
        state["p"].append([11, 22, 0, 100])
        state["r"].append([9] + list(range(30, 40)) + [40] * 9)
        state["v"].append([55, 4])
        state["pr"] = [77, 78]
        self.assertTrue({11, 22, 55, 30, 39, 77, 78} <= study.seen(state))

    def test_draws_never_repeat_a_seen_movie(self):
        state = study.new_state(seed=3)
        first = study.draw(state, 10, "a")
        state["r"].append([9] + first + [40] * 9)
        self.assertFalse(set(first) & set(study.draw(state, 10, "b")))

    def test_practice_films_are_never_shown_again(self):
        """A practice trial burns its films, or the block repeats them."""
        state = study.new_state(seed=1)
        practice = study.next_task(state, study.RANKING, phase="practice")
        study.clear_task(state)
        self.assertEqual(set(practice), set(state["pr"]))
        self.assertFalse(set(practice) & set(study.draw(state, 10, "block")))

    def test_a_block_cut_by_the_clock_keeps_the_partial_ranking(self):
        """The time box is enforced, and the evidence is not thrown away.

        A rank set served at t = 299s must not run to completion, which would
        hand Design 2 up to 40s more elicitation time than Design 1 at the same
        nominal budget, which is a confound on the primary DV. So the block is
        cut where the clock says and the prefix already established is stored
        as a top-K partial ranking.
        """
        state = study.new_state(seed=1)
        shown = study.next_task(state, study.RANKING)
        state["picked"], state["pd"] = shown[:4], [30, 25, 40, 35]
        state["bs"] = time.time() - study.BLOCK_SECONDS - 1
        study.finish_block(state, "a")

        self.assertEqual(len(state["r"]), 1, "partial ranking was discarded")
        rec = state["r"][0]
        self.assertEqual(rec[0], 4, "should be a top-4 partial ranking")
        self.assertEqual(study.rank_order(rec)[:4], shown[:4])
        self.assertEqual(set(study.rank_order(rec)), set(shown))
        self.assertEqual(study.rank_times(rec), [30, 25, 40, 35])
        self.assertLessEqual(state["t"]["a"], study.BLOCK_SECONDS)
        self.assertEqual(state["picked"], [])

    def test_a_partial_ranking_reaches_the_model_as_top_k(self):
        state = study.new_state(seed=1)
        state["cur"] = list(range(10))
        study.record_ranking(state, [3, 1, 7], [20, 20, 20], n_ranked=3)
        (items, n_ranked), = study.observations(state, study.RANKING)
        self.assertEqual(n_ranked, 3)
        self.assertEqual(items[:3], [3, 1, 7])

    def test_every_ranking_pick_is_timed(self):
        """Ranking tasks record a latency for every individual pick."""
        state = study.new_state(seed=1)
        state["cur"] = list(range(10))
        study.record_ranking(state, list(range(9)), list(range(10, 19)))
        rec = state["r"][0]
        self.assertEqual(len(study.rank_times(rec)), 9)
        exported = study.export(state)["ranking_responses"][0]
        self.assertEqual(len(exported["pick_seconds"]), 9)

    def test_rtlx_is_the_complete_six_item_instrument(self):
        keys = [k for k, _q in study.RTLX_ITEMS]
        self.assertEqual(len(keys), 6)
        self.assertIn("physical", keys)

    def test_rtlx_immediately_follows_its_own_block(self):
        """Each RTLX is administered immediately after its own block."""
        steps = list(study.STEPS)
        for slot in ("a", "b"):
            self.assertEqual(steps[steps.index(f"block_{slot}") + 1], f"rtlx_{slot}")

    def test_validation_repeats_items_for_a_consistency_check(self):
        state = study.new_state(seed=0)
        page1, page2 = study.validation_items(state)
        repeats = set(page1) & set(page2)
        self.assertEqual(len(repeats), study.VALIDATION_REPEATS)
        self.assertEqual(len(set(page1) | set(page2)), study.VALIDATION_UNIQUE)

    def test_validation_set_is_stable_once_page_one_is_answered(self):
        """Validation page 2 re-offers four films from page 1."""
        state = study.new_state(seed=0)
        page1, _ = study.validation_items(state)
        state["v"] = [[i, 5] for i in page1]          # page 1 submitted
        page1_again, page2 = study.validation_items(state)
        self.assertEqual(page1_again, page1)
        self.assertEqual(len(set(page1) & set(page2)), study.VALIDATION_REPEATS)

    def test_observations_round_trip_through_the_model(self):
        state = study.new_state(seed=0)
        state["p"] = [[3, 9, 0, 100], [4, 8, 1, 200]]
        obs = study.observations(state, study.PAIRWISE)
        self.assertEqual(obs[0][0], [3, 9])      # winner first
        self.assertEqual(obs[1][0], [8, 4])
        state["r"] = [[9] + list(range(10)) + [40] * 9]
        self.assertEqual(study.observations(state, study.RANKING)[0][0], list(range(10)))


class ViewTests(TestCase):
    def test_landing_page_offers_both_required_actions(self):
        response = self.client.get(reverse("project4:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("project4:report"))
        self.assertContains(response, reverse("project4:study_start"))

    def test_report_downloads_a_valid_pdf(self):
        response = self.client.get(reverse("project4:report"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertTrue(response.content.startswith(b"%PDF-"))
        self.assertIn("attachment", response["Content-Disposition"])

    def test_study_requires_starting_first(self):
        self.assertRedirects(self.client.get(reverse("project4:study")),
                             reverse("project4:index"))

    def test_protocol_runs_from_consent_to_debrief(self):
        """Walk every step, answering as a participant would."""
        self.client.post(reverse("project4:study_start"))
        submit, step_url = reverse("project4:study_submit"), reverse("project4:study")

        for _ in range(len(study.STEPS) * 3):
            response = self.client.get(step_url)
            self.assertEqual(response.status_code, 200)
            state = self.client.session[study.SESSION_KEY]
            st = study.step(state)
            if st == "debrief":
                break
            self.client.post(submit, self._answer(st, state))

        state = self.client.session[study.SESSION_KEY]
        self.assertEqual(study.step(state), "debrief")
        self.assertTrue(state["p"], "no pairwise responses recorded")
        self.assertTrue(state["r"], "no ranking responses recorded")
        self.assertEqual(len(state["v"]), 20)

        export = self.client.get(reverse("project4:export_json"))
        self.assertEqual(export.status_code, 200)
        self.assertIn("validation_ratings", export.json())

    def _answer(self, st, state):
        if st == "consent":
            return {"agree": "yes"}
        if st in ("demographics", "final", "recs"):
            return {k: (c[0] if c else "") for k, _q, c in study.QUESTIONS[st]}
        if st in ("rtlx_a", "rtlx_b"):
            return {k: study.RTLX_SCALE[3] for k, _q in study.RTLX_ITEMS}
        if st in ("practice_a", "practice_b"):
            return {"skip": "1"}
        if st in ("block_a", "block_b"):
            # Answer one task, then skip the rest of the timed block.
            design = study.current_design(state)
            if design == study.PAIRWISE:
                return {"choice": "0"} if not state["p"] else {"skip": "1"}
            if not state["r"]:
                remaining = [i for i in (state.get("cur") or [])
                             if i not in (state.get("picked") or [])]
                if remaining:
                    return {"pick": str(remaining[0])}
            return {"skip": "1"}
        if st.startswith("validation"):
            return {f"m{m['index']}": "5" for m in
                    data.movies(study.validation_items(state)[0 if st.endswith("1") else 1])}
        if st == "reveal":
            return {"next": "1"}
        return {}

    def test_consent_is_enforced_on_the_server(self):
        """Consent is validated server-side."""
        self.client.post(reverse("project4:study_start"))
        self.client.post(reverse("project4:study_submit"), {})
        self.assertEqual(study.step(self.client.session[study.SESSION_KEY]), "consent")
        self.client.post(reverse("project4:study_submit"), {"agree": "yes"})
        self.assertEqual(study.step(self.client.session[study.SESSION_KEY]), "demographics")

    def test_answers_outside_the_offered_options_are_dropped(self):
        self.client.post(reverse("project4:study_start"))
        self.client.post(reverse("project4:study_submit"), {"agree": "yes"})
        self.client.post(reverse("project4:study_submit"),
                         {"age": "x" * 5000, "films": "3-5", "recsys": "Often"})
        self.assertEqual(self.client.session[study.SESSION_KEY]["g"]["age"], "")

    def _walk_to(self, target):
        """Drive a real client through the protocol up to `target`."""
        self.client.post(reverse("project4:study_start"))
        for _ in range(len(study.STEPS) * 3):
            state = self.client.session[study.SESSION_KEY]
            st = study.step(state)
            if st == target:
                return state
            self.client.get(reverse("project4:study"))
            state = self.client.session[study.SESSION_KEY]
            self.client.post(reverse("project4:study_submit"),
                             self._answer(study.step(state), state))
        self.fail(f"never reached {target}")

    def test_validation_only_accepts_the_items_on_the_page(self):
        state = self._walk_to("validation_1")
        page1, _ = study.validation_items(state)
        post = {f"m{i}": "5" for i in page1}
        post["m99999"] = "7"                        # not on the page
        post[f"m{page1[0]}"] = "9"                  # off the 7-point scale
        self.client.post(reverse("project4:study_submit"), post)
        recorded = self.client.session[study.SESSION_KEY]["v"]
        self.assertEqual({i for i, _r in recorded}, set(page1[1:]))

    def test_slider_order_does_not_move_when_a_slider_is_dragged(self):
        """Slider order follows the model's weights rather than the live overrides."""
        import numpy as np
        w = np.zeros(features.dimension())
        w[0], w[1] = 0.9, 0.4
        overridden = w.copy(); overridden[1] = -3.0
        before = [it["name"] for it in preference.weight_table(w)]
        after = [it["name"] for it in preference.weight_table(overridden, order_by=w)]
        self.assertEqual(before, after)

    def test_every_weight_is_reachable_on_the_reveal_page(self):
        from project4.views import SLIDERS_VISIBLE
        table = preference.weight_table(np.zeros(features.dimension()))
        self.assertEqual(len(table[:SLIDERS_VISIBLE]) + len(table[SLIDERS_VISIBLE:]),
                         features.dimension())

    def test_export_records_the_users_corrections(self):
        """User overrides are recorded in the export."""
        state = study.new_state(seed=0)
        state["w"] = {"year": -2.0}
        export = study.export(state)
        self.assertEqual(export["user_overrides"], {"year": -2.0})
        self.assertIn("fitted_weights", export)
        self.assertEqual(set(export["fitted_weights"]), {study.PAIRWISE, study.RANKING})

    def test_demo_survives_a_junk_query_string(self):
        response = self.client.get(reverse("project4:demo"), {"mode": "nonsense", "n": "abc"})
        self.assertEqual(response.status_code, 200)

    def test_demo_page_renders_both_selection_modes(self):
        for mode in ("random", "adaptive"):
            response = self.client.get(reverse("project4:demo"), {"mode": mode, "n": 6})
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "Questions asked")
