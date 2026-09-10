"""Sanity checks for the Project 2 explainability logic."""

import warnings

import numpy as np
from django.test import TestCase
from django.urls import reverse

from . import counterfactuals as cf
from . import data, feature_effects as fe, ml


class ExplainabilityLogicTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        warnings.filterwarnings("ignore")
        cls.df = data.load_penguins()
        cls.classes = data.species_classes(cls.df)
        cls.mads = data.numeric_mads(cls.df)
        X_train, X_test, y_train, y_test = data.get_split(cls.df)
        cls.split = (X_train, y_train, X_test, y_test)  # order ml.get_candidates expects

    def test_selection_extremes(self):
        """λ=0 picks the most accurate model; large λ picks the simplest."""
        for model_type in ("tree", "logreg"):
            cands = ml.get_candidates(model_type, *self.split)
            best_acc = ml.select_model(cands, ml.LAMBDA_MIN)
            simplest = ml.select_model(cands, ml.LAMBDA_MAX)
            self.assertAlmostEqual(best_acc["acc"], max(c["acc"] for c in cands))
            self.assertEqual(simplest["omega"], min(c["omega"] for c in cands))

    def test_pdp_probabilities_sum_to_one(self):
        pipe = ml.select_model(ml.get_candidates("logreg", *self.split), 0.0)["pipe"]
        _, curves = fe.pdp(pipe, self.df, "bill_length_mm", self.classes)
        total = np.sum([curves[c] for c in pipe.classes_], axis=0)
        self.assertTrue(np.allclose(total, 1.0, atol=1e-6))

    def test_ale_is_centered(self):
        # ALE is centered to ~zero mean (tree uses data-weighted centering, so the
        # unweighted mean over edges is near zero, but not exactly zero).
        for model_type in ("tree", "logreg"):
            pipe = ml.select_model(ml.get_candidates(model_type, *self.split), 0.0)["pipe"]
            _, curves = fe.ale(pipe, self.df, "flipper_length_mm", self.classes, model_type)
            for c in pipe.classes_:
                self.assertLess(abs(float(np.mean(curves[c]))), 0.05)

    def test_counterfactuals_have_target_class(self):
        pipe = ml.select_model(ml.get_candidates("tree", *self.split), 0.0)["pipe"]
        x = self.df.loc[0, data.FEATURES]
        pred = pipe.predict(self.df.loc[[0], data.FEATURES])[0]
        target = next(c for c in self.classes if c != pred)
        results = cf.generate_counterfactuals(pipe, x, target, self.df, self.mads, k=3)
        self.assertTrue(results)
        for r in results:
            verify = pipe.predict(r["row"].to_frame().T[data.FEATURES])[0]
            self.assertEqual(verify, target)

    def test_counterfactuals_are_ranked_by_distance(self):
        pipe = ml.select_model(ml.get_candidates("tree", *self.split), 0.0)["pipe"]
        x = self.df.loc[0, data.FEATURES]
        pred = pipe.predict(self.df.loc[[0], data.FEATURES])[0]
        target = next(c for c in self.classes if c != pred)
        results = cf.generate_counterfactuals(pipe, x, target, self.df, self.mads, k=5)
        distances = [r["distance"] for r in results]
        self.assertEqual(distances, sorted(distances))

    def test_counterfactuals_can_change_a_categorical_feature(self):
        # Row 0 is a Torgersen Adelie, and Gentoo only occurs on Biscoe in this
        # dataset, so reaching a Gentoo counterfactual should require changing
        # `island` for at least one of the results, which exercises the categorical
        # (resample-to-a-different-category) noising path, not just numeric noise.
        pipe = ml.select_model(ml.get_candidates("tree", *self.split), 0.0)["pipe"]
        x = self.df.loc[0, data.FEATURES]
        self.assertEqual(x["island"], "Torgersen")
        results = cf.generate_counterfactuals(pipe, x, "Gentoo", self.df, self.mads, k=5)
        self.assertTrue(results)
        self.assertTrue(any("island" in r["changed"] for r in results))


class ViewTests(TestCase):
    def test_index_loads_with_default_params(self):
        response = self.client.get(reverse("project2:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Decision tree")

    def test_invalid_model_falls_back_to_tree(self):
        response = self.client.get(reverse("project2:index"), {"model": "not-a-model"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["model_type"], "tree")

    def test_logreg_model_is_selectable(self):
        response = self.client.get(reverse("project2:index"), {"model": "logreg"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["model_type"], "logreg")
        self.assertEqual(response.context["omega_label"], "non-zero coefficients")

    def test_lambda_is_clamped_to_range(self):
        too_high = self.client.get(reverse("project2:index"), {"lam": "999"})
        self.assertEqual(too_high.context["lam"], round(ml.LAMBDA_MAX, 3))
        too_low = self.client.get(reverse("project2:index"), {"lam": "-5"})
        self.assertEqual(too_low.context["lam"], round(ml.LAMBDA_MIN, 3))

    def test_invalid_feature_falls_back_to_first_numeric(self):
        response = self.client.get(reverse("project2:index"), {"feature": "not-a-feature"})
        self.assertEqual(response.context["feature"], data.NUMERIC_FEATURES[0])
