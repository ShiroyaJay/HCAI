"""Sanity checks for the Project 2 explainability logic."""

import warnings

import numpy as np
from django.test import TestCase

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
        # unweighted mean over edges is near—but not exactly—zero).
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
