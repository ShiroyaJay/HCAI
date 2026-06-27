"""Tests for the Project 1 supervised-learning interface.

Covers the ML engine (ml.py) on its own and the full guided flow through the
views, including the friendly error cases the app promises never to crash on.
"""

import io

import pandas as pd
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from . import ml


def _category_df(n=40):
    """A small two-class, two-feature category dataset."""
    rows = []
    for i in range(n):
        a, b = (1.0, 1.0) if i % 2 == 0 else (5.0, 5.0)
        rows.append((a + i * 0.01, b + i * 0.01, "A" if i % 2 == 0 else "B"))
    return pd.DataFrame(rows, columns=["x", "y", "group"])


def _number_df(n=60):
    """A small dataset whose last column is a continuous number to predict."""
    rows = [(s, r, s * 1000 + r * 500) for s in range(10, 10 + n) for r in (1,)]
    return pd.DataFrame(rows, columns=["size", "rooms", "price"])


class MlEngineTests(TestCase):
    def test_detect_problem_type(self):
        self.assertEqual(ml.detect_problem_type(pd.Series(["a", "b", "a"])), "category")
        self.assertEqual(ml.detect_problem_type(pd.Series([0, 1, 2, 1, 0])), "category")
        self.assertEqual(ml.detect_problem_type(pd.Series(range(100))), "number")

    def test_teach_computer_category(self):
        result = ml.teach_computer(_category_df(), "group")
        self.assertTrue(result["ok"])
        self.assertEqual(result["problem"], "category")
        self.assertIn("out of", result["headline"])  # "right N out of M times"
        self.assertTrue(result["compared"])  # several methods were compared

    def test_teach_computer_number(self):
        result = ml.teach_computer(_number_df(), "price")
        self.assertTrue(result["ok"])
        self.assertEqual(result["problem"], "number")
        self.assertIn("off by about", result["headline"])
        self.assertTrue(result["compared"])

    def test_explain_preference_returns_a_why(self):
        result = ml.teach_computer(_category_df(), "group", prefer="explain")
        self.assertTrue(result["ok"])
        self.assertTrue(result["explanation"])
        self.assertIn("paid attention", result["explanation"])

    def test_accurate_preference_has_no_explanation(self):
        result = ml.teach_computer(_category_df(), "group", prefer="accurate")
        self.assertIsNone(result["explanation"])

    def test_problem_override_forces_category(self):
        # A continuous price, forced to be treated as a group instead of a number.
        result = ml.teach_computer(_number_df(), "price", problem_override="category")
        self.assertEqual(result["problem"], "category")

    def test_number_override_on_text_target_falls_back_safely(self):
        # You can't predict a *number* when the answers are words — handled, no crash.
        result = ml.teach_computer(_category_df(), "group", problem_override="number")
        self.assertEqual(result["problem"], "category")

    def test_id_column_is_dropped(self):
        df = _category_df()
        df.insert(0, "id", range(1, len(df) + 1))
        clean = ml.drop_id_column(df)
        self.assertNotIn("id", clean.columns)
        self.assertIn("group", clean.columns)  # the target is kept

    def test_id_named_target_is_not_dropped(self):
        # A last column literally named 'id' is the thing to guess — keep it.
        df = pd.DataFrame({"x": [1, 2], "id": ["a", "b"]})
        self.assertIn("id", ml.drop_id_column(df).columns)

    def test_no_numeric_features_is_handled(self):
        df = pd.DataFrame({"name": ["a", "b", "c", "d", "e"],
                           "fruit": ["x", "y", "x", "y", "x"]})
        result = ml.teach_computer(df, "fruit")
        self.assertFalse(result["ok"])
        self.assertIn("numbers", result["reason"])

    def test_too_few_rows_is_handled(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": ["x", "y"]})
        result = ml.teach_computer(df, "label")
        self.assertFalse(result["ok"])
        self.assertIn("5 examples", result["reason"])


class FlowViewTests(TestCase):
    def _upload(self, csv_text, name="data.csv"):
        f = SimpleUploadedFile(name, csv_text.encode("utf-8"), content_type="text/csv")
        return self.client.post(reverse("project1:upload"), {"file": f})

    def test_index_loads(self):
        self.assertEqual(self.client.get(reverse("project1:index")).status_code, 200)

    def test_data_without_upload_redirects_home(self):
        resp = self.client.get(reverse("project1:data"))
        self.assertRedirects(resp, reverse("project1:index"))

    def test_example_then_full_classification_flow(self):
        # Built-in example loads and lands on the data page.
        self.assertRedirects(self.client.get(reverse("project1:example")),
                             reverse("project1:data"))
        self.assertEqual(self.client.get(reverse("project1:data")).status_code, 200)
        self.assertEqual(self.client.get(reverse("project1:choose")).status_code, 200)
        # Train on the example's target and read the friendly headline.
        self.client.post(reverse("project1:train"), {"target": "variety"})
        page = self.client.get(reverse("project1:results"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "out of 10")

    def test_uploaded_regression_flow(self):
        df = _number_df()
        self.assertRedirects(self._upload(df.to_csv(index=False)),
                             reverse("project1:data"))
        self.client.post(reverse("project1:train"), {"target": "price"})
        page = self.client.get(reverse("project1:results"))
        self.assertContains(page, "off by about")

    def test_unreadable_file_shows_friendly_error(self):
        resp = self._upload("not a table at all")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "at least two columns")

    def test_no_file_shows_friendly_error(self):
        resp = self.client.post(reverse("project1:upload"))
        self.assertContains(resp, "choose a file")

    def test_all_text_columns_shows_friendly_error(self):
        csv = "name,color,fruit\na,red,yes\nb,green,no\nc,red,yes\nd,green,no\ne,red,yes\n"
        self.assertRedirects(self._upload(csv), reverse("project1:data"))
        self.client.post(reverse("project1:train"), {"target": "fruit"})
        page = self.client.get(reverse("project1:results"))
        self.assertContains(page, "column of numbers")
