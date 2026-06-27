# Project 1 — Plan

> App: `project1` · Route: `/project1/` · Official brief: [`../HCAI-project_01.pdf`](../HCAI-project_01.pdf)

## 🎯 The Goal (the heart of this project)

**A grandma can use it.**

A non-technical person — someone's grandmother, an elderly user, anyone with zero machine-learning background — should be able to open this app, follow it, *understand what is happening*, and get a useful result **without help and without ever feeling lost or stupid.**

The official task is a *supervised learning interface*: upload a dataset, look at it, train a machine-learning model, see how well it does. That's the engine. **Our contribution — the human-centric part — is making that engine usable and understandable by a complete beginner.** The ML is the easy part; the humanity is the graded part.

This goal decides every design choice. When in doubt, ask: *"Would Grandma understand this screen? Would she know what to do next?"* If not, it's wrong.

---

## 🧓 What "a grandma can use it" actually means (design rules)

These are hard rules for this project, not nice-to-haves:

1. **No jargon. Ever.** Banned words on screen: *model, classifier, regression, hyperparameter, train/test split, accuracy, feature, label, algorithm, dataset.* Use plain replacements:
   - "machine learning model" → **"the computer's helper"** / **"the guesser"**
   - "features" → **"the information we know"** (the columns)
   - "label / target" → **"the thing we want to guess"** (last column)
   - "train the model" → **"teach the computer"**
   - "accuracy 0.93" → **"It guessed right about 9 times out of 10."**
2. **One step at a time (a wizard).** Never show everything at once. A clear linear flow: *Step 1 → Step 2 → Step 3*, with a big **Next** button. The user is never asked to make two decisions on one screen.
3. **The app makes the technical decisions; the human keeps the human ones — and oversight.** Grandma is *never required* to choose an algorithm, a split ratio, or a score: the app fills in sensible defaults so one click is always enough. But "easy" must not mean "powerless" — for anyone who wants it, the human-meaningful choices stay available *in plain words* (what to guess; is it a group or a number; be most-accurate vs. explain-its-guesses), and the result is transparent enough to judge whether to trust it. Easy by default, in control when you want to be. (See the Task-4 answer below.)
4. **Big, calm, readable.** Large fonts, large buttons, lots of spacing, high contrast. One primary action per screen, visually obvious.
5. **Always explain in everyday words.** Every screen has one friendly sentence saying *what just happened* and *what to do now*. Short. Warm. No condescension.
6. **No dead ends.** Every error is a kind, plain-language message that says how to fix it ("This file doesn't look like a table of numbers. Try a different file, or use our example below."). Never a stack trace, never red text with codes.
7. **A safety net: a built-in example.** A big **"I don't have a file — show me with an example"** button loads a friendly sample (e.g. flowers, or houses) so she can experience the whole thing with zero preparation.

---

## 🪜 The user's journey (what Grandma sees)

A 3-step guided flow, plus a results screen:

**Step 1 — "Let's start"**
- Warm welcome, one sentence on what this does ("This helps the computer learn to make guesses from a table of information.").
- Two big buttons: **"Use my own file"** (upload a CSV) or **"Show me with an example"**.

**Step 2 — "Here's your information"**
- Show the table in a clean, readable way (first few rows).
- One friendly sentence: "Great! You gave us information about 150 flowers."
- A simple **picture** of the data (a colorful scatter plot) with a caption that reads like a sentence, not a chart title.
- Big **Next** button.

**Step 3 — "What should the computer guess?"**
- The *one* human decision: pick the thing to predict (defaults to the last column, pre-selected, with a plain explanation). For most users she just clicks **"Teach the computer"**.
- Everything technical (which algorithm, splitting the data, settings) happens automatically behind this button.

**Results — "How did it do?"**
- The headline in human terms: **"The computer guessed right about 9 out of 10 times. "** with a friendly icon.
- A simple visual (e.g. a bar or a smiley scale) — not a confusion matrix.
- One gentle line of honesty if it did poorly ("It found this one tricky — that's okay, some things are hard to guess.").
- Buttons: **"Try another file"** / **"Start over"**.

---

## 🔧 What's under the hood (the technical engine)

Hidden from the user, but this is what powers it. Built with **scikit-learn** (allowed by the brief).

- **Data loading:** read uploaded CSV with pandas. Convention from the brief: first row = column names, **last column = the thing to predict**. Strip any obvious `id` column.
- **Auto-detect the problem type:** few distinct values in the target → it's a *category guess* (classification); many continuous values → it's a *number guess* (regression). The user never hears these words; the app just adapts.
- **Visualization:** a scatter plot of two informative columns, colored by the answer (classification) or value (regression). Rendered with matplotlib.
- **Teaching the computer (training):** automatically and sensibly —
  - split data into a part to learn from and a part to test on (e.g. 80/20),
  - quietly try several models (decision tree / logistic / nearest-neighbours for categories; decision tree / straight-line / nearest-neighbours for numbers), each across several settings, and keep the single best on the test part (this satisfies the brief's "train for several hyperparameter values" and the "several learning algorithms" hint),
  - score it (accuracy for categories, R²/error for numbers) — then **translate the score into a plain sentence.**
- **The HCAI question (Task 4), our answer:** Human-Centric AI is *not* "automate everything so the human does nothing" — that is automation-centric design, and it strips the human of agency and oversight. Nor is it "expose every knob" — that is just burden a non-expert can't use. Our answer is **appropriate automation with the human kept in the loop**:
  - **Automated (no human stake — automating it is the *right* call):** the train/test split, the hyperparameter search, feature scaling, and trying several algorithms. A user gains nothing by steering these, so we don't ask.
  - **The human's, in plain language (available, never required):**
    - *What to guess* — the target column (a domain decision, not a technical one).
    - *Problem framing* — group-guess vs. number-guess is auto-detected and shown, and the user can correct it (the brief, §2.4, explicitly raises this).
    - *What they value* — "be as right as possible" vs. "explain its guesses." A genuine human trade-off (accuracy vs. interpretability), expressed with zero jargon; the app maps it to the mechanism (best-of-search vs. a readable tree that names the clues it used).
    - *Oversight* — an honest score in plain words, a "why it guessed" line, and a "technical details" panel, so the human can decide whether to **trust** the result rather than just receive it.
  - **Why this is still grandma-simple:** every one of those choices has a sensible default, so a passive user clicks straight through and succeeds. Control is offered through progressive disclosure ("Want a say? (optional)"), never imposed. *Easy by default; in control on request.* This is the deliberate, defensible answer we justify in the writeup — and it demonstrates HCAI (agency + oversight + appropriate automation), not merely good usability.

> **Optional "expert" toggle (stretch goal):** a small, hidden-by-default "Show me more" that exposes the real numbers and choices for a technical grader — so the app is grandma-simple *and* demonstrably real ML underneath. Only after the core flow works.

---

## 🖥️ Plumbing & files

Follow the course skeleton conventions exactly (same as `home`/`demos`):

- `project1/views.py` — the step views + the training logic (or split training into `project1/ml.py`).
- `project1/forms.py` — the CSV upload form (`forms.FileField`), mirroring `demos/forms.py`.
- `project1/urls.py` — `app_name = 'project1'`, paths for each step.
- `project1/templates/project1/*.html` — one template per step, each `{% extends 'base.html' %}`.
- `project1/static/project1/style.css` — project-specific styling for the big/calm/readable look (the accessibility lives here).
- Register `'project1'` in `INSTALLED_APPS`, add `path('project1/', include('project1.urls'))` to `pbl/urls.py`, and add `{"name": "Project 1", "url_name": "project1:index"}` to the home page's projects list (Task 2).
- **Plots:** locally, save to `media/` like the demo. ⚠️ For Vercel later this breaks (read-only disk) — plan to switch to in-memory base64 images or Chart.js. Build it the simple way first.

---

## ✅ Build order (each step shippable & testable)

- [x] 1. Create `urls.py` + a trivial `index` view; register app + wire into `pbl/urls.py`; add link on home page. **Verified:** `/project1/` → 200, "Project 1" button on home links to it.
- [x] 2. Step 1 screen (welcome + two big buttons) using `base.html`, with accessible `project1/style.css` (big text, big buttons). *Buttons are placeholders (`href="#"`) until Step 3.*
- [x] 3. CSV upload + read with pandas; Step 2 (`data.html`) shows the table nicely, highlights the target column, friendly count message. CSV kept in the session between steps. **Verified:** uploaded Iris → table renders.
- [x] 4. The built-in **example** button → loads the bundled `project1/data/iris.csv` (150 flowers; last column `variety`). **Verified:** works with zero upload; `/data/` without data redirects to start; bad/empty files show a kind error.
- [x] 5. Auto problem-type detection (`ml.py`) + scatter-plot picture with a plain caption on Step 2. Classification → 2 features coloured by group; regression → feature vs. target. **Verified:** Iris (category) and a houses CSV (number) both render & are served; plot saved per-session to `media/`.
- [x] 6. Step 3 `choose.html` ("what to guess", dropdown defaulting to last column) + **"Teach the computer"** → `ml.teach_computer` auto-splits, tries several models each over several settings, keeps the single best (satisfies "several hyperparameter values" + the "several learning algorithms" hint); the compared methods are shown in the results "technical details" panel. **Verified** for category & number targets.
- [x] 7. Results screen (`results.html`): score **translated to a human sentence** (classification → "right N times out of 10"; regression → "off by about X"), a friendly quality line, a coloured meter, and an optional "Show me the technical details" panel for graders. **Verified** end-to-end via Django test client.
- [x] 8. Kind error handling everywhere: unreadable/empty file, no upload, all-text columns (no numbers to learn from), too-few-rows (would break the split) — all show a friendly recoverable message, never a crash. **Verified** by a test suite.
- [x] 9. Accessibility polish: large fonts & buttons, generous spacing, high contrast, one main action per screen, plus a "Step X of 3" orientation pill so users always know where they are.
- [ ] 10. (Stretch) "Show me more" expert toggle. (Stretch) usability check — literally ask a non-technical person to try it.

---

## 📏 Definition of done

- A non-technical person can go upload → understand → result **without help**.
- No banned jargon appears anywhere on screen.
- The score is always shown as a plain-language sentence.
- Every error is friendly and recoverable; the example button means it *always* works.
- Underneath, it's a genuine sklearn supervised-learning pipeline (so it satisfies the brief's Tasks 3 & 4).
- We can clearly explain *what the user controls vs. what's automated, and why* (Task 4's core question).
