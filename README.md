# Human-Centric Artificial Intelligence (HCAI)

> Course project repository — TUHH, Semester 2

This repository holds my coursework for **Human-Centric Artificial Intelligence**. The course is about the human-centric aspects of machine learning, so every project is articulated around interaction with humans: each one is a small application meant to be used by real users.

---

## Final Goal

One single **Django website** with all sub-projects (4 small projects) behind a shared **launch page** (`home` app). Each sub-project is a Django app inside this one project, reachable from links on the home page.

**Requirements (from the course):**

- **Framework:** Django (mandated by the course).
- **One Django project:** All sub-projects are grouped in one single Django project.
- **Launch page:** The apps are accessed from a home/launch page showing group info (names + matriculation numbers) and links to each project.
- **Official submission:** The project is submitted **as a Git repository**.
  
---

## Architecture (defined by the course skeleton)

The project **starts from the official course skeleton**: <https://github.com/ppaamm/HCAI-PBL>. It already provides the Django project, the `home` launch app, and a `demos` app with reference examples (file upload, showing matplotlib plots via the media directory).

```
                  ┌────────────────────────────────────┐
      User        │  One Django project (HCAI-PBL)     │
        │         └────────────────┬───────────────────┘
        ▼                          │
   /home/  ── launch page: group info + links to all projects
        │
        ├── /project1/   app 1  (Automated ML — supervised learning interface)
        ├── /project2/   app 2  (Explainability — trees/logreg, counterfactuals, PDP/ALE)
        ├── /project3/   app 3  (Learning-to-defer — active learning with a human expert)
        ├── /project4/   app 4  (Preference elicitation — Plackett-Luce, user-study interface)
        ├── /demos/      course skeleton examples (file upload, matplotlib via media)
        └── /admin/      Django admin
```

**How each new sub-project gets added** (same recipe every time):

1. `python manage.py startapp projectN`
2. Register it in `INSTALLED_APPS` (settings.py)
3. Create the app's `urls.py` with `app_name = 'projectN'` (namespaced URLs)
4. Include it in the root `urls.py`: `path('projectN/', include('projectN.urls'))`
5. Add its link to the home page's projects list: `{"name": "Project N", "url_name": "projectN:index"}`

**Course-mandated conventions (must keep):**

- Templates live at `templates/[APP_NAME]/[FILE_NAME].html` inside each app.
- Global CSS lives in the root `static/` directory (shared by all projects); each app gets its own stylesheet under `static/[app_name]/` (same pattern as `static/home`).
- Home page content (group members, project links) is driven from Python (views).
- Matplotlib plots can't render directly in Django: generate the figure, save it to `media/`, then load it as an image (see `demos` app). Alternative: JS chart libraries (Chart.js).

---

## Repository Structure

```
HCAI/
├── README.md               # Overview of the whole subject project (this file)
│
│   # ── from the course skeleton (HCAI-PBL) ──
├── manage.py               # Django CLI entry point
├── pbl/                    # Django project config: settings.py, root urls.py, wsgi.py, asgi.py
├── home/                   # Launch page app (provided) — group info + project links
├── demos/                  # Provided examples: file upload, plots via media
├── templates/              # Project-wide base.html (every page extends it)
├── static/                 # Global CSS + per-app stylesheets (static/home, ...)
├── media/                  # Generated artifacts (e.g. matplotlib figures)
├── requirements.txt        # Django 4.2 LTS, scikit-learn, pandas, matplotlib, numpy
├── venv/                   # Local virtualenv (git-ignored)
│
│   # ── added by me, one per sub-project ──
├── project1/               # Automated ML (supervised learning interface)
│   ├── templates/project1/ # course-mandated template path
│   ├── urls.py             # namespaced: app_name = 'project1'
│   ├── views.py
│   └── models.py           # Django models if needed (e.g. algorithms/variables)
└── project2/ … project4/   # same layout, one per sub-project
```

> Whole-project vs. sub-projects: this README covers the overall structure and how to run everything. What each sub-project does, and why it was built that way, is documented inside the app itself. Projects 3 and 4 each ship a written PDF report reachable from their own page.

---

## Sub-Projects

Each sub-project implements one of the course briefs.

| # | App | Topic |
|---|-----|-------|
| 1 | `project1` | Automated Machine Learning — supervised learning interface (upload a CSV, visualize it, then train and evaluate sklearn models) |
| 2 | `project2` | Explainability — interpretable models (decision tree / logistic regression with λ regularization), counterfactual explanations, and from-scratch PDP/ALE feature-effect plots on Palmer Penguins |
| 3 | `project3` | Active Learning for Learning-to-Defer — AG News classifier that defers to a (simulated or human) expert, learns the expert's competence profile via active learning, and generates a downloadable PDF report from the experiment artifacts |
| 4 | `project4` | Preference elicitation — a movie recommender that learns a user's taste vector from a handful of choices (Plackett–Luce over 21 interpretable features), plus the full design and running interface for a user study comparing pairwise choice against ranking |

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **Django** | Course requirement |
| ML | **scikit-learn** (+ pandas, matplotlib) | Recommended by the course briefs |
| Frontend | Django templates + global/per-app CSS | Skeleton convention |
| Plots | matplotlib → `media/` images, or Chart.js | The two options named in the brief |
| Versioning | **Git** | Official submission format |

---

## Running Locally

Needs **Python 3.9+**. From the repo root (the directory holding `manage.py`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate            # required: projects 1, 3 and 4
                                    # use sessions and will 500 without it
python manage.py runserver
# → http://127.0.0.1:8000/  (root redirects to the /home/ launch page)
```

All four datasets are vendored, so nothing is downloaded at request time.

---
