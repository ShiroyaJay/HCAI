# Human-Centric Artificial Intelligence (HCAI)

> Course project repository — TUHH, Semester 2

This repository holds my coursework for **Human-Centric Artificial Intelligence**. The course is about the human-centric aspects of machine learning, so every project is articulated around **interaction with humans**: each one is a small application meant to be *used* by real users, not just a model script.

---

## 🎯 Final Goal

One single **Django website** that bundles all sub-projects (5 small projects) behind a shared **launch page** (`home` app). Each sub-project is a Django app inside this one project, reachable from links on the home page.

**Requirements (from the course):**

- **Framework:** Django (mandated by the course).
- **One Django project:** All sub-projects are grouped in one single Django project.
- **Launch page:** The apps are accessed from a home/launch page showing group info (names + matriculation numbers) and links to each project.
- **Official submission:** The project is sent to the professors **as a Git repository**.
- Group work allowed (up to 5 students); expectations scale with group size.

**My additional goal:**

- **Deploy to Vercel** so the whole project is also accessible via **one public link** — the professor (or anyone) can open the URL and use every sub-project live, without cloning and running anything.

In short: **5 small HCAI apps → 1 Django project → Git repo (official submission) + Vercel link (live access).**

---

## What is Human-Centric AI?

Human-Centric AI (HCAI) is an approach to building intelligent systems where the goal is not just to maximize accuracy or automation, but to **amplify, augment, and respect human agency**. It sits at the intersection of machine learning, human-computer interaction, and ethics.

Core principles that guide the work in this repository:

- **Human control & oversight** — people stay in the loop and can understand, override, or correct the system.
- **Transparency & explainability** — decisions made by AI should be interpretable, not black boxes.
- **Fairness & non-discrimination** — actively detect and mitigate bias across groups.
- **Privacy & trust** — handle data responsibly; earn and keep user trust.
- **Usability & collaboration** — AI as a partner that supports human goals, not a replacement that sidelines them.
- **Accountability** — clear responsibility for outcomes and decisions.

A recurring design question across all projects (straight from the course): **what should the user be in control of, and what should be automated?**

---

## 🏗️ Architecture (defined by the course skeleton)

The project **starts from the official course skeleton**: <https://github.com/ppaamm/HCAI-PBL>. It already provides the Django project, the `home` launch app, and a `demos` app with reference examples (file upload, showing matplotlib plots via the media directory).

```
                  ┌────────────────────────────────────┐
   User/Professor │  One Django project (HCAI-PBL)     │
        │         └────────────────┬───────────────────┘
        ▼                          │
   /home/  ── launch page: group info + links to all projects
        │
        ├── /project1/   app 1  (Automated ML — supervised learning interface)
        ├── /project2/   app 2  (Explainability — trees/logreg, counterfactuals, PDP/ALE)
        ├── /project3/   app 3  (topic TBD)
        ├── /project4/   app 4  (topic TBD)
        ├── /project5/   app 5  (topic TBD)
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
- Home page content (group members, project links) is driven **from Python (views), not hardcoded in HTML**.
- Matplotlib plots can't render directly in Django: generate the figure → save to `media/` → load as image (see `demos` app). Alternative: JS chart libraries (Chart.js).
- Style matters but is not the point: *"reasonable style, not too much effort"* — the course is not about UX design.

---

## 📁 Repository Structure

```
HCAI/
├── README.md               # Overview of the whole subject project (this file)
├── CLAUDE.md               # Working guidelines for AI-assisted development
├── HCAI-project_01.pdf     # Official brief for Project 1
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
│   ├── models.py           # Django models if needed (e.g. algorithms/variables)
│   └── PLAN.md             # this project's own goal, plan, and tasks
├── project2/ … project5/   # same layout, added as each brief is released
│
│   # ── deployment (my Vercel goal) ──
├── requirements.txt
├── vercel.json             # Vercel build & routing config
└── api/index.py            # Serverless WSGI entry point wrapping Django
```

> **Whole-project vs. sub-projects:** this README only covers the overall structure and goals. Each sub-project gets its **own plan** (goal, tasks, design decisions) inside its app folder when its brief is released — those plans don't live here.

---

## 📚 Sub-Projects

Each sub-project has its own official brief (PDF) and will get its own detailed plan when work on it starts.

| # | App | Topic | Brief | Status |
|---|-----|-------|-------|--------|
| 1 | `project1` | Automated Machine Learning — supervised learning interface (upload CSV → visualize → train & evaluate sklearn models) | [`HCAI-project_01.pdf`](./HCAI-project_01.pdf) | 🟡 In progress |
| 2 | `project2` | Explainability — interpretable models (decision tree / logistic regression with λ regularization), counterfactual explanations, and from-scratch PDP/ALE feature-effect plots on Palmer Penguins | [`project2/HCAI-project_02.pdf`](./project2/HCAI-project_02.pdf) | 🟢 Done |
| 3 | `project3` | _TBD_ | — | ⬜ Not started |
| 4 | `project4` | _TBD_ | — | ⬜ Not started |
| 5 | `project5` | _TBD_ | — | ⬜ Not started |

> Grading note from the course: *"The minimum expected is a working solution for the given tasks, but the more you do, the higher your grade."*

---

## 🛠️ Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **Django** | Course requirement |
| ML | **scikit-learn** (+ pandas, matplotlib) | Recommended by the course briefs |
| Frontend | Django templates + global/per-app CSS | Skeleton convention |
| Plots | matplotlib → `media/` images, or Chart.js | The two options named in the brief |
| Versioning | **Git** | Official submission format |
| Deployment | **Vercel** (Python serverless + WhiteNoise for static) | My single-public-link goal |

---

## 🗺️ Roadmap (whole project)

**Phase 0 — Setup**
- [x] Clone the official skeleton (`HCAI-PBL`) into this repo and get `python manage.py runserver` working
- [x] Set up venv + `requirements.txt` (Django 4.2 LTS for Python 3.9); root `/` redirects to the `/home/` launch page
- [ ] Task 1: replace the placeholder students with the real group members' names + matriculation numbers (edit `home/views.py`, from Python — not the HTML)
- [ ] Initialize Git history properly (this is the submission artifact)

**Phase 1..5 — One sub-project at a time** (each gets its own plan)
- [ ] Project 1: Automated ML interface — plan in `project1/PLAN.md` when work starts
- [ ] Projects 2–5: as each brief is released

**Phase 6 — Deployment & handoff**
- [ ] Adapt the project for Vercel (`api/index.py`, `vercel.json`, WhiteNoise, env-driven settings)
- [ ] Solve the `media/` problem on Vercel (serverless filesystem is read-only — render plots in-memory/base64 or via Chart.js instead of writing files)
- [ ] Final check of the live link end-to-end
- [ ] Submit: Git repository (official) + Vercel URL (live access)

> **Deployment caveats to keep in mind while building:** Vercel functions have a size limit (~250 MB) and short timeouts — keep models light and training fast; avoid writing to disk at request time (affects the matplotlib-to-media pattern); SQLite won't persist — prefer stateless apps or session-based state, external Postgres only if truly needed.

---

## 💻 Running Locally

```bash
# from the repo root (where manage.py lives)
source venv/bin/activate          # first time: python3 -m venv venv && pip install -r requirements.txt
python manage.py migrate          # first time only
python manage.py runserver
# → http://127.0.0.1:8000/  (root redirects to the /home/ launch page)
```

---

## Author

**Jay Shiroya** — TUHH, Semester 2
Subject: Human-Centric Artificial Intelligence
# HCAI
