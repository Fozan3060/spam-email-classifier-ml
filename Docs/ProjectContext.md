# Spam Email Classifier ML Project – Context File

This document defines the full development plan, architecture, git workflow, and setup steps for the project.

---

# 1. Project Overview

Full-stack machine learning system to classify emails as spam or not spam using multiple models:

* KNN
* Naive Bayes
* Logistic Regression
* SVM
* Neural Network

Includes:

* Model comparison (accuracy, precision, recall, F1)
* FastAPI backend
* React frontend (Bun + Tailwind v4)

---

# 2. Repository Details

Repository Name:
spam-email-classifier-ml

---

# 3. Project Structure

spam-email-classifier-ml/
│
├── frontend/
├── backend/
├── ml/
├── models/
├── data/
├── notebooks/
├── tests/
├── README.md
└── .gitignore

---

# 4. Architecture

Type: Modular Monolith

Flow:
Frontend (React)
↓
FastAPI Backend
↓
ML Model (loaded for inference)

Notes:

* Training is separate from inference
* Backend only loads trained model

---

# 5. Git Workflow

Main Branches:

* main (production)
* develop (active development)

Feature Branch Naming:
feature/<feature-name>

---

# 6. Branch Setup Commands

Initialize repo:

git init
git add .
git commit -m "initial project setup"

Set main branch:

git branch -M main

Create develop branch:

git checkout -b develop

Create first feature branch:

git checkout -b feature/project-setup

---

# 7. Branching Strategy

For every new feature:

git checkout develop
git pull
git checkout -b feature/<feature-name>

After completion:

git checkout develop
git merge feature/<feature-name>

Release:

git checkout main
git merge develop

---

# 8. Feature Branch Plan (Phases)

Phase 1: Setup

* feature/project-setup

Phase 2: ML Pipeline

* feature/dataset-loading
* feature/text-preprocessing
* feature/feature-extraction
* feature/naive-bayes-model
* feature/other-models
* feature/model-evaluation

Phase 3: Backend

* feature/fastapi-setup
* feature/prediction-api

Phase 4: Frontend

* feature/react-setup
* feature/ui-input
* feature/api-integration
* feature/result-display

Phase 5: Advanced Features

* feature/model-comparison-dashboard
* feature/visualizations

---

# 9. Frontend Setup (Bun + React + Tailwind)

Create app:

bun create vite frontend

Select:

* React
* TypeScript + React Compiler

Install dependencies:

cd frontend
bun install

Run app:

bun run dev

---

# 10. Tailwind Setup

Install:

bun add -d tailwindcss postcss autoprefixer

Initialize:

bunx tailwindcss init -p

Update tailwind.config.js:

content: [
"./index.html",
"./src/**/*.{js,ts,jsx,tsx}",
]

Add to CSS:

@tailwind base;
@tailwind components;
@tailwind utilities;

---

# 11. Backend Setup (FastAPI)

Create backend folder:

mkdir backend
cd backend

Create virtual environment:

python -m venv venv

Activate:

Mac/Linux:
source venv/bin/activate

Windows:
venv\Scripts\activate

Install dependencies:

pip install fastapi uvicorn scikit-learn pandas numpy

Run server:

uvicorn main:app --reload

---

# 12. ML Pipeline Structure

(Add ROC-AUC evaluation as part of model comparison)

Metrics to include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* AUC Score

Note:

* ROC-AUC will help compare model performance beyond accuracy
* Especially useful for imbalanced datasets like spam detection

ml/
│
├── data_loader.py
├── preprocessing.py
├── feature_extraction.py
├── models/
│   ├── knn_model.py
│   ├── naive_bayes_model.py
│   ├── logistic_model.py
│   ├── svm_model.py
│   └── neural_network_model.py
│
├── train.py
├── evaluate.py
└── metrics.py

ml/
│
├── data_loader.py
├── preprocessing.py
├── feature_extraction.py
├── models/
│   ├── knn_model.py
│   ├── naive_bayes_model.py
│   ├── logistic_model.py
│   ├── svm_model.py
│   └── neural_network_model.py
│
├── train.py
├── evaluate.py
└── metrics.py

---

# 13. Development Order

1. Project setup and git
2. Dataset loading
3. Text preprocessing
4. Feature extraction (TF-IDF / BoW)
5. Train Naive Bayes (baseline)
6. Evaluation metrics
7. Train other models
8. FastAPI backend
9. React frontend

---

# 14. Git Commit Strategy

Good commit examples:

* setup project structure
* add dataset loader
* implement preprocessing pipeline
* train naive bayes model
* add fastapi prediction endpoint
* connect frontend with backend

---

# 15. Important Rules

* Do NOT train models in API
* Save trained model in /models
* Backend only loads model

---

# 16. .gitignore

node_modules/
venv/
**pycache**/
.env
models/*.pkl

---

# 17. Future Enhancements

Core Enhancements:

* Model comparison dashboard (table + filters)
* Confusion matrix visualization (per model)
* ROC Curve visualization (per model)
* AUC comparison across models (bar/line chart)
* Performance charts (Accuracy, Precision, Recall, F1)
* Docker deployment (frontend + backend)

Advanced Enhancements (to make it stand out):

* Threshold tuning UI (adjust decision threshold and see metrics change)
* Class imbalance handling (SMOTE or class weights) with comparison
* Cross-validation results (k-fold) visualization
* Model persistence/versioning (save multiple trained models)
* Inference latency comparison (ms per model)
* API rate limiting & logging
* Basic auth for dashboard
* CI/CD pipeline (GitHub Actions) for lint/test/build
* Docker Compose for one-command run
* Simple caching layer for repeated predictions
* Downloadable report (CSV/JSON of metrics)

UI Ideas:

* Tabs: Predict | Compare Models | Visualizations
* Charts: ROC, Confusion Matrix heatmap, Metric bars
* Clean cards for each model with key stats

Notes:

* Keep features incremental; don’t build everything at once.
* Prioritize: ROC/AUC + Confusion Matrix + Comparison Dashboard first.
