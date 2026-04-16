# Spam Email Classifier ML Project – Context File

This document defines the architecture, structure, and development workflow for the project.

---

# 1. Project Overview

Full-stack machine learning system to classify emails as spam or not spam using multiple models:

* Naive Bayes (MultinomialNB)
* KNN
* SVM
* Logistic Regression
* Neural Network (MLPClassifier)

Includes:

* Model comparison (accuracy, precision, recall, F1, AUC, confusion matrix, ROC curve)
* SMOTE for class imbalance handling
* GridSearchCV for hyperparameter tuning
* FastAPI backend for inference
* React frontend with classifier and dashboard (Bun + Vite + Tailwind v4)

---

# 2. Repository Details

Repository Name: spam-email-classifier-ml

Dataset: SMS Spam Collection (UCI ML Repository) - 5,169 messages

---

# 3. Project Structure

```
spam-email-classifier-ml/
├── ml/
│   ├── data_loader.py
│   ├── eda.py
│   ├── text_processing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── naive_bayes.py
│   │   ├── knn.py
│   │   ├── svm_model.py
│   │   ├── logistic.py
│   │   └── neural_network.py
│   └── saved_models/
│       ├── best_model.pkl
│       ├── vectorizer.pkl
│       ├── scaler.pkl
│       ├── needs_scaling.pkl
│       ├── results.json
│       └── roc_curve.png
├── backend/
│   ├── main.py
│   └── predictor.py
├── frontend/
│   └── src/
│       ├── App.tsx
│       └── components/
│           ├── Dashboard.tsx
│           └── ui/
│               ├── button.tsx
│               ├── card.tsx
│               └── textarea.tsx
├── data/
│   └── spam.csv
├── Docs/
│   └── ProjectContext.md
├── .gitignore
└── README.md
```

---

# 4. Architecture

Type: Modular Monolith

Flow:
```
Frontend (React) → FastAPI Backend → ML Model (loaded for inference)
```

Notes:

* Training is separate from inference
* Backend only loads the saved model
* train.py saves results.json which the API reads for the dashboard
* Models are trained with SMOTE + GridSearchCV, best model is saved as pickle

---

# 5. Git Workflow

Main Branches:

* main (production)
* develop (active development)

Feature Branch Naming: feature/<feature-name>

---

# 6. Branching Strategy

For every new feature:

```
git checkout develop
git pull
git checkout -b feature/<feature-name>
```

After completion: create PR to develop

Release:

```
git checkout main
git merge develop
```

---

# 7. ML Pipeline

1. Data Loading (data_loader.py) - Load CSV, drop junk columns, rename, encode labels, drop nulls/duplicates
2. EDA (eda.py) - Class distribution, message length analysis, visualizations
3. Text Preprocessing (text_processing.py) - Lowercase, remove special chars, remove stopwords, lemmatization
4. Feature Extraction (feature_extraction.py) - TF-IDF vectorization with train/test split (80/20)
5. Training (train.py) - Train all 5 models before/after SMOTE + GridSearchCV
6. Evaluation (evaluate.py + metrics.py) - Accuracy, precision, recall, F1, confusion matrix, ROC/AUC

Each model file in models/ has a standardized interface:
* create_model() - returns fresh model instance
* get_param_grid() - returns hyperparameter grid for GridSearchCV
* NEEDS_SCALING - boolean flag for whether model needs scaled features

---

# 8. Frontend

Stack: Bun + Vite + React + TypeScript + Tailwind CSS v4

Two views:
* Classify tab - paste email, get spam/ham prediction with confidence
* Dashboard tab - model comparison table, ROC curve, interactive model cards, metric explainer

---

# 9. Backend

Stack: FastAPI + Uvicorn

Endpoints:
* GET / - health check
* POST /predict - classify an email message
* GET /models - return model comparison data (reads from results.json)
* /static - serves ROC curve image

---

# 10. Running the Project

Train models:
```
cd ml && python3 train.py
```

Start API:
```
cd backend && uvicorn main:app --reload
```

Start frontend:
```
cd frontend && bun install && bun run dev
```

---

# 11. Important Rules

* Do NOT train models in the API
* Backend only loads saved model for inference
* fit on train, transform on test (data leakage prevention)
* SMOTE only on training data, never test data
* MaxAbsScaler to preserve sparse matrix format

---

# 12. .gitignore

```
__pycache__/
*.pyc
*.pyo
.env
*.pkl
*.joblib
.DS_Store
*.csv
!data/spam.csv
```
