# Spam Email Classifier

Full-stack machine learning system that classifies emails as spam or not spam using 5 classification models with performance comparison, real-time predictions via FastAPI, and a React dashboard.

## Tech Stack

**ML Pipeline:** Python, scikit-learn, NLTK, pandas, matplotlib, SMOTE (imbalanced-learn)

**Backend:** FastAPI, Uvicorn, joblib

**Frontend:** React, TypeScript, Bun, Vite, Tailwind CSS v4, shadcn-style components

## Models

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| Neural Network | 97.9% | 92.0% | 98.9% |
| Logistic Regression | 97.0% | 89.2% | 98.9% |
| SVM | 97.6% | 90.8% | 98.4% |
| Naive Bayes | 96.7% | 88.4% | 98.3% |
| KNN | 94.7% | 76.6% | 81.0% |

## Project Structure

```
spam-email-classifier-ml/
├── ml/
│   ├── data_loader.py          # load and clean dataset
│   ├── eda.py                  # exploratory data analysis
│   ├── text_processing.py      # NLP preprocessing pipeline
│   ├── feature_extraction.py   # TF-IDF vectorization
│   ├── train.py                # orchestrator - trains all models
│   ├── evaluate.py             # model evaluation logic
│   ├── metrics.py              # metric calculations
│   ├── models/                 # one file per model
│   │   ├── naive_bayes.py
│   │   ├── knn.py
│   │   ├── svm_model.py
│   │   ├── logistic.py
│   │   └── neural_network.py
│   └── saved_models/           # saved model, vectorizer, scaler, results
├── api/
│   ├── main.py                 # FastAPI endpoints
│   └── predictor.py            # loads model and predicts
├── frontend/
│   └── src/
│       ├── App.tsx             # classifier + navigation
│       └── components/
│           └── Dashboard.tsx   # model comparison dashboard
└── data/
    └── spam.csv                # SMS Spam Collection dataset
```

## ML Pipeline

1. **Data Loading** - Load SMS Spam Collection dataset, clean columns, encode labels
2. **EDA** - Class distribution analysis, message length analysis, visualizations
3. **Text Preprocessing** - Lowercase, remove special characters, remove stopwords, lemmatization
4. **Feature Extraction** - TF-IDF vectorization (fit on train, transform on test to prevent data leakage)
5. **Training** - Train all 5 models with SMOTE (class imbalance handling) and GridSearchCV (hyperparameter tuning)
6. **Evaluation** - Accuracy, precision, recall, F1, confusion matrix, ROC/AUC curve

## Features

- Real-time email classification with confidence scores
- Model comparison dashboard with interactive model cards
- ROC curve comparison across all models
- Metric explainer for non-technical users
- Dynamic results that update when models are retrained

## Getting Started

### Train the models
```bash
cd ml
python3 train.py
```

### Start the API
```bash
cd api
uvicorn main:app --reload
```

### Start the frontend
```bash
cd frontend
bun install
bun run dev
```

Open http://localhost:5173 to use the app.

## Dataset

SMS Spam Collection from UCI ML Repository - 5,169 messages (87% ham, 13% spam).
