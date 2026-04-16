from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from predictor import predict_spam
import os
import json

app = FastAPI(title="Spam Email Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins (localhost + deployed frontend)
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve saved_models folder so frontend can load the ROC curve image
saved_models_dir = os.path.join(os.path.dirname(__file__), '..', 'ml', 'saved_models')
if os.path.exists(saved_models_dir):
    app.mount("/static", StaticFiles(directory=saved_models_dir), name="static")

# model descriptions — static info about each model (doesn't change with retraining)
MODEL_INFO = {
    "Naive Bayes": {
        "type": "Probabilistic",
        "icon": "brain",
        "color": "#8b5cf6",
        "description": "Calculates the probability of an email being spam based on word frequencies. Fast and simple — the go-to baseline for text classification.",
        "how_it_works": "Looks at each word independently and asks: 'How likely is this word to appear in spam vs ham?' Multiplies all word probabilities together to get a final spam score.",
        "strengths": ["Very fast to train", "Works well with small datasets", "Great baseline model"],
        "weaknesses": ["Assumes words are independent", "Can't capture word combinations"],
    },
    "KNN": {
        "type": "Distance-Based",
        "icon": "users",
        "color": "#06b6d4",
        "description": "Classifies emails by finding the most similar emails in the training data. Like asking your neighbors — if most similar emails are spam, this one probably is too.",
        "how_it_works": "Converts each email into numbers (TF-IDF), then measures the distance to all training emails. The K closest emails 'vote' — majority wins.",
        "strengths": ["Simple to understand", "No training phase needed", "Works with any data type"],
        "weaknesses": ["Slow on large datasets", "Sensitive to irrelevant features", "Struggles with high dimensions"],
    },
    "SVM": {
        "type": "Boundary-Based",
        "icon": "separator-horizontal",
        "color": "#f59e0b",
        "description": "Finds the best boundary line that separates spam from ham. Maximizes the gap between the two classes for the clearest separation possible.",
        "how_it_works": "Imagine plotting all emails as points in space. SVM draws a line (or hyperplane) that best separates spam points from ham points, maximizing the margin between them.",
        "strengths": ["Great with high-dimensional data", "Effective on text classification", "Handles clear margins well"],
        "weaknesses": ["Slow to train on large datasets", "Hard to interpret results", "Needs feature scaling"],
    },
    "Logistic Regression": {
        "type": "Statistical",
        "icon": "trending-up",
        "color": "#10b981",
        "description": "Uses a mathematical function (sigmoid) to calculate spam probability between 0% and 100%. Despite the name, it's a classification algorithm, not regression.",
        "how_it_works": "Assigns a weight to each word. Words like 'free' and 'winner' get high positive weights (spam signals). Sums all weights and passes through a sigmoid function to get a probability.",
        "strengths": ["Fast and efficient", "Gives probability scores", "Easy to interpret which words matter"],
        "weaknesses": ["Assumes linear relationship", "Can underfit complex patterns"],
    },
    "Neural Network": {
        "type": "Deep Learning",
        "icon": "network",
        "color": "#ef4444",
        "description": "Inspired by the human brain — layers of connected neurons that learn complex patterns. The most powerful model but also the most complex.",
        "how_it_works": "Data passes through layers of neurons. Each neuron takes inputs, applies weights, and passes through an activation function. The network adjusts weights through backpropagation until it learns to distinguish spam from ham.",
        "strengths": ["Learns complex patterns", "Best overall performance", "Adapts to any data type"],
        "weaknesses": ["Slower to train", "Needs more data", "Hard to interpret (black box)"],
    }
}


class EmailRequest(BaseModel):
    message: str


class EmailResponse(BaseModel):
    prediction: str
    is_spam: bool
    confidence: float | None
    processed_text: str


@app.get("/")
def root():
    return {"message": "Spam Email Classifier API is running"}


@app.post("/predict", response_model=EmailResponse)
def predict(request: EmailRequest):
    result = predict_spam(request.message)
    return result


@app.get("/models")
def get_models():
    # load dynamic results from the json saved by train.py
    results_path = os.path.join(saved_models_dir, 'results.json')

    if not os.path.exists(results_path):
        return {"error": "No training results found. Run train.py first."}

    with open(results_path, 'r') as f:
        training_data = json.load(f)

    # merge static model info with dynamic training metrics
    models_list = []
    for name, info in MODEL_INFO.items():
        metrics = training_data['results'].get(name, {})
        models_list.append({
            "name": name,
            **info,
            "metrics": metrics
        })

    return {
        "models": models_list,
        "best_model": training_data['best_model'],
        "dataset_info": {
            "total_samples": 5169,
            "spam_count": 653,
            "ham_count": 4516,
            "spam_percentage": 12.64,
            "ham_percentage": 87.36
        }
    }
