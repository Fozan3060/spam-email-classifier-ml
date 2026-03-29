from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predictor import predict_spam

app = FastAPI(title="Spam Email Classifier API")

# allow react frontend to call this api (cors)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# request body shape — what the frontend sends
class EmailRequest(BaseModel):
    message: str


# response body shape — what we send back
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
    # take the message from request body, run prediction, return result
    result = predict_spam(request.message)
    return result
