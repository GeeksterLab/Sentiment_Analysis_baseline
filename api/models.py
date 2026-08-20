"""
- POST /predict         → prediction for 1 phrase
- POST /predict-batch   → prediction for several phrases
- POST /upload          → CSV upload / PDF

"""

# ╔════════════════════════════════════════════════════════════╗
# ║ 🚚 IMPORTS
# ╚════════════════════════════════════════════════════════════╝

from fastapi import UploadFile, File

# ╔════════════════════════════════════════════════════════════╗
# ║ 🌐 API
# ╚════════════════════════════════════════════════════════════╝
from fastapi import APIRouter

text_predict = APIRouter()
text_predict_batch = APIRouter()
text_upload_csv = APIRouter()

# ╔════════════════════════════════════════════════════════════╗
# ║ 🛣️ ROUTERS
# ╚════════════════════════════════════════════════════════════╝
from api.schemas import (
    TextInput,
    PredictSentiment,
    # TextBatchInput,
)

from api.utils import predict_sentiment
from io import StringIO

import csv


# ── PREDICT ──────────────────────────────────────────────────
@text_predict.post("/predict", response_model=PredictSentiment, tags=["Predict"])
def predict_text(
    input: TextInput,
) -> PredictSentiment:

    result = predict_sentiment(input.text)

    prediction = int(result["prediction"])
    probability = float(result["probability"])
    confiance = float(result["confiance"])

    return PredictSentiment(
        prediction=prediction,
        label=(
            "Negative"
            if prediction == 0
            else "Neutral" if prediction == 1 else "Positive"
        ),
        probability=round(probability, 2),
        confiance=confiance,
    )


# # ── PREDICT BATCH ──────────────────────────────────────────────────
# @text_predict_batch.post("/predict-batch", tags=["PredictBatch"])
# def predict_batch_text(
#     data: TextBatchInput,
#     request: Request,
# ) -> list[PredictSentiment]:

#     model = request.app.state.model

#     results = []

#     for item in data.data:
#         result = predict_sentiment(item.text, model)

#         prediction = int(model["prediction"])
#         probability = float(model["probability"])
#         confiance = float(model["confiance"])

#         results.append(
#             PredictSentiment(
#                 prediction=prediction,
#                 label=(
#                     "Negative"
#                     if prediction == 0
#                     else "Neutral" if prediction == 1 else "Positive"
#                 ),
#                 probability=round(probability, 2),
#                 confiance=confiance,
#             )
#         )

#     return results


# ── UPLOAD ──────────────────────────────────────────────────
@text_upload_csv.post("/upload-csv", tags=["UploadCSV"])
async def upload_csv(file: UploadFile = File(...)):

    data = []

    # Read file as bytes and decode bytes info text stream
    file_bytes = await file.read()
    buffer = StringIO(file_bytes.decode("utf-8"))

    # Process CSV
    csvReader = csv.DictReader(buffer)

    # Close buffer and file
    buffer.close()
    await file.close()

    # Return JSON
    return data
