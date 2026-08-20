"""
Listing de tous les types de variables qui entrent et sortent de l'API.
"""

# ╔════════════════════════════════════════════════════════════╗
# ║ 🚚 IMPORTS
# ╚════════════════════════════════════════════════════════════╝
from typing import List
from pydantic import BaseModel


# ╔════════════════════════════════════════════════════════════╗
# ║ 📝 SCHEMAS
# ╚════════════════════════════════════════════════════════════╝
class TextInput(BaseModel):
    """
    Prediction request structure.
    """

    text: str


class PredictSentiment(BaseModel):
    """
    Prediction response structure.
    """

    prediction: int
    label: str
    probability: float
    confiance: float


class TextBatchInput(BaseModel):
    """
    Batch prediction request structure.
    """

    data: List[TextInput]
