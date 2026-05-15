from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """"Type d'objet à envoyer par la requête de l'API pour une prédiction"""
    designation: str = Field(..., example="Peluche pokemon pikachu")
    description: str = Field(..., example="Peluche en tissu pour enfant")


class PredictionResponse(BaseModel):
    """Type d'objet de la réponse de l'API pour une prédiction"""
    predicted_prdtypecode: int
    model_name: str
    status: str