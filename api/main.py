from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from api.predictor import RakutenPredictor

app = FastAPI(
    title="Rakuten Product Classification API",
    description="API de prédiction de prdtypecode pour les produits Rakuten",
    version="1.0.0"
)

predictor = RakutenPredictor() # Chargement des artéfacts


@app.get("/")
def root():
    return {"message": "API Rakuten active"}


@app.get("/health") #Vérification serveur en ligne
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse) # Endpoint de prédiction : reçoit une requête avec designation et description, retourne le prdtypecode prédit
def predict(request: PredictionRequest):
    try:
        predicted_label = predictor.predict(
            designation=request.designation,
            description=request.description
        )

        return PredictionResponse(
            predicted_prdtypecode=predicted_label,
            model_name=predictor.model_name,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
