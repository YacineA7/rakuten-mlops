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


@app.get("/model-info")
def model_info():
    return predictor.get_model_info()


@app.post("/reload")
def reload_model():
    try:
        result = predictor.promote_test_if_better()
        return {
            "status": "success",
            "reload_result": result,
            "active_model": predictor.get_model_info()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse) # Endpoint de prédiction : reçoit une requête avec designation et description, retourne le prdtypecode prédit
def predict(request: PredictionRequest):
    try:
        predicted_label = predictor.predict(
            designation=request.designation,
            description=request.description
        )

        info = predictor.get_model_info()

        return PredictionResponse(
            predicted_prdtypecode=predicted_label,
            model_name=f"{info['model_name']} {info['model_version']}" if info else info["model_name"],
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
