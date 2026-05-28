"""
API de prédiction de prdtypecode pour les produits Rakuten
Endpoints :
- GET / : Vérification de l'API
- GET /health : Vérification du statut du serveur
- GET /model-info : Informations sur le modèle actif
- POST /reload : Recharger le modèle de test si meilleur que le modèle actif
- POST /predict : Prédire le prdtypecode à partir de la designation et description du produit
"""

from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from api.schemas import PredictionRequest, PredictionResponse
from api.predictor import RakutenPredictor

app = FastAPI(
    title="Rakuten Product Classification API",
    description="API de prédiction de prdtypecode pour les produits Rakuten",
    version="1.0.0"
)

predictor = RakutenPredictor() # Chargement des artéfacts

REQUEST_COUNT = Counter(
    "rakuten_api_requests_total",
    "Nombre total de requêtes HTTP reçues",
    ["method", "endpoint", "http_status"]
    )

REQUEST_LATENCY = Histogram(
    "rakuten_api_duration_seconds",
    "Temps de réponse des requêtes HTTP en secondes",
    ["method", "endpoint"]
    )

PREDICTION_COUNT = Counter(
    "rakuten_api_predictions_total",
    "Nombre total de prédictions effectuées",
    )

RELOAD_COUNT = Counter(
    "rakuten_api_model_reloads_total",
    "Nombre total de reload déclenchés",
    )

ACTIVE_MODEL_INFO = Gauge(
    "rakuten_api_model_info",
    "Informations sur le modèle actif",
    ["model_name", "model_version"]
    )


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    method = request.method
    path = request.url.path
    start_time = perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code # Code de statut de la requête
        return response
    except Exception:
        status_code = 500
        REQUEST_COUNT.labels(method=method, endpoint=path, http_status=str(status_code)).inc() # Incrémenter le compteur de requêtes pour les erreurs
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(perf_counter() - start_time) # Observer la latence pour les erreurs
        raise
    finally:
        if "status_code" in locals():
            REQUEST_COUNT.labels(method=method, endpoint=path, http_status=str(status_code)).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(perf_counter() - start_time)


@app.get("/")
def root():
    return {"message": "API Rakuten active"}


@app.get("/health") #Vérification serveur en ligne
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST) # Retourne les métriques prometheus


@app.get("/model-info")
def model_info():
    info = predictor.get_model_info()
    model_name = info.get("model_name", "unknown_model")
    model_version = str(info.get("model_version", "unknown"))

    ACTIVE_MODEL_INFO.labels(
        model_name=model_name,
        model_version=model_version
    ).set(1) # Met à jour le gauge avec les informations du modèle actif

    return info

@app.post("/reload")
def reload_model():
    try:
        result = predictor.promote_test_if_better()
        RELOAD_COUNT.inc() # Incrémenter le compteur de reloads

        active_info = predictor.get_model_info()
        model_name = active_info.get("model_name", "unknown_model")
        model_version = str(active_info.get("model_version", "unknown"))

        ACTIVE_MODEL_INFO.labels(
            model_name=model_name,
            model_version=model_version
        ).set(1) # Met à jour le gauge avec les informations du modèle actif après reload

        return {
            "status": "success",
            "reload_result": result,
            "active_model": predictor.get_model_info()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse) # Endpoint de prédiction : reçoit une requête avec designation et description, retourne le prdtypecode prédit
def predict(request: PredictionRequest):
    """
    Endpoint de prédiction : reçoit une requête avec designation et description, retourne le prdtypecode prédit"""
    try:
        predicted_label = predictor.predict(
            designation=request.designation,
            description=request.description
        )
        PREDICTION_COUNT.inc() # Incrémenter le compteur de prédictions

        info = predictor.get_model_info() or {}
        model_name = info.get("model_name", "unknown_model")
        model_version = info.get("model_version")

        ACTIVE_MODEL_INFO.labels(
            model_name=model_name,
            model_version=str(model_version) if model_version is not None else "unknown"
        ).set(1)

        full_model_name = (
            f"{model_name} {model_version}"
            if model_version is not None
            else model_name
        )

        return PredictionResponse(
            predicted_prdtypecode=predicted_label,
            model_name=full_model_name,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))