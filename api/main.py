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

predictor = None # Chargement des artéfacts

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

PREDICTION_ERRORS_TOTAL = Counter(
    "rakuten_api_prediction_errors_total",
    "Nombre total d'erreurs sur l'endpoint /predict"
)

RELOAD_ERRORS_TOTAL = Counter(
    "rakuten_api_reload_errors_total",
    "Nombre total d'erreurs sur l'endpoint /reload"
)

RELOAD_RESULTS_TOTAL = Counter(
    "rakuten_api_reload_results_total",
    "Nombre total de résultats de reload par action",
    ["action"]
)

ACTIVE_MODEL_VERSION = Gauge(
    "rakuten_api_active_model_version",
    "Version du modèle actuellement chargé"
)

MODEL_SCORE = Gauge(
    "rakuten_api_model_score",
    "Score du modèle vu par l'API",
    ["model_name", "model_version", "score_type", "status"]
)

PREDICTED_CLASS_TOTAL = Counter(
    "rakuten_api_predicted_class_total",
    "Nombre total de prédictions par classe prédite",
    ["predicted_prdtypecode"]
)

PREDICT_DURATION_SECONDS = Histogram(
    "rakuten_api_predict_duration_seconds",
    "Temps d'exécution de la prédiction",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
)

RELOAD_DURATION_SECONDS = Histogram(
    "rakuten_api_reload_duration_seconds",
    "Temps d'exécution du reload de modèle",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60)
)

MODEL_INFO_REQUESTS_TOTAL = Counter(
    "rakuten_api_model_info_requests_total",
    "Nombre total d'appels à l'endpoint /model-info"
)


@app.on_event("startup")
def load_predictor():
    global predictor
    try:
        predictor = RakutenPredictor()
    except Exception as e:
        predictor = None
        print(f"[API] Predictor non disponible au démarrage: {e}")


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
    MODEL_INFO_REQUESTS_TOTAL.inc() # Incrémenter le compteur d'appels à l'endpoint /model-info

    info = predictor.get_model_info()
    model_name = info.get("model_name", "unknown_model")
    model_version = str(info.get("model_version", "unknown"))

    ACTIVE_MODEL_INFO.labels(
        model_name=model_name,
        model_version=model_version
    ).set(1) # Met à jour le gauge avec les informations du modèle actif

    if model_version.isdigit():
        ACTIVE_MODEL_VERSION.set(int(model_version))

    return info

@app.post("/reload")
def reload_model():
    start = perf_counter()
    try:
        result = predictor.promote_test_if_better()
        RELOAD_COUNT.inc()

        action = result.get("action", "unknown")
        RELOAD_RESULTS_TOTAL.labels(action=action).inc()

        if "test_score" in result:
            test_version = str(result.get("test_version") or result.get("new_prod_version") or "unknown")
            MODEL_SCORE.labels(
                model_name="xgboost_text_tfidf",
                model_version=test_version,
                score_type="f1_weighted",
                status="test"
            ).set(float(result["test_score"]))

        if "prod_score" in result:
            prod_version = str(result.get("prod_version") or result.get("archived_version") or "unknown")
            MODEL_SCORE.labels(
                model_name="xgboost_text_tfidf",
                model_version=prod_version,
                score_type="f1_weighted",
                status="prod"
            ).set(float(result["prod_score"]))

        active_info = predictor.get_model_info()
        model_name = active_info.get("model_name", "unknown_model")
        model_version = str(active_info.get("model_version", "unknown"))

        ACTIVE_MODEL_INFO.labels(
            model_name=model_name,
            model_version=model_version
        ).set(1)

        if model_version.isdigit():
            ACTIVE_MODEL_VERSION.set(int(model_version))

        RELOAD_DURATION_SECONDS.observe(perf_counter() - start)

        return {
            "status": "success",
            "reload_result": result,
            "active_model": predictor.get_model_info()
        }
    except Exception as e:
        RELOAD_ERRORS_TOTAL.inc()
        RELOAD_DURATION_SECONDS.observe(perf_counter() - start)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = perf_counter()
    try:
        predicted_label = predictor.predict(
            designation=request.designation,
            description=request.description
        )
        PREDICTION_COUNT.inc()
        PREDICTED_CLASS_TOTAL.labels(
            predicted_prdtypecode=str(predicted_label)
        ).inc()

        info = predictor.get_model_info() or {}
        model_name = info.get("model_name", "unknown_model")
        model_version = info.get("model_version")

        ACTIVE_MODEL_INFO.labels(
            model_name=model_name,
            model_version=str(model_version) if model_version is not None else "unknown"
        ).set(1)

        if model_version is not None and str(model_version).isdigit():
            ACTIVE_MODEL_VERSION.set(int(model_version))

        PREDICT_DURATION_SECONDS.observe(perf_counter() - start)

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
        PREDICTION_ERRORS_TOTAL.inc()
        PREDICT_DURATION_SECONDS.observe(perf_counter() - start)
        raise HTTPException(status_code=500, detail=str(e))