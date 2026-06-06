"""
Script de reload du modèle en production.
Ce script envoie une requête à l'API pour déclencher le processus de reload du modèle en production.
"""

import json
import os
import requests

from time import perf_counter
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway

import warnings
warnings.filterwarnings("ignore")

API_URL = os.getenv("RAKUTEN_API_URL", "http://api:8000")

PROMETHEUS_PUSHGATEWAY_URL = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "pushgateway:9091")
JOB_NAME = "rakuten_reload"


def build_registry():
    registry = CollectorRegistry()

    reload_script_runs_total = Counter(
        "rakuten_reload_script_runs_total",
        "Nombre total d'exécutions du script reload",
        ["status"],
        registry=registry
    )

    reload_script_duration_seconds = Histogram(
        "rakuten_reload_script_duration_seconds",
        "Durée totale du script reload",
        buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
        registry=registry
    )

    reload_script_http_requests_total = Counter(
        "rakuten_reload_script_http_requests_total",
        "Nombre total d'appels HTTP effectués par reload_script",
        ["http_status"],
        registry=registry
    )

    reload_script_result_total = Counter(
        "rakuten_reload_script_result_total",
        "Nombre total de résultats métier du reload_script",
        ["action"],
        registry=registry
    )

    reload_script_score = Gauge(
        "rakuten_reload_script_score",
        "Scores observés par reload_script",
        ["score_type", "status"]
    , registry=registry)

    reload_script_model_version = Gauge(
        "rakuten_reload_script_model_version",
        "Version de modèle observée par reload_script",
        ["version_role"],
        registry=registry
    )

    return registry, {
        "reload_script_runs_total": reload_script_runs_total,
        "reload_script_duration_seconds": reload_script_duration_seconds,
        "reload_script_http_requests_total": reload_script_http_requests_total,
        "reload_script_result_total": reload_script_result_total,
        "reload_script_score": reload_script_score,
        "reload_script_model_version": reload_script_model_version,
    }


def main():
    start = perf_counter() # Démarre le chronomètre pour mesurer la durée totale du script de reload
    registry, metrics = build_registry() # Initialise la registry Prometheus et les métriques spécifiques au script de reload

    try:
        print("[RELOAD] Envoi de la requête de reload à l'API...")
        response = requests.post(f"{API_URL}/reload", timeout=60) # Envoie une requête POST à l'API
        metrics["reload_script_http_requests_total"].labels( # Incrémente le compteur de requêtes HTTP pour le reload_script
            http_status=str(response.status_code)
        ).inc()

        print(f"[RELOAD] Statut de la réponse HTTP : {response.status_code}") 
        response.raise_for_status() # Vérifie que la requête a réussi, sinon lève une exception avec le message d'erreur

        payload = response.json() # Récupère le contenu de la réponse au format JSON
        print("[RELOAD] Rechargement des modèles terminé !")
        print(json.dumps(payload, indent=4, ensure_ascii=False))

        reload_result = payload.get("reload_result", {}) # Récupère les résultats du reload depuis la réponse
        active_model = payload.get("active_model") or {} # Récupère les informations du modèle actif après reload

        action = reload_result.get("action", "unknown")
        metrics["reload_script_result_total"].labels(action=action).inc() 

        if "test_score" in reload_result:
            metrics["reload_script_score"].labels(
                score_type="f1_weighted",
                status="test"
            ).set(float(reload_result["test_score"]))

        if "prod_score" in reload_result:
            metrics["reload_script_score"].labels(
                score_type="f1_weighted",
                status="prod"
            ).set(float(reload_result["prod_score"]))

        if "new_prod_version" in reload_result and str(reload_result["new_prod_version"]).isdigit():
            metrics["reload_script_model_version"].labels(
                version_role="new_prod"
            ).set(int(reload_result["new_prod_version"]))

        if "prod_version" in reload_result and str(reload_result["prod_version"]).isdigit():
            metrics["reload_script_model_version"].labels(
                version_role="previous_prod"
            ).set(int(reload_result["prod_version"]))

        active_version = active_model.get("model_version")
        if active_version is not None and str(active_version).isdigit():
            metrics["reload_script_model_version"].labels(
                version_role="active_after_reload"
            ).set(int(active_version))

        metrics["reload_script_runs_total"].labels(status="success").inc() # Incrémente le compteur d'exécutions réussies du script de reload
        metrics["reload_script_duration_seconds"].observe(perf_counter() - start) # Observe la durée de l'exécution

    except Exception: # En cas d'erreur, incrémente le compteur d'exécutions échouées et observe la durée avant de push les métriques à Prometheus
        metrics["reload_script_runs_total"].labels(status="failed").inc()
        metrics["reload_script_duration_seconds"].observe(perf_counter() - start)
        try:
            push_to_gateway(PROMETHEUS_PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
        except Exception as push_error:
            print(f"[RELOAD][PROMETHEUS] Push impossible : {push_error}")

        raise

    try:
        push_to_gateway(PROMETHEUS_PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)  # Push des métriques à Prometheus via le Pushgateway après l'exécution du script, que ce soit en cas de succès ou d'échec
    except Exception as e:
        print(f"[RELOAD][PROMETHEUS] Push impossible : {e}")


if __name__ == "__main__":
    main()