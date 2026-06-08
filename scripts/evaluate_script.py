"""
Ce script est responsable de l'évaluation du modèle XGBoost sur l'ensemble de validation préparé par ingest_script.py. 
Il charge les artefacts nécessaires, calcule les métriques d'évaluation (accuracy, f1_macro, f1_weighted), 
construit un rapport de classification détaillé et une matrice de confusion, puis sauvegarde tous ces résultats dans le dossier reports/.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import os
from time import perf_counter
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway

import warnings
warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROMETHEUS_PUSHGATEWAY_URL = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "pushgateway:9091")
JOB_NAME = "rakuten_evaluate"

X_VALID_PATH = ARTIFACTS_DIR / "X_valid.npz"
Y_VALID_PATH = ARTIFACTS_DIR / "y_valid.npy"

MODEL_PATH = MODEL_DIR / "xgb_model.joblib"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"


def load_eval_artifacts():
    """
    Charge tout ce qui est nécessaire à l'évaluation :
    - X_valid : matrice TF-IDF sparse préparée par ingest_script
    - y_valid : labels encodés de validation
    - model   : modèle XGBoost entraîné
    - label_encoder : utile pour retrouver les vrais codes de classes
    """
    X_valid = sparse.load_npz(X_VALID_PATH)
    y_valid = np.load(Y_VALID_PATH)

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    print(f"[EVAL] X_valid chargé : {X_valid.shape}")
    print(f"[EVAL] y_valid chargé : {y_valid.shape}")
    print(f"[EVAL] Modèle chargé : {MODEL_PATH}")
    print(f"[EVAL] LabelEncoder chargé : {LABEL_ENCODER_PATH}")

    return X_valid, y_valid, model, label_encoder


def predict_validation(model, X_valid):
    """
    Applique le modèle sur l'ensemble de validation.
    Avec XGBoost configuré en multi:softmax, predict() renvoie directement
    les classes encodées (0 à 26), pas des probabilités.
    """
    y_pred = model.predict(X_valid)

    print(f"[EVAL] Prédictions générées : {len(y_pred)}")
    return y_pred


def evaluate_model(y_valid, y_pred):
    """
    Calcule les métriques principales utilisées dans le projet :
    - accuracy
    - f1_macro
    - f1_weighted

    f1_weighted est la métrique la plus importante ici car le dataset
    Rakuten contient 27 classes très déséquilibrées.
    """
    accuracy = accuracy_score(y_valid, y_pred)
    f1_macro = f1_score(y_valid, y_pred, average="macro")
    f1_weighted = f1_score(y_valid, y_pred, average="weighted")

    metrics = {
        "accuracy": round(float(accuracy), 6),
        "f1_macro": round(float(f1_macro), 6),
        "f1_weighted": round(float(f1_weighted), 6)
    }

    print(f"[EVAL] Accuracy     : {metrics['accuracy']}")
    print(f"[EVAL] F1 macro     : {metrics['f1_macro']}")
    print(f"[EVAL] F1 weighted  : {metrics['f1_weighted']}")

    return metrics


def build_classification_report(y_valid, y_pred, label_encoder):
    """
    Construit un classification_report détaillé pour chaque classe.

    output_dict=True permet d'obtenir un dictionnaire Python au lieu d'un texte,
    ce qui facilite la sauvegarde en JSON.

    target_names permet d'afficher les vrais codes Rakuten (10, 40, 50, ...)
    au lieu des labels encodés 0, 1, 2...
    """
    target_names = [str(label) for label in label_encoder.classes_]

    report = classification_report(
        y_valid,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    print("[EVAL] Classification report construit.")
    return report


def build_confusion_matrix(y_valid, y_pred, label_encoder):
    """
    Construit la matrice de confusion avec les vraies classes Rakuten
    comme index et colonnes.
    """
    cm = confusion_matrix(y_valid, y_pred)

    class_labels = [str(label) for label in label_encoder.classes_]

    cm_df = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    print(f"[EVAL] Matrice de confusion calculée : {cm_df.shape}")
    return cm_df


def save_evaluation_outputs(metrics, report, cm_df):
    """
    Sauvegarde tous les résultats d'évaluation :
    - metrics JSON pour pilotage simple
    - classification_report JSON pour détail par classe
    - confusion_matrix CSV pour analyse métier ou visualisation
    """

    # Sauvegarde des métriques globales
    with open(REPORTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Sauvegarde du rapport détaillé
    with open(REPORTS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Sauvegarde de la matrice de confusion
    cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv", encoding="utf-8")

    print(f"[EVAL] Résultats sauvegardés dans : {REPORTS_DIR}/")
    print("[EVAL]   → evaluation_metrics.json")
    print("[EVAL]   → classification_report.json")
    print("[EVAL]   → confusion_matrix.csv")


def build_registry():
    registry = CollectorRegistry()

    evaluate_runs_total = Counter(
        "rakuten_evaluate_runs_total",
        "Nombre total d'exécutions du script evaluate",
        ["status"],
        registry=registry
    )

    evaluate_duration_seconds = Histogram(
        "rakuten_evaluate_duration_seconds",
        "Durée totale du script evaluate",
        buckets=(1, 5, 10, 30, 60, 120, 300, 600),
        registry=registry
    )

    evaluate_validation_samples = Gauge(
        "rakuten_evaluate_validation_samples",
        "Nombre d'échantillons évalués",
        registry=registry
    )

    evaluate_predictions_total = Gauge(
        "rakuten_evaluate_predictions_total",
        "Nombre total de prédictions générées pendant l'évaluation",
        registry=registry
    )

    evaluate_score = Gauge(
        "rakuten_evaluate_score",
        "Scores calculés par evaluate_script",
        ["score_type"],
        registry=registry
    )

    evaluate_artifacts_written_total = Counter(
        "rakuten_evaluate_artifacts_written_total",
        "Nombre d'artefacts écrits par evaluate_script",
        ["artifact_type"],
        registry=registry
    )

    return registry, {
        "evaluate_runs_total": evaluate_runs_total,
        "evaluate_duration_seconds": evaluate_duration_seconds,
        "evaluate_validation_samples": evaluate_validation_samples,
        "evaluate_predictions_total": evaluate_predictions_total,
        "evaluate_score": evaluate_score,
        "evaluate_artifacts_written_total": evaluate_artifacts_written_total,
    }


def main():
    start = perf_counter() # Démarre le chronomètre pour mesurer la durée totale de l'évaluation
    registry, metrics_registry = build_registry() # Initialise la registry Prometheus et les métriques spécifiques au script d'évaluation

    try:
        print("=" * 60)
        print("[EVAL] Démarrage de l'évaluation")
        print("=" * 60)

        X_valid, y_valid, model, label_encoder = load_eval_artifacts() # Chargement des artefacts nécessaires à l'évaluation
        metrics_registry["evaluate_validation_samples"].set(int(X_valid.shape[0]))

        y_pred = predict_validation(model, X_valid) # Génération des prédictions sur l'ensemble de validation
        metrics_registry["evaluate_predictions_total"].set(int(len(y_pred)))

        metrics = evaluate_model(y_valid, y_pred) # Calcul des métriques d'évaluation
        metrics_registry["evaluate_score"].labels(score_type="accuracy").set(float(metrics["accuracy"]))
        metrics_registry["evaluate_score"].labels(score_type="f1_macro").set(float(metrics["f1_macro"]))
        metrics_registry["evaluate_score"].labels(score_type="f1_weighted").set(float(metrics["f1_weighted"]))

        report = build_classification_report(y_valid, y_pred, label_encoder) # Construction du rapport de classification détaillé par classe
        cm_df = build_confusion_matrix(y_valid, y_pred, label_encoder) # Construction de la matrice de confusion

        save_evaluation_outputs(metrics, report, cm_df) # Sauvegarde de tous les résultats d'évaluation (métriques globales, rapport détaillé, matrice de confusion)
        metrics_registry["evaluate_artifacts_written_total"].labels(artifact_type="evaluation_metrics_json").inc()
        metrics_registry["evaluate_artifacts_written_total"].labels(artifact_type="classification_report_json").inc()
        metrics_registry["evaluate_artifacts_written_total"].labels(artifact_type="confusion_matrix_csv").inc()

        metrics_registry["evaluate_runs_total"].labels(status="success").inc()
        metrics_registry["evaluate_duration_seconds"].observe(perf_counter() - start)

    except Exception: # En cas d'erreur, incrémente le compteur d'exécutions échouées et observe la durée avant de push les métriques à Prometheus
        metrics_registry["evaluate_runs_total"].labels(status="failed").inc()
        metrics_registry["evaluate_duration_seconds"].observe(perf_counter() - start)
        try:
            push_to_gateway(PROMETHEUS_PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
        except Exception as push_error:
            print(f"[EVAL][PROMETHEUS] Push impossible : {push_error}")
            
        raise

    try:
        push_to_gateway(PROMETHEUS_PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)  # Push des métriques à Prometheus via le Pushgateway après l'exécution du script, que ce soit en cas de succès ou d'échec
    except Exception as e:
        print(f"[EVAL][PROMETHEUS] Push impossible : {e}")


if __name__ == "__main__":
    main()