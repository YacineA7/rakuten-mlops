from datetime import datetime

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

DOCKER_COMPOSE_CMD = f"cd {PROJECT_DIR} && docker compose"


@dag( # Définition du DAG Airflow pour la pipeline de machine learning de Rakuten
    dag_id="rakuten_ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None, 
    catchup=False,
    tags=["rakuten", "ml", "training"],
)

def rakuten_ml_pipeline():
    """
    Pipeline de machine learning pour le projet Rakuten, comprenant les étapes suivantes :
    1. Ingestion des données
    2. Entraînement du modèle
    3. Évaluation du modèle
    4. Reload du modèle dans l'API si les performances sont améliorées
    Chaque étape est implémentée en tant que tâche Airflow utilisant des scripts Python dédiés.
    """
    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"{DOCKER_COMPOSE_CMD} run --rm ingest",
    )

    train = BashOperator(
        task_id="train",
        bash_command=f"{DOCKER_COMPOSE_CMD} run --rm train",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"{DOCKER_COMPOSE_CMD} run --rm evaluate",
    )

    reload_model = BashOperator(
        task_id="reload_model",
        bash_command=f"{DOCKER_COMPOSE_CMD} run --rm reload",
    )

    ingest >> train >> evaluate >> reload_model


dag = rakuten_ml_pipeline()