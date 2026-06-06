#!/usr/bin/env bash
set -e

echo "========================================"
echo "LANCEMENT COMPLET DU PROJET RAKUTEN"
echo "========================================"

echo
echo "[1/8] Nettoyage de l'existant"
docker compose down --remove-orphans

echo
echo "[2/8] Build des images"
docker compose build

echo
echo "[3/8] Démarrage des services socle"
docker compose up -d mlflow api gateway

echo
echo "[4/8] Vérification du status des services"
docker compose ps

echo
echo "[5/8] Tests HTTP"
echo "- Test MLflow"
curl -f http://localhost:5000 > /dev/null && echo "MLflow OK"
echo "- Test Gateway health"
curl -f http://localhost:8080/health > /dev/null && echo "Gateway health OK"

echo
echo "[6/8] Lancement du pipeline batch manuel"
docker compose run --rm ingest
docker compose run --rm train
docker compose run --rm train python scripts/bootstrap_registry.py
docker compose run --rm evaluate
docker compose exec api python scripts/reload_script.py
docker compose run --rm predict

echo
echo "[7/8] Démarrage du monitoring"
docker compose up -d prometheus

echo
echo "[8/8] Démarrage d'Airflow"
docker compose up -d airflow-postgres
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler

echo
echo "========================================"
echo "LANCEMENT TERMINE"
echo "========================================"
echo "MLflow     : http://localhost:5000"
echo "Gateway    : http://localhost:8080"
echo "Airflow    : http://localhost:8081"
echo "Prometheus : http://localhost:9090"
echo "Grafana    : http://localhost:3000 (admin/admin)"
echo "========================================"

echo
echo "Logs API en suivi (Ctrl+C pour quitter)"
docker compose logs -f api
