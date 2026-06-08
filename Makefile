COMPOSE = docker compose
AIRFLOW = $(COMPOSE) exec airflow-webserver airflow
API = http://localhost:8080

.PHONY: help build up down restart logs ps ingest train evaluate reload pipeline airflow-ui mlflow-ui grafana-ui prometheus-ui api-info api-metrics predict clean test

help:
	@echo "Commandes disponibles:"
	@echo "  make build        - Build les images"
	@echo "  make up           - Démarre toute la stack"
	@echo "  make down         - Arrête toute la stack"
	@echo "  make restart      - Redémarre la stack"
	@echo "  make logs         - Suit les logs"
	@echo "  make ps           - Liste les services"
	@echo "  make ingest       - Lance l'ingestion"
	@echo "  make train        - Lance l'entraînement"
	@echo "  make evaluate     - Lance l'évaluation"
	@echo "  make reload       - Recharge le modèle dans l'API"
	@echo "  make pipeline     - Lance ingest -> train -> evaluate -> reload"
	@echo "  make airflow-ui   - Ouvre Airflow (http://localhost:8081)"
	@echo "  make mlflow-ui    - Ouvre MLflow (http://localhost:5000)"
	@echo "  make grafana-ui   - Ouvre Grafana (http://localhost:3000)"
	@echo "  make prometheus-ui- Ouvre Prometheus (http://localhost:9090)"
	@echo "  make api-info     - Teste GET /model-info via le gateway"
	@echo "  make api-metrics  - Teste GET /metrics via le gateway"
	@echo "  make predict      - Exemple POST /predict via le gateway"
	@echo "  make clean        - Stoppe et supprime volumes orphelins"
	@echo "  make test         - Lance les tests unitaires avec pytest"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart: down up

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

ingest:
	$(COMPOSE) run --rm ingest

train:
	$(COMPOSE) run --rm train

evaluate:
	$(COMPOSE) run --rm evaluate

reload:
	$(COMPOSE) run --rm reload

pipeline:
	$(MAKE) ingest
	$(MAKE) train
	$(MAKE) evaluate
	$(MAKE) reload

test:
	pytest tests/

airflow-ui:
	@echo "Airflow: http://localhost:8081 (admin/admin)"

mlflow-ui:
	@echo "MLflow: http://localhost:5000"

grafana-ui:
	@echo "Grafana: http://localhost:3000 (admin/admin)"

prometheus-ui:
	@echo "Prometheus: http://localhost:9090"

api-info:
	curl -u admin:admin $(API)/model-info

api-metrics:
	curl -u admin:admin $(API)/metrics

predict:
	curl -u admin:admin -X POST $(API)/predict \
	  -H 'Content-Type: application/json' \
	  -d '{"designation":"Lot de 2 mugs en céramique","description":"Mugs blancs pour café et thé, contenance 30cl"}'

clean:
	$(COMPOSE) down -v --remove-orphans