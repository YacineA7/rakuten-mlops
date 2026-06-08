# 🛒 Pipeline MLOps — Classification E-Commerce Rakuten

Pipeline MLOps complet pour la **classification de produits e-commerce en 27 catégories** à partir de descriptions textuelles, avec orchestration Airflow, suivi MLflow, exposition FastAPI et monitoring Prometheus/Grafana.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-XGBoost-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-green.svg)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.10-red.svg)](https://airflow.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Quick Start](#-quick-start-5-minutes)
- [Architecture](#-architecture)
- [Services disponibles](#-services-disponibles)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [DAG Airflow](#-dag-airflow--pipeline-automatisé)
- [API FastAPI](#-api-fastapi)
- [Monitoring](#-monitoring--prometheus--grafana)
- [MLflow Tracking](#-mlflow-tracking)
- [Structure du projet](#-structure-du-projet)
- [Variables d'environnement](#-variables-denvironnement)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Vue d'ensemble

Ce projet implémente un pipeline MLOps complet pour classifier automatiquement des produits e-commerce Rakuten en **27 catégories** à partir de leurs descriptions textuelles.

Le pipeline repose sur une approche **NLP + Machine Learning** :
- **Prétraitement NLP** → TF-IDF sur les descriptions nettoyées (titre + désignation)
- **Modèle** → XGBoost entraîné sur les features TF-IDF
- **Tracking** → MLflow pour le suivi des expériences et la comparaison des runs
- **Orchestration** → Airflow déclenche les 4 étapes (ingest → train → evaluate → reload)
- **API** → FastAPI expose la prédiction en temps réel
- **Monitoring** → Prometheus + Grafana pour l'observabilité du pipeline

### 📊 Données

| Jeu de données | Taille | Description |
|----------------|--------|-------------|
| **Train** | ~84 916 produits | Descriptions textuelles + labels (27 classes) |
| **Classes** | 27 catégories | Ex : Livres, Jeux vidéo, Mode, Électronique… |
| **Features** | TF-IDF | `TFIDF_MAX_FEATURES=50 000` par défaut |
| **Source** | Challenge ENS | [challengedata.ens.fr](https://challengedata.ens.fr/participants/challenges/35/) |

---

## ⚡ Quick Start (5 minutes)

```bash
# 1. Cloner le dépôt
git clone https://github.com/YacineA7/Rakuten-MLE-DEC25.git
cd Rakuten-MLE-DEC25

# 2. Placer les données brutes
# Télécharger depuis : https://challengedata.ens.fr/participants/challenges/35/
# Placer dans : data/raw/

# 3. Corriger les permissions
sudo chown -R $(id -u):$(id -g) mlruns models artifacts logs data

# 4. Démarrer la stack complète
docker compose up -d

# 5. Accéder aux interfaces
# - API FastAPI :  http://localhost:80        (via Gateway Nginx)
# - MLflow :       http://localhost:5000
# - Airflow :      http://localhost:8081      (admin / admin)
# - Grafana :      http://localhost:3000      (admin / admin)
# - Prometheus :   http://localhost:9090
# - Pushgateway :  http://localhost:9091
```

### 🚀 Lancer le pipeline ML complet (via Airflow)

```bash
# Ouvrir Airflow : http://localhost:8081
# → Activer le DAG "rakuten_ml_pipeline"
# → Trigger DAG ▶

# Ou directement via Docker Compose :
docker compose run --rm ingest
docker compose run --rm train
docker compose run --rm evaluate
docker compose run --rm reload
```

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client / Navigateur                          │
└───────────────────┬────────────────────────────┬────────────────────┘
                    │                            │
                    ▼                            ▼
        ┌─────────────────────┐      ┌─────────────────────┐
        │  Gateway Nginx :80  │      │   Grafana :3000     │
        │  /auth /predict     │      │   Dashboards ML     │
        │  /train /reload     │      └──────────┬──────────┘
        └──────────┬──────────┘                 │
                   │                            ▼
                   ▼                 ┌─────────────────────┐
        ┌─────────────────────┐      │  Prometheus :9090   │
        │  API FastAPI :8000  │◄─────┤  Scrape métriques   │
        │  /predict           │      └──────────┬──────────┘
        │  /info              │                 │
        │  /reload            │                 ▼
        └──────────┬──────────┘      ┌─────────────────────┐
                   │                 │  Pushgateway :9091  │
                   │                 │  ← Push jobs ML     │
                   │                 └─────────────────────┘
                   │                          ▲
┌──────────────────┼──────────────────────────┼──────────────────────┐
│  Phase 2 – Orchestration Airflow             │                      │
│                  │                           │                      │
│   ┌──────────────▼─────────────────────┐    │                      │
│   │      Airflow Webserver :8081        │    │                      │
│   │      DAG : rakuten_ml_pipeline      │    │                      │
│   │                                     │    │                      │
│   │  ┌────────┐  ┌───────┐  ┌────────┐ │    │                      │
│   │  │ ingest │→ │ train │→ │evaluate│ │    │                      │
│   │  └────────┘  └───┬───┘  └───┬────┘ │    │                      │
│   │                  │          │       │    │                      │
│   │              ┌───▼──────────▼────┐  │    │                      │
│   │              │  MLflow :5000     │  │    │                      │
│   │              │  Tracking runs    │  │    │                      │
│   │              └───────────────────┘  │    │                      │
│   │                                     │    │                      │
│   │  ┌──────────────────────────────┐   │    │                      │
│   │  │ reload → POST /reload → API  │───┼────┘                      │
│   │  └──────────────────────────────┘   │                           │
│   └─────────────────────────────────────┘                           │
│                                                                      │
│   Données : Volume Docker (/data, /artifacts, /models)               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Services disponibles

| Service | Container | Description | Port |
|---------|-----------|-------------|------|
| **Gateway** | `rakuten-gateway` | Nginx reverse proxy + auth HTTP | `80` |
| **API** | `rakuten-api` | FastAPI — inférence + /reload | `8000` (interne) |
| **MLflow** | `rakuten-mlflow` | Tracking expériences ML | `5000` |
| **Airflow Webserver** | `airflow-webserver` | Interface DAG + monitoring | `8081` |
| **Airflow Scheduler** | `airflow-scheduler` | Planificateur de DAGs | — |
| **Airflow Init** | `airflow-init` | Migration DB + création admin | — |
| **Airflow Postgres** | `airflow-postgres` | Base de données Airflow | — |
| **Ingest** | `rakuten-ingest` | Ingestion et préparation des données | — |
| **Train** | `rakuten-train` | Entraînement XGBoost + push métriques | — |
| **Evaluate** | `rakuten-evaluate` | Évaluation + rapport de performance | — |
| **Predict** | `rakuten-predict` | Inférence batch sur le test set | — |
| **Reload** | `rakuten-reload` | Reload du modèle dans l'API | — |
| **Prometheus** | `rakuten-prometheus` | Collecte métriques | `9090` |
| **Pushgateway** | `rakuten-pushgateway` | Réception métriques batch jobs | `9091` |
| **Grafana** | `rakuten-grafana` | Dashboards de monitoring | `3000` |

---

## 🔧 Installation

### Prérequis

- **Docker** ≥ 24
- **Docker Compose** ≥ 2.x (plugin `docker compose`)
- **RAM** : 8 Go minimum recommandés
- **Espace disque** : ~5 Go (données + images Docker)

### Étapes d'installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/YacineA7/Rakuten-MLE-DEC25.git
cd Rakuten-MLE-DEC25

# 2. Créer les répertoires attendus
mkdir -p data artifacts models predictions reports mlruns logs \
         airflow/logs airflow/plugins airflow/config

# 3. Préparer les données brutes (depuis le challenge ENS)
# Placer X_train_update.csv, Y_train_CVw08PX.csv dans data/

# 4. Ajuster les permissions pour Airflow (UID 50000)
sudo chown -R $(id -u):$(id -g) .
echo "AIRFLOW_UID=$(id -u)" >> .env

# 5. Construire et démarrer la stack
docker compose build
docker compose up -d

# 6. Vérifier que tous les services sont actifs
docker compose ps
```

### Vérification rapide

```bash
# API accessible ?
curl http://localhost/info

# MLflow accessible ?
curl http://localhost:5000

# Airflow accessible ?
curl http://localhost:8081/health
```

---

## 📚 Utilisation

### Pipeline pas-à-pas via Docker Compose

```bash
# Étape 1 — Ingestion des données
docker compose run --rm ingest
# → Nettoie et prépare les données brutes dans /artifacts

# Étape 2 — Entraînement
docker compose run --rm train
# → Entraîne le modèle XGBoost, push métriques vers Pushgateway
# → Enregistre le run dans MLflow (http://localhost:5000)

# Étape 3 — Évaluation
docker compose run --rm evaluate
# → Calcule accuracy, F1-score, matrice de confusion
# → Génère un rapport dans /reports

# Étape 4 — Reload modèle dans l'API
docker compose run --rm reload
# → Appelle POST http://api:8000/reload
# → Le nouveau modèle est chargé à chaud sans redémarrage
```

### Commandes de gestion

```bash
# Voir les logs d'un service
docker compose logs -f train
docker compose logs -f api

# Statut des containers
docker compose ps

# Arrêter la stack
docker compose down

# Rebuild et restart
docker compose down && docker compose up -d --build

# Nettoyage complet
docker compose down -v --remove-orphans
docker system prune -f
```

---

## 🛠️ Makefile — Référence des commandes

```bash
# Afficher toutes les commandes disponibles
make help
```

### 🔄 Pipeline ML

```bash
# ── Étapes individuelles ────────────────────────────────────────
make ingest          # Ingestion et nettoyage des données
make train           # Entraînement XGBoost + MLflow + push métriques
make evaluate        # Évaluation du modèle + rapport de performance
make predict         # Inférence batch sur le test set
make reload          # Reload du modèle dans l'API (POST /reload)

# ── Pipeline complet (séquentiel) ───────────────────────────────
make pipeline        # ingest → train → evaluate → reload
```

### 🐳 Stack Docker

```bash
# ── Construction ────────────────────────────────────────────────
make build           # Construire toutes les images Docker
make build-api       # Construire uniquement l'image API
make build-train     # Construire uniquement l'image train

# ── Démarrage / Arrêt ───────────────────────────────────────────
make up              # Démarrer la stack complète (docker compose up -d)
make down            # Arrêter tous les services
make restart         # Redémarrer la stack (down + up)

# ── Logs et statut ──────────────────────────────────────────────
make status          # Statut des containers (docker compose ps)
make logs            # Logs de tous les services
make logs-api        # Logs de l'API FastAPI
make logs-train      # Logs du container train
make logs-airflow    # Logs de l'Airflow webserver
```

### 📊 Monitoring

```bash
make monitoring      # Démarrer Prometheus + Grafana + Pushgateway
```

### 🧹 Nettoyage

```bash
make clean           # Arrêter les services + supprimer volumes/images locales
make prune           # Nettoyage complet des ressources Docker inutilisées
```

### 💡 Workflow recommandé

```bash
# 1. Première installation
make build && make up

# 2. Lancer le pipeline ML complet
make pipeline

# 3. Vérifier les résultats
make logs-train      # Voir les métriques d'entraînement
# → MLflow : http://localhost:5000
# → Grafana : http://localhost:3000

# 4. Tester l'API
curl -X POST http://localhost/predict \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d '{"designation": "Harry Potter", "description": "Roman jeunesse"}'
```## 🛠️ Makefile — Référence des commandes

```bash
# Afficher toutes les commandes disponibles
make help
```

### 🔄 Pipeline ML

```bash
# ── Étapes individuelles ────────────────────────────────────────
make ingest          # Ingestion et nettoyage des données
make train           # Entraînement XGBoost + MLflow + push métriques
make evaluate        # Évaluation du modèle + rapport de performance
make predict         # Inférence batch sur le test set
make reload          # Reload du modèle dans l'API (POST /reload)

# ── Pipeline complet (séquentiel) ───────────────────────────────
make pipeline        # ingest → train → evaluate → reload
```

### 🐳 Stack Docker

```bash
# ── Construction ────────────────────────────────────────────────
make build           # Construire toutes les images Docker
make build-api       # Construire uniquement l'image API
make build-train     # Construire uniquement l'image train

# ── Démarrage / Arrêt ───────────────────────────────────────────
make up              # Démarrer la stack complète (docker compose up -d)
make down            # Arrêter tous les services
make restart         # Redémarrer la stack (down + up)

# ── Logs et statut ──────────────────────────────────────────────
make status          # Statut des containers (docker compose ps)
make logs            # Logs de tous les services
make logs-api        # Logs de l'API FastAPI
make logs-train      # Logs du container train
make logs-airflow    # Logs de l'Airflow webserver
```

### 📊 Monitoring

```bash
make monitoring      # Démarrer Prometheus + Grafana + Pushgateway
```

### 🧹 Nettoyage

```bash
make clean           # Arrêter les services + supprimer volumes/images locales
make prune           # Nettoyage complet des ressources Docker inutilisées
```

### 💡 Workflow recommandé

```bash
# 1. Première installation
make build && make up

# 2. Lancer le pipeline ML complet
make pipeline

# 3. Vérifier les résultats
make logs-train      # Voir les métriques d'entraînement
# → MLflow : http://localhost:5000
# → Grafana : http://localhost:3000

# 4. Tester l'API
curl -X POST http://localhost/predict \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d '{"designation": "Harry Potter", "description": "Roman jeunesse"}'
```

---
## 🔄 DAG Airflow — Pipeline automatisé

Le DAG `rakuten_ml_pipeline` orchestre les 4 étapes du pipeline de manière séquentielle via `docker compose run --rm`.

### Accès

```
URL     : http://localhost:8081
Login   : admin
Password: admin
```

### Structure du DAG

```
ingest → train → evaluate → reload
```

| Tâche | Container | Description |
|-------|-----------|-------------|
| `ingest` | `rakuten-ingest` | Ingestion et nettoyage des données |
| `train` | `rakuten-train` | Entraînement XGBoost + MLflow logging |
| `evaluate` | `rakuten-evaluate` | Métriques de performance + rapport |
| `reload_model` | `rakuten-reload` | Reload du modèle en production |

### Déclencher le pipeline

```bash
# Via l'interface Airflow : http://localhost:8081
# → Trouver "rakuten_ml_pipeline"
# → Cliquer ▶ (Trigger DAG)

# Ou via CLI Airflow dans le container :
docker compose exec airflow-webserver airflow dags trigger rakuten_ml_pipeline
```

### Paramètres du DAG

```python
dag_id       = "rakuten_ml_pipeline"
schedule     = None          # Déclenchement manuel uniquement
start_date   = datetime(2026, 1, 1)
catchup      = False
tags         = ["rakuten", "ml", "training"]
```

---

## 🌐 API FastAPI

L'API est exposée via le Gateway Nginx sur le port `80`.

### Endpoints principaux

| Méthode | Endpoint | Description | Auth |
|---------|----------|-------------|------|
| `GET` | `/info` | Informations sur le modèle en production | Non |
| `POST` | `/predict` | Prédiction de catégorie produit | Oui (Basic) |
| `POST` | `/reload` | Recharge le modèle depuis le disque | Oui (Basic) |

### Exemples d'utilisation

**Prédiction :**
```bash
curl -X POST http://localhost/predict \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d '{"designation": "Livre Harry Potter", "description": "Roman de sorcellerie"}'

# Réponse
{
  "predicted_class": 2280,
  "label": "Livres",
  "confidence": 0.87,
  "model_version": "xgboost_v3"
}
```

**Informations modèle :**
```bash
curl http://localhost/info

# Réponse
{
  "model": "XGBoostClassifier",
  "version": "xgboost_v3",
  "n_classes": 27,
  "loaded_at": "2026-06-07T05:00:00"
}
```

**Reload du modèle :**
```bash
curl -X POST http://localhost/reload \
  -u admin:admin

# Réponse
{
  "status": "ok",
  "message": "Modèle rechargé avec succès",
  "model_version": "xgboost_v4"
}
```

### Documentation interactive

Accéder au Swagger UI : **http://localhost/docs** (ou http://localhost:8000/docs en direct)

---

## 📊 Monitoring — Prometheus & Grafana

### Grafana

```
URL      : http://localhost:3000
Login    : admin
Password : admin
```

Les dashboards disponibles couvrent :

| Panneau | Métriques |
|---------|-----------|
| **Pipeline status** | Durée d'exécution de chaque job ML |
| **Accuracy** | F1-score et accuracy par run |
| **Prédictions API** | Nombre de requêtes, latence `/predict` |
| **Statut jobs** | Succès / Échec ingest, train, evaluate, reload |

### Prometheus

```
URL : http://localhost:9090
```

**Métriques collectées (via Pushgateway) :**

| Métrique | Description |
|----------|-------------|
| `ml_train_accuracy` | Accuracy sur le jeu de validation après entraînement |
| `ml_train_f1_score` | F1-score macro moyen |
| `ml_train_duration_seconds` | Durée de l'entraînement |
| `ml_train_status` | 1 = succès, 0 = échec |
| `ml_evaluate_accuracy` | Accuracy lors de l'évaluation |
| `ml_evaluate_f1_score` | F1-score lors de l'évaluation |
| `ml_reload_status` | 1 = reload réussi, 0 = échec |
| `api_predict_requests_total` | Nombre de requêtes `/predict` |
| `api_predict_latency_seconds` | Latence de l'inférence |

### Pushgateway

Les scripts `train_script.py`, `evaluate_script.py` et `reload_script.py` pushent leurs métriques vers :
```
http://pushgateway:9091
```

Variable d'environnement utilisée :
```bash
PROMETHEUS_PUSHGATEWAY_URL=http://pushgateway:9091
```

---

## 🧪 MLflow Tracking

```
URL : http://localhost:5000
```

Chaque run d'entraînement est automatiquement loggé avec :

| Éléments trackés | Détails |
|-----------------|---------|
| **Paramètres** | `tfidf_max_features`, `n_estimators`, `max_depth`, `learning_rate` |
| **Métriques** | `accuracy`, `f1_score_macro`, `train_duration_seconds` |
| **Artefacts** | Modèle sérialisé (`model.pkl`), matrice de confusion, rapport de classification |
| **Tags** | `model_type=xgboost`, `dataset=rakuten`, run ID Airflow |

### Comparer des runs

```bash
# Ouvrir MLflow UI
open http://localhost:5000

# → Onglet "Experiments" → "rakuten_experiment"
# → Sélectionner plusieurs runs → "Compare"
```

---

## 📁 Structure du projet

```
Rakuten-MLE-DEC25/
│
├── 📁 api/                          # API FastAPI
│   ├── main.py                      # Endpoints /predict, /info, /reload
    ├── schema.py
    ├── predictor.py
│   └── nomenclature.csv             # Mapping code → libellé catégorie
│
├── 📁 scripts/                      # Scripts ML exécutés par Docker
│   ├── ingest_script.py             # Ingestion et nettoyage des données
│   ├── train_script.py              # Entraînement XGBoost + MLflow + Prometheus
│   ├── evaluate_script.py           # Évaluation + rapport + Prometheus
│   ├── predict_script.py            # Inférence batch sur le test set
│   ├── boostrap_registry.py         # Scraping MLflow métriques
│   ├── push_metrics.py              # Push manuel des métriques vers Prometheus Pushgateway
│   └── reload_script.py             # Reload du modèle dans l'API
│
├── 📁 tests/                      # Configuration Nginx
│   ├── conftest.py                  # Fichier de configuration pour les tests unitaires avec pytest
│   ├── test_api.py                  # Test des endpoints de l'API
│   └── test_scripts.py              # Test des fonctions de preprocessing du pipeline
│
├── 📁 dags/                         # DAGs Airflow
│    └── rakuten-pipeline.py          # DAG : ingest → train → evaluate → reload
│
├── 📁 data/                         # Données brutes (montées en volume Docker)
|   └── raw/
│        ├── X_train_update.csv       # Descriptions produits
│        ├── Y_train_CVw08PX.csv      # Labels (27 classes)
|        └── X_test_update.csv        # Données de validations              
│
├── 📁 artifacts/                    # Features préparées (TF-IDF, encodeurs)
│   ├── tfidf_vectorizer.pkl         # Vectoriseur TF-IDF entraîné
│   ├── label_encoder.pkl            # Encodeur des classes
│   └── X_train_tfidf.pkl            # Features d'entraînement
│
├── 📁 models/                       # Modèles entraînés
│   └── xgb_model.joblib              # Modèle XGBoost (dernier run)
│
├── 📁 reports/                      # Rapports d'évaluation
│   ├── classification_report.json   # Rapport sklearn par classe
│   ├── evaluation_metrics.json      # Metriques d'évaluations enregistrées
│   └── confusion_matrix.csv         # Matrice de confusion
│
├── 📁 predictions/                  # Prédictions batch
│   └── predictions.csv              # Résultats sur le test set
│
├── 📁 mlruns/                       # Artefacts MLflow (montés en volume)
│
├── 📁 monitoring/                   # Configuration observabilité
│   ├── prometheus.yml               # Scrape config Prometheus
│   └── grafana/
│       └── provisioning/
│           ├── datasources/         # Datasource Prometheus
│           └── dashboards/          # Dashboards JSON
│
├── 📁 gateway/                      # Configuration Nginx
│   ├── nginx.conf                   # Reverse proxy + rate limiting
│   └── .htpasswd                    # Auth Basic HTTP
│
├── 📁 airflow/                      # Volumes Airflow
│   ├── logs/                        # Logs des tâches
│   ├── plugins/                     # Plugins custom
│   └── config/                      # Config Airflow
│
├── 📄 Dockerfile                    # Image commune Python (ingest/train/eval/reload/api)
├── 📄 Dockerfile.airflow            # Image Airflow + Docker CLI
├── 📄 docker-compose.yml            # Orchestration complète
├── 📄 requirements.txt              # Dépendances Python
└── 📄 README.md                     # Ce fichier
```

---

## 🔐 Variables d'environnement

### Variables clés par service

| Variable | Service | Valeur par défaut | Description |
|----------|---------|-------------------|-------------|
| `MLFLOW_TRACKING_URI` | train, api | `http://mlflow:5000` | URI du serveur MLflow |
| `PROMETHEUS_PUSHGATEWAY_URL` | train, evaluate, reload, ingest | `http://pushgateway:9091` | URL du Pushgateway |
| `RAKUTEN_API_URL` | reload | `http://api:8000` | URL interne de l'API |
| `PYTHONPATH` | tous | `/app` | Path Python dans les containers |
| `AIRFLOW_UID` | airflow | `50000` | UID utilisateur Airflow |

### Fichier `.env` (recommandé)

```bash
# Créer un .env à la racine du projet
cp .env.example .env  # si disponible, sinon créer manuellement

```

---

## 🐛 Troubleshooting

### Airflow : tâches qui échouent avec "Connection refused"

```bash
# Vérifier que le socket Docker est bien monté
docker compose exec airflow-scheduler docker ps

# Si erreur de permission :
sudo chmod 666 /var/run/docker.sock
```

### Pushgateway : métriques non reçues

```bash
# Tester la connectivité depuis un container ML
docker compose run --rm train sh -c \
  'echo $PROMETHEUS_PUSHGATEWAY_URL && \
   getent hosts pushgateway || echo "DNS KO"'

# Vérifier que pushgateway est bien démarré
docker compose ps pushgateway
```

### MLflow : impossible de se connecter

```bash
# Vérifier les logs MLflow
docker compose logs mlflow

# Tester depuis un container
docker compose run --rm train sh -c \
  'curl -s http://mlflow:5000/health'
```

### API : modèle non chargé au démarrage

```bash
# Vérifier que le fichier xgb_model.joblib existe
ls -la models/

# Recharger manuellement le modèle
curl -X POST http://localhost/reload -u admin:admin
```

### Permissions : "Permission denied" sur volumes

```bash
sudo chown -R $(id -u):$(id -g) \
  mlruns models artifacts logs data predictions reports
```

### Rebuild complet

```bash
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

---

## 📖 Références

- [Challenge Rakuten — ENS](https://challengedata.ens.fr/participants/challenges/35/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

*Dernière mise à jour : Juin 2026*
