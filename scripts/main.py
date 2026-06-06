"""
Script principal de l'ingestion, de l'entraînement, de l'evaluation et de la prédiction.
Ce script orchestre l'exécution des quatre étapes du pipeline de machine learning :
1. Ingestion des données brutes (ingest_script.py)
2. Entraînement du modèle (train_script.py)
3. Prédiction (predict_script.py)
4. Évaluation (evaluate_script.py)
"""

from ingest_script import main as ingest_main
from train_script import main as train_main
from predict_script import main as predict_main
from evaluate_script import main as evaluate_main


def main():
    print("[MAIN] Démarrage du pipeline complet")

    try:
        print("[MAIN] Étape 1/4 - Ingestion")
        ingest_main()

        print("[MAIN] Étape 2/4 - Entraînement")
        train_main()

        print("[MAIN] Étape 3/4 - Prédiction")
        predict_main()

        print("[MAIN] Étape 4/4 - Évaluation")
        evaluate_main()

        print("[MAIN] Pipeline terminé avec succès")

    except Exception as e:
        print(f"[MAIN] Erreur pipeline : {e}")
        raise

if __name__ == "__main__":
    main()
