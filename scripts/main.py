from ingest_script import main as ingest_main
from train_script import main as train_main
from predict_script import main as predict_main
from evaluate_script import main as evaluate_main

def main():
    # Step 1: Chargement et pre-processing des données
    ingest_main()
    
    # Step 2: Entrainement du modele
    train_main()
    
    # Step 3: Inférence et prédiction
    predict_main()
    
    # Step 4: Evaluation du modele
    evaluate_main()

# Execute la fonction principale
if __name__ == "__main__":
    main()