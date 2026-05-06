import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ETAPE 1 — LECTURE DES DONNÉES
# ============================================================
print("=== ETAPE 1 : Chargement des données ===")

X_train_init = pd.read_csv("../data/X_train_update.csv", nrows=50000)
y_train_init = pd.read_csv("../data/Y_train_CVw08PX.csv", nrows=10000)
X_test_init  = pd.read_csv("../data/X_test_update.csv", nrows=50000)

print(f"X_train : {X_train_init.shape}")
print(f"y_train : {y_train_init.shape}")
print(f"X_test  : {X_test_init.shape}")


# ============================================================
# ETAPE 2 — ALIGNEMENT DES DONNÉES
# ============================================================
print("\n=== ETAPE 2 : Alignement des données ===")

# Alignement de X_train sur y_train (10 000 lignes)
X_train_init = X_train_init.iloc[:len(y_train_init)]
print(f"X_train après alignement : {X_train_init.shape}")
print(f"y_train                  : {y_train_init.shape}")


# ============================================================
# ETAPE 3 — NETTOYAGE DES DONNÉES
# ============================================================
print("\n=== ETAPE 3 : Nettoyage des données ===")

# Vérification des valeurs manquantes
print("Valeurs manquantes X_train :", X_train_init.isnull().sum().sum())
print("Valeurs manquantes y_train :", y_train_init.isnull().sum().sum())
print("Valeurs manquantes X_test  :", X_test_init.isnull().sum().sum())

# Suppression des valeurs manquantes
X_train_init = X_train_init.dropna()
X_test_init  = X_test_init.dropna()

# Réalignement après suppression
y_train_init = y_train_init.iloc[:len(X_train_init)]
print(f"Après nettoyage — X_train : {X_train_init.shape}, y_train : {y_train_init.shape}")


# ============================================================
# ETAPE 4 — PRÉPARATION DES VARIABLES
# ============================================================
print("\n=== ETAPE 4 : Préparation des variables ===")

# Récupération de la variable cible
y = y_train_init.iloc[:, 0]

# Encodage si la cible est textuelle
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Classes encodées : {le.classes_}")

# Séparation train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_init, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"X_train : {X_train.shape}, X_val : {X_val.shape}")


# ============================================================
# ETAPE 5 — NORMALISATION
# ============================================================
print("\n=== ETAPE 5 : Normalisation ===")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test_init)

print("Normalisation effectuée ✅")


# ============================================================
# ETAPE 6 — OPTIMISATION BAYÉSIENNE
# ============================================================
print("\n=== ETAPE 6 : BayesSearchCV ===")

espace_recherche = {
    'C': Real(1e-4, 1e+2, prior='log-uniform'),
}

bayes_search = BayesSearchCV(
    estimator=SVC(kernel='linear', probability=True, class_weight='balanced'),
    search_spaces=espace_recherche,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

bayes_search.fit(X_train_scaled, y_train)

print(f"\nC optimal       : {bayes_search.best_params_['C']:.6f}")
print(f"Meilleur score CV : {bayes_search.best_score_*100:.2f}%")


# ============================================================
# ETAPE 7 — ÉVALUATION DU MODÈLE
# ============================================================
print("\n=== ETAPE 7 : Évaluation ===")

meilleur_modele = bayes_search.best_estimator_
y_pred  = meilleur_modele.predict(X_val_scaled)
y_proba = meilleur_modele.predict_proba(X_val_scaled)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_val, y_pred))

# F1-score
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1-score  : {f1:.4f}")

# Cross-entropy
ce = log_loss(y_val, y_proba)
print(f"Cross-entropy : {ce:.4f}")


# ============================================================
# ETAPE 8 — MATRICE DE CONFUSION
# ============================================================
print("\n=== ETAPE 8 : Matrice de confusion ===")

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Greens", values_format="d")
plt.title("Matrice de confusion - SVM Linéaire")
plt.tight_layout()
plt.savefig("../reports/figures/confusion_matrix_SVM.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# ETAPE 9 — PRÉDICTION SUR X_TEST
# ============================================================
print("\n=== ETAPE 9 : Prédictions finales ===")

y_test_pred = meilleur_modele.predict(X_test_scaled)
print(f"Prédictions générées : {len(y_test_pred)} lignes")

# Sauvegarde des prédictions
predictions = pd.DataFrame({'prediction': y_test_pred})
predictions.to_csv("../reports/predictions_SVM.csv", index=False)
print("Prédictions sauvegardées ✅")