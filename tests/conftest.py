"""
Fichier de configuration pour les tests unitaires avec pytest. 
Il ajoute le répertoire racine du projet au sys.path pour permettre l'importation des modules du projet lors des tests.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))