"""Script de reload du modèle en production.
Ce script envoie une requête à l'API pour déclencher le processus de reload du modèle en production."""

import json
import os
import requests

API_URL = os.getenv("RAKUTEN_API_URL", "http://api:8000")


def main():
    print('[RELOAD] Envoi de la requête de reload à l\'API...')
    response = requests.post(f"{API_URL}/reload", timeout=60) # Envoie une requête POST à l'API
    print(f'[RELOAD] Statut de la réponse HTTP : {response.status_code}') # Affiche le code de statut de la réponse
    response.raise_for_status() # Vérifie que la requête a réussi, sinon lève une exception avec le message d'erreur

    payload = response.json() # Récupère le contenu de la réponse au format JSON

    print('[RELOAD] Rechargement des modèles terminé !')
    print(json.dumps(payload, indent=4, ensure_ascii=False)) # Affiche le contenu de la réponse de manière lisible

    reload_result = payload.get("reload_result", {})
    active_model = payload.get("active_model")

    if "new_prod_version" in reload_result:
        print(f"[RELOAD] Nouvelle version prod : {reload_result['new_prod_version']}")
    if "archived_version" in reload_result:
        print(f"[RELOAD] Ancienne version archivée : {reload_result['archived_version']}")
    if "test_version" in reload_result:
        print(f"[RELOAD] Version test évaluée : {reload_result['test_version']}")
    if "prod_version" in reload_result:
        print(f"[RELOAD] Version prod comparée : {reload_result['prod_version']}")
    if "test_score" in reload_result:
        print(f"[RELOAD] Score test : {reload_result['test_score']}")
    if "prod_score" in reload_result:
        print(f"[RELOAD] Score prod : {reload_result['prod_score']}")
    if "reason" in reload_result:
        print(f"[RELOAD] Raison : {reload_result['reason']}")

    print(f"[RELOAD] Modèle en production après reload : {active_model}")

    print('[RELOAD] Rechargement terminé, vérifiez les logs de l\'API pour plus de détails sur le processus de reload.')


if __name__ == "__main__":
    main()