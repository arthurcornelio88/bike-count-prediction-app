# 📘 Documentation API – `ds_traffic_cyclist1`

## Objectif

API de prédiction du trafic cycliste utilisant des modèles de machine learning, déployée sur Render.com.

---

## 🔧 Utilisation locale (DEV)

### En terminal CLI: 


```bash
curl -X POST 'http://localhost:8080/predict' \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}],"model_type": "nn","metric": "r2"}'
```

### Accès à l’interface Swagger :

```
http://127.0.0.1:8000/docs#/default/predict_predict_post
```

### Exemple de requête :

```json
{
  "records": [
    {
      "nom_du_compteur": "35 boulevard de Ménilmontant NO-SE",
      "date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00",
      "coordonnées_géographiques": "48.8672, 2.3501",
      "mois_annee_comptage": "mai 2025"
    }
  ],
  "model_type": "nn",
  "metric": "r2"
}
```

### Réponse attendue :

```json
{
  "predictions": [
    [
      364.9353942871094
    ]
  ]
}
```

---

## 🌐 Utilisation en production (PROD)

### Exemple `curl` :

```bash
curl -X POST https://ds-traffic-cyclist1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}],"model_type": "nn","metric": "r2"}'
```

---

## 🔁 Déploiement Render (CI/CD)

### Remarque :

La fonctionnalité `render.yaml` (Infrastructure as Code) est disponible uniquement sur les offres payantes de Render.
Les paramètres doivent donc être configurés via l’interface utilisateur.

---

### Paramètres du service

* **Name** : `ds_traffic_cyclist1`
* **Region** : `Frankfurt`
* **Repository** : `https://github.com/arthurcornelio88/ds_traffic_cyclist1`
* **Branch** : `master` (ou `backend_creation` pour staging)
* **Build Command** :

  ```bash
  uv sync --frozen
  ```
* **Start Command** :

  ```bash
  uvicorn app.fastapi_app:app --host 0.0.0.0 --port 10000
  ```

---

### Variables d’environnement

| Clé                              | Valeur                  |
| -------------------------------- | ----------------------- |
| `GOOGLE_APPLICATION_CREDENTIALS` | `/etc/secrets/gcp.json` |

---

### Fichiers secrets

| Nom du fichier | Description                                           |
| -------------- | ----------------------------------------------------- |
| `gcp.json`     | Contenu JSON de la clé GCP (copié-collé manuellement) |

Le fichier est monté dans le conteneur à l'emplacement `/etc/secrets/gcp.json`.

---

### Notes complémentaires

* Le port utilisé par Render est `10000`. Il doit être explicitement défini dans la commande de démarrage de `uvicorn`.
* Le fichier `gcp.json` est monté comme fichier secret et doit être référencé via la variable `GOOGLE_APPLICATION_CREDENTIALS`.