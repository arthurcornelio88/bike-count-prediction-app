## For testing FastAPI 

### in DEV

http://127.0.0.1:8000/docs#/default/predict_predict_post

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

Expected 

{
  "predictions": [
    [
      364.9353942871094
    ]
  ]
}

### In PROD

curl -X POST https://ds-traffic-cyclist1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}],"model_type": "nn","metric": "r2"}'