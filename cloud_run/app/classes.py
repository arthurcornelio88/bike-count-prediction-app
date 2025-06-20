
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime
import app.app_config as _  # forcer le sys.path side effect
 

# Cleaning and transforming class

class RawCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keep_compteur=True):
        self.keep_compteur = keep_compteur

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("✅ RawCleanerTransformer called")
        X = X.copy()

        # D'abord nettoyer les noms pour éviter erreurs
        X.columns = (
            X.columns
            .str.replace(r'\W+', '_', regex=True)
            .str.replace(r'^_+', '', regex=True)
            .str.lower()
            .str.strip()
        )

        # Conversion de la colonne datetime
        X['date_et_heure_de_comptage'] = pd.to_datetime(
            X['date_et_heure_de_comptage'], utc=True).dt.tz_convert('Europe/Paris')

        # Feature engineering
        X['heure'] = X['date_et_heure_de_comptage'].dt.hour
        X['jour_mois'] = X['date_et_heure_de_comptage'].dt.day
        X['mois'] = X['date_et_heure_de_comptage'].dt.month
        X['annee'] = X['date_et_heure_de_comptage'].dt.year
        X['jour_semaine'] = X['date_et_heure_de_comptage'].dt.day_name()

        # Suppression explicite de la colonne timestamp
        if 'date_et_heure_de_comptage' in X.columns:
            X.drop(columns='date_et_heure_de_comptage', inplace=True)

        # Coordonnées
        X[['latitude', 'longitude']] = X['coordonnées_géographiques'].str.split(',', expand=True).astype(float)

        colonnes_a_supprimer = [
            'mois_annee_comptage', 'identifiant_du_site_de_comptage', 'identifiant_du_compteur',
            'nom_du_site_de_comptage', 'lien_vers_photo_du_site_de_comptage',
            'identifiant_technique_compteur', 'id_photos', "date_d_installation_du_site_de_comptage",
            'test_lien_vers_photos_du_site_de_comptage_', 'id_photo_1', 'url_sites', 'type_dimage',
            'coordonnées_géographiques'
        ]
        X.drop(columns=[col for col in colonnes_a_supprimer if col in X.columns], inplace=True)

        # Types
        X['latitude'] = X['latitude'].astype('float32')
        X['longitude'] = X['longitude'].astype('float32')
        X['jour_semaine'] = X['jour_semaine'].map({
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }).astype('int8')


        if self.keep_compteur:
            X['nom_du_compteur'] = X['nom_du_compteur'].astype('category')
        else:
            X.drop(columns='nom_du_compteur', inplace=True)

        # 🔒 Ordre canonique des colonnes
        X = X[sorted(X.columns)]

        return X

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['heure_sin'] = np.sin(2 * np.pi * X['heure'] / 24)
        X['heure_cos'] = np.cos(2 * np.pi * X['heure'] / 24)
        X['mois_sin'] = np.sin(2 * np.pi * X['mois'] / 12)
        X['mois_cos'] = np.cos(2 * np.pi * X['mois'] / 12)
        X['annee'] = X['annee'].map({2024: 0, 2025: 1})
        return X.drop(columns=['heure', 'mois'])

class AffluenceLabeler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold="mean"):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "comptage_horaire" not in X.columns:
            print("⚠️ Skip AffluenceLabeler: 'comptage_horaire' not found.")
            return X  # Ne rien faire si la target n’est pas là

        X["affluence"] = 0
        for compteur in X["nom_du_compteur"].unique():
            serie = X[X["nom_du_compteur"] == compteur]["comptage_horaire"]
            seuil = serie.mean() if self.threshold == "mean" else serie.median()
            X.loc[
                (X["nom_du_compteur"] == compteur) & (X["comptage_horaire"] > seuil),
                "affluence"
            ] = 1
        return X


# AffluenceClassifierPipeline — pipeline complet
class AffluenceClassifierPipeline:
    def __init__(self):
        self.cleaner = None
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = ["heure", "jour_mois", "mois", "jour_semaine", "annee", "compteur_id"]

    def _prepare_features(self, df, fit=True):
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Encodage du compteur
        if fit:
            df["compteur_id"] = self.label_encoder.fit_transform(df["nom_du_compteur"])
        else:
            df["compteur_id"] = self.label_encoder.transform(df["nom_du_compteur"])

        return df[self.features], df["affluence"] if "affluence" in df else None

    def fit(self, df_raw):
        self.cleaner = Pipeline([
            ("raw", RawCleanerTransformer(keep_compteur=True)),
            ("label", AffluenceLabeler())
        ])
        df_clean = self.cleaner.fit_transform(df_raw)

        X, y = self._prepare_features(df_clean, fit=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        print("✅ Confusion Matrix\n", confusion_matrix(self.y_test, y_pred))
        print("✅ Classification Report\n", classification_report(self.y_test, y_pred))
        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1_score": f1_score(self.y_test, y_pred)
        }

    def predict(self, df_raw):
        cleaner_predict = Pipeline([
            ("raw", RawCleanerTransformer(keep_compteur=True)),
            # pas de ("label", ...) ici
        ])
        df_clean = cleaner_predict.transform(df_raw)

        X, _ = self._prepare_features(df_clean, fit=False)
        return self.model.predict(X)


    def save(self, dir_path="affluence_rf_prod"):
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.cleaner, os.path.join(dir_path, "cleaner.joblib"))
        joblib.dump(self.label_encoder, os.path.join(dir_path, "label_encoder.joblib"))
        joblib.dump(self.model, os.path.join(dir_path, "model.joblib"), compress=3)

        scores = self.evaluate()
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "rf_class",
            "env": "prod",
            "test_mode": False,
            **{k: round(v, 4) for k, v in scores.items()}
        }
        with open(os.path.join(dir_path, "model_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        print("✅ Modèle et artefacts sauvegardés.")

    @classmethod
    def load(cls, dir_path):
        print(f"📥 Chargement pipeline Affluence depuis {dir_path}")
        instance = cls()
        instance.cleaner = joblib.load(os.path.join(dir_path, "cleaner.joblib"))
        instance.label_encoder = joblib.load(os.path.join(dir_path, "label_encoder.joblib"))
        instance.model = joblib.load(os.path.join(dir_path, "model.joblib"), mmap_mode='r')
        return instance