"""
Configuration centralisée des variables d'environnement
Gère DEV/PROD avec Google Secret Manager
"""
import os
from typing import Dict, Optional
from google.cloud import secretmanager


class SecretManager:
    """Gestionnaire des secrets Google Cloud"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()

    def get_secret(self, secret_id: str) -> Optional[str]:
        """Récupère un secret depuis Secret Manager"""
        try:
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8").strip()
        except Exception as e:
            print(f"⚠️ Failed to load secret {secret_id}: {e}")
            return None


class EnvironmentConfig:
    """Gestionnaire centralisé des variables d'environnement"""

    def __init__(self):
        self.env = os.getenv("ENV", "DEV")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "datascientest-460618")

    def setup_environment_variables(self):
        """Configure toutes les variables d'environnement selon l'environnement"""
        print(f"🔧 Setting up environment variables for {self.env} mode...")

        if self.env == "PROD":
            self._setup_production_env()
        else:
            self._setup_development_env()

        # Debug: afficher les variables importantes
        self._debug_environment_vars()

    def _setup_production_env(self):
        """Configuration production avec secrets Google Cloud"""
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set for production mode")

        # Charger les secrets
        secret_manager = SecretManager(self.project_id)

        # Mapping des secrets pour bike traffic
        secret_mapping = {
            "GCS_BUCKET": "gcs-bucket-bike",
            "BQ_PROJECT": "bq-project-bike",
            "BQ_RAW_DATASET": "bq-raw-dataset-bike",
            "BQ_PREDICT_DATASET": "bq-predict-dataset-bike",
            "BQ_LOCATION": "bq-location",
            "PROD_API_URL": "prod-bike-api-url",
            "API_KEY_SECRET": "bike-api-key-secret",
        }

        # Charger les secrets
        for env_var, secret_id in secret_mapping.items():
            secret_value = secret_manager.get_secret(secret_id)
            if secret_value:
                clean_value = secret_value.strip()
                os.environ[env_var] = clean_value
                print(f"✅ Loaded secret {secret_id} → {env_var} = {clean_value[:20]}...")
            else:
                print(f"⚠️ Failed to load secret {secret_id}")

        # Configuration des chemins production
        bucket = os.environ.get("GCS_BUCKET", "df_traffic_cyclist1")
        os.environ["SHARED_DATA_PATH"] = f"gs://{bucket}/shared_data"
        os.environ["MODEL_PATH"] = f"gs://{bucket}/models"

        # Valeurs par défaut pour les variables manquantes
        self._set_default_if_missing("BQ_LOCATION", "europe-west1")

    def _setup_development_env(self):
        """Configuration développement locale"""
        print("🔧 Setting up development environment...")

        # Chemins locaux
        os.environ["SHARED_DATA_PATH"] = "/app/shared_data"
        os.environ["MODEL_PATH"] = "/app/models"
        os.environ["GCS_BUCKET"] = os.getenv("GCS_BUCKET", "df_traffic_cyclist1")

        # Configuration BigQuery développement
        os.environ["BQ_PROJECT"] = os.getenv("BQ_PROJECT", "datascientest-460618")
        os.environ["BQ_RAW_DATASET"] = os.getenv("BQ_RAW_DATASET", "bike_traffic_raw")
        os.environ["BQ_PREDICT_DATASET"] = os.getenv("BQ_PREDICT_DATASET", "bike_traffic_predictions")
        os.environ["BQ_LOCATION"] = os.getenv("BQ_LOCATION", "europe-west1")

        # API URL développement
        os.environ["API_URL"] = os.getenv("API_URL_DEV", "http://regmodel-backend:8000")
        os.environ["API_KEY_SECRET"] = os.getenv("API_KEY_SECRET", "dev-key-unsafe")

        # Créer les répertoires nécessaires
        import pathlib
        pathlib.Path("/app/shared_data").mkdir(parents=True, exist_ok=True)
        pathlib.Path("/app/models").mkdir(parents=True, exist_ok=True)

    def _set_default_if_missing(self, env_var: str, default_value: str):
        """Définit une valeur par défaut si la variable n'existe pas"""
        if not os.environ.get(env_var):
            os.environ[env_var] = default_value
            print(f"⚠️ {env_var} not set, using default: {default_value}")

    def _debug_environment_vars(self):
        """Affiche les variables d'environnement importantes pour debug"""
        important_vars = [
            "ENV",
            "GOOGLE_CLOUD_PROJECT",
            "GCS_BUCKET",
            "BQ_PROJECT",
            "BQ_RAW_DATASET",
            "BQ_PREDICT_DATASET",
            "BQ_LOCATION",
            "API_URL",
            "SHARED_DATA_PATH",
            "MODEL_PATH",
        ]

        print("\n🔍 DEBUG: Environment variables:")
        for var in important_vars:
            value = os.environ.get(var, "NOT SET")
            # Masquer les URLs complètes pour la sécurité
            if "URI" in var or "URL" in var or "SECRET" in var or "KEY" in var:
                display_value = value[:20] + "..." if len(value) > 20 else value
            else:
                display_value = value
            print(f"  {var} = {display_value}")
        print()

    def get_config(self) -> Dict[str, str]:
        """
        Retourne un dictionnaire avec toutes les configurations

        Returns:
            Dictionary avec les clés:
            - ENV, API_URL, BQ_PROJECT, BQ_RAW_DATASET,
              BQ_PREDICT_DATASET, BQ_LOCATION, GCS_BUCKET
        """
        return {
            "ENV": os.environ.get("ENV", "DEV"),
            "API_URL": os.environ.get("API_URL", os.environ.get("PROD_API_URL", "")),
            "BQ_PROJECT": os.environ.get("BQ_PROJECT", ""),
            "BQ_RAW_DATASET": os.environ.get("BQ_RAW_DATASET", ""),
            "BQ_PREDICT_DATASET": os.environ.get("BQ_PREDICT_DATASET", ""),
            "BQ_LOCATION": os.environ.get("BQ_LOCATION", "europe-west1"),
            "GCS_BUCKET": os.environ.get("GCS_BUCKET", ""),
        }


# Fonction utilitaire pour initialiser l'environnement
def setup_environment() -> EnvironmentConfig:
    """Point d'entrée principal pour configurer l'environnement"""
    config = EnvironmentConfig()
    config.setup_environment_variables()
    return config


def get_env_config() -> Dict[str, str]:
    """
    Helper function pour récupérer rapidement la config dans les DAGs

    Returns:
        Dictionary avec les configurations essentielles
    """
    config = EnvironmentConfig()
    config.setup_environment_variables()
    return config.get_config()


if __name__ == "__main__":
    # Test configuration
    config = setup_environment()
    print("\n✅ Configuration loaded successfully")
