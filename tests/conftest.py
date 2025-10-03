"""Shared pytest fixtures and configuration."""

import os
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def suppress_tf_warnings():
    """Suppress TensorFlow warnings during tests."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


@pytest.fixture
def sample_bike_data():
    """Create realistic sample bike traffic data."""
    np.random.seed(42)

    n_samples = 200
    compteurs = [
        'Totem 73 boulevard de Sébastopol S-N',
        '35 boulevard de Ménilmontant NO-SE',
        'Face au 48 avenue de Clichy NO-SE'
    ]

    data = {
        'nom_du_compteur': np.random.choice(compteurs, n_samples),
        'Date et heure de comptage': pd.date_range(
            '2024-04-01', periods=n_samples, freq='h', tz='Europe/Paris'
        ).astype(str),
        'Coordonnées géographiques': [
            '48.8672, 2.3501' if i % 3 == 0 else
            '48.8661, 2.3835' if i % 3 == 1 else
            '48.8837, 2.3264'
            for i in range(n_samples)
        ],
        'Mois année comptage': [
            'avril 2024' if i < 100 else 'mai 2024'
            for i in range(n_samples)
        ],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_predictions():
    """Create sample prediction data."""
    return {
        'records': [
            {
                'nom_du_compteur': 'Totem 73 boulevard de Sébastopol S-N',
                'date_et_heure_de_comptage': '2025-05-17 18:00:00+02:00',
                'coordonnées_géographiques': '48.8672, 2.3501',
                'mois_annee_comptage': 'mai 2025'
            }
        ],
        'model_type': 'rf',
        'metric': 'r2'
    }


@pytest.fixture
def mock_gcs_credentials(monkeypatch, tmp_path):
    """Mock GCS credentials for testing."""
    fake_creds = tmp_path / "fake_gcp.json"
    fake_creds.write_text('{"type": "service_account", "project_id": "test"}')

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake_creds))

    return str(fake_creds)
