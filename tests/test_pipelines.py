"""Tests for ML pipelines (RFPipeline, NNPipeline)."""

import pytest
import pandas as pd
import numpy as np
from app.classes import RFPipeline, NNPipeline


@pytest.fixture
def sample_data():
    """Create sample bike traffic data for testing."""
    data = {
        'nom_du_compteur': ['Totem 73 boulevard de Sébastopol S-N'] * 100,
        'Date et heure de comptage': pd.date_range(
            '2024-04-01', periods=100, freq='h', tz='Europe/Paris'
        ).astype(str),
        'Coordonnées géographiques': ['48.8672, 2.3501'] * 100,
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_X_y(sample_data):
    """Create X, y split for training."""
    y = np.random.randint(50, 300, size=len(sample_data))
    return sample_data, y


class TestRFPipeline:
    """Test RandomForest pipeline."""

    def test_rf_pipeline_instantiation(self):
        """Test RF pipeline can be instantiated."""
        rf = RFPipeline()
        assert rf is not None
        assert hasattr(rf, 'model')
        assert hasattr(rf, 'cleaner')

    def test_rf_pipeline_fit(self, sample_X_y):
        """Test RF pipeline can fit on sample data."""
        X, y = sample_X_y
        rf = RFPipeline()

        # Should not raise
        rf.fit(X, y)

        # Model should be fitted
        assert hasattr(rf.model, 'n_estimators')
        assert rf.model.n_estimators == 50

    def test_rf_pipeline_predict(self, sample_X_y):
        """Test RF pipeline can predict."""
        X, y = sample_X_y
        rf = RFPipeline()
        rf.fit(X, y)

        # Predict on subset
        X_test = X.head(10)
        predictions = rf.predict(X_test)

        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
        assert all(predictions >= 0)  # Bike count should be positive

    def test_rf_pipeline_reproducibility(self, sample_X_y):
        """Test RF predictions are reproducible with same data."""
        X, y = sample_X_y

        rf1 = RFPipeline()
        rf1.fit(X, y)
        pred1 = rf1.predict(X.head(5))

        rf2 = RFPipeline()
        rf2.fit(X, y)
        pred2 = rf2.predict(X.head(5))

        # Same random_state should give same predictions
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=1)

    def test_rf_pipeline_save_load(self, sample_X_y, tmp_path):
        """Test RF pipeline can be saved and loaded."""
        X, y = sample_X_y
        rf = RFPipeline()
        rf.fit(X, y)

        # Save
        model_dir = tmp_path / "rf_model"
        rf.save(str(model_dir))

        # Check files exist
        assert (model_dir / "cleaner.joblib").exists()
        assert (model_dir / "preprocessor.joblib").exists()
        assert (model_dir / "model.joblib").exists()

        # Load
        rf_loaded = RFPipeline.load(str(model_dir))

        # Predictions should match
        pred_original = rf.predict(X.head(5))
        pred_loaded = rf_loaded.predict(X.head(5))

        np.testing.assert_array_almost_equal(
            pred_original, pred_loaded, decimal=2
        )


class TestNNPipeline:
    """Test Neural Network pipeline."""

    def test_nn_pipeline_instantiation(self):
        """Test NN pipeline can be instantiated."""
        nn = NNPipeline()
        assert nn is not None
        assert hasattr(nn, 'embedding_dim')
        assert nn.embedding_dim == 8

    def test_nn_pipeline_fit(self, sample_X_y):
        """Test NN pipeline can fit on sample data."""
        X, y = sample_X_y
        y = y.astype('float32')

        nn = NNPipeline()

        # Fit with few epochs for speed
        nn.fit(X, y, epochs=2, batch_size=32)

        # Model should be built
        assert nn.model is not None
        assert nn.n_features is not None

    def test_nn_pipeline_predict(self, sample_X_y):
        """Test NN pipeline can predict."""
        X, y = sample_X_y
        y = y.astype('float32')

        nn = NNPipeline()
        nn.fit(X, y, epochs=2, batch_size=32)

        # Predict
        X_test = X.head(10)
        predictions = nn.predict(X_test).flatten()

        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
        assert all(predictions >= 0)

    def test_nn_pipeline_embedding_dim(self, sample_X_y):
        """Test NN pipeline with custom embedding dimension."""
        X, y = sample_X_y
        y = y.astype('float32')

        nn = NNPipeline(embedding_dim=16)
        assert nn.embedding_dim == 16

        nn.fit(X, y, epochs=1, batch_size=32)
        assert nn.model is not None

    def test_nn_pipeline_save_load(self, sample_X_y, tmp_path):
        """Test NN pipeline can be saved and loaded."""
        X, y = sample_X_y
        y = y.astype('float32')

        nn = NNPipeline()
        nn.fit(X, y, epochs=2, batch_size=32)

        # Save
        model_dir = tmp_path / "nn_model"
        nn.save(str(model_dir))

        # Check files exist
        assert (model_dir / "cleaner.joblib").exists()
        assert (model_dir / "label_encoder.joblib").exists()
        assert (model_dir / "scaler.joblib").exists()
        assert (model_dir / "model.keras").exists()

        # Load
        nn_loaded = NNPipeline.load(str(model_dir))

        # Predictions should be close
        pred_original = nn.predict(X.head(5)).flatten()
        pred_loaded = nn_loaded.predict(X.head(5)).flatten()

        # Allow some tolerance for NN predictions
        np.testing.assert_allclose(
            pred_original, pred_loaded, rtol=0.1
        )

    def test_nn_pipeline_model_architecture(self, sample_X_y):
        """Test NN model has expected architecture."""
        X, y = sample_X_y
        y = y.astype('float32')

        nn = NNPipeline(embedding_dim=8)
        nn.fit(X, y, epochs=1, batch_size=32)

        # Check model inputs
        assert len(nn.model.inputs) == 2  # compteur_id + features_scaled

        # Check model has parameters
        total_params = nn.model.count_params()
        assert total_params > 0


class TestPipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_rf_pipeline_with_minimal_data(self):
        """Test RF pipeline with minimal dataset."""
        X = pd.DataFrame({
            'nom_du_compteur': ['Compteur A'] * 10,
            'Date et heure de comptage': pd.date_range(
                '2024-04-01', periods=10, freq='h', tz='Europe/Paris'
            ).astype(str),
            'Coordonnées géographiques': ['48.8672, 2.3501'] * 10,
        })
        y = np.random.randint(50, 300, size=10)

        rf = RFPipeline()
        rf.fit(X, y)

        predictions = rf.predict(X.head(3))
        assert len(predictions) == 3

    def test_nn_pipeline_with_different_batch_sizes(self, sample_X_y):
        """Test NN pipeline with various batch sizes."""
        X, y = sample_X_y
        y = y.astype('float32')

        for batch_size in [16, 32, 64]:
            nn = NNPipeline()
            nn.fit(X, y, epochs=1, batch_size=batch_size)
            predictions = nn.predict(X.head(5))
            assert len(predictions) == 5
