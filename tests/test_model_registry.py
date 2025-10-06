"""
Tests for model registry summary.json logic.

Simple tests covering:
- update_summary() - adding entries to summary.json
- get_best_model_from_summary() - selecting best model by metric
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from backend.regmodel.app.model_registry_summary import update_summary


class TestUpdateSummary:
    """Test summary.json update logic."""

    def test_update_summary_creates_new_file(self, tmp_path):
        """Test creating new summary.json file."""
        summary_path = str(tmp_path / "summary.json")

        update_summary(
            summary_path=summary_path,
            model_type="rf",
            run_id="test_run_123",
            model_uri="gs://bucket/models/rf",
            env="prod",
            test_mode=False,
            r2=0.85,
            rmse=12.5
        )

        # Check file exists
        assert os.path.exists(summary_path)

        # Check content
        with open(summary_path, "r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["model_type"] == "rf"
        assert data[0]["run_id"] == "test_run_123"
        assert data[0]["r2"] == 0.85
        assert data[0]["rmse"] == 12.5
        assert data[0]["env"] == "prod"

    def test_update_summary_appends_to_existing(self, tmp_path):
        """Test appending to existing summary.json."""
        summary_path = str(tmp_path / "summary.json")

        # First entry
        update_summary(
            summary_path=summary_path,
            model_type="rf",
            run_id="run_1",
            model_uri="gs://bucket/rf_1",
            r2=0.80
        )

        # Second entry
        update_summary(
            summary_path=summary_path,
            model_type="nn",
            run_id="run_2",
            model_uri="gs://bucket/nn_1",
            r2=0.90
        )

        with open(summary_path, "r") as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["run_id"] == "run_1"
        assert data[1]["run_id"] == "run_2"

    def test_update_summary_with_all_metrics(self, tmp_path):
        """Test summary with all metric types."""
        summary_path = str(tmp_path / "summary.json")

        update_summary(
            summary_path=summary_path,
            model_type="rf_class",
            run_id="classifier_run",
            model_uri="gs://bucket/classifier",
            accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89
        )

        with open(summary_path, "r") as f:
            data = json.load(f)

        entry = data[0]
        assert entry["accuracy"] == 0.92
        assert entry["precision"] == 0.90
        assert entry["recall"] == 0.88
        assert entry["f1_score"] == 0.89

    def test_update_summary_handles_corrupted_json(self, tmp_path):
        """Test handling of corrupted summary.json."""
        summary_path = str(tmp_path / "summary.json")

        # Create corrupted JSON
        with open(summary_path, "w") as f:
            f.write("invalid json{{{")

        # Should handle gracefully
        update_summary(
            summary_path=summary_path,
            model_type="rf",
            run_id="new_run",
            model_uri="gs://bucket/rf",
            r2=0.85
        )

        with open(summary_path, "r") as f:
            data = json.load(f)

        # Should have reinitialized and added entry
        assert len(data) == 1
        assert data[0]["run_id"] == "new_run"


class TestGetBestModel:
    """Test best model selection logic."""

    def test_get_best_model_by_r2(self, tmp_path):
        """Test selecting best model by r2 (highest)."""
        summary_path = str(tmp_path / "summary.json")

        # Create summary with multiple models
        summary = [
            {"model_type": "rf", "env": "prod", "test_mode": False, "r2": 0.80, "run_id": "run_1", "model_uri": "gs://bucket/rf_1"},
            {"model_type": "rf", "env": "prod", "test_mode": False, "r2": 0.90, "run_id": "run_2", "model_uri": "gs://bucket/rf_2"},
            {"model_type": "rf", "env": "prod", "test_mode": False, "r2": 0.85, "run_id": "run_3", "model_uri": "gs://bucket/rf_3"},
        ]

        with open(summary_path, "w") as f:
            json.dump(summary, f)

        # Mock GCS and model loading
        with patch("backend.regmodel.app.model_registry_summary._download_gcs_dir") as mock_download:
            with patch("backend.regmodel.app.model_registry_summary.RFPipeline.load") as mock_load:
                test_dir = str(tmp_path / "model")
                os.makedirs(test_dir, exist_ok=True)
                mock_download.return_value = test_dir
                mock_load.return_value = MagicMock()

                from backend.regmodel.app.model_registry_summary import get_best_model_from_summary

                model = get_best_model_from_summary(
                    model_type="rf",
                    summary_path=summary_path,
                    env="prod",
                    metric="r2",
                    test_mode=False
                )

                # Should have selected run_2 (r2=0.90)
                mock_download.assert_called_once()
                call_args = mock_download.call_args[0]
                assert "rf_2" in call_args[0]

    def test_get_best_model_by_rmse(self, tmp_path):
        """Test selecting best model by rmse (lowest)."""
        summary_path = str(tmp_path / "summary.json")

        summary = [
            {"model_type": "rf", "env": "prod", "test_mode": False, "rmse": 15.0, "run_id": "run_1", "model_uri": "gs://bucket/rf_1"},
            {"model_type": "rf", "env": "prod", "test_mode": False, "rmse": 10.0, "run_id": "run_2", "model_uri": "gs://bucket/rf_2"},
            {"model_type": "rf", "env": "prod", "test_mode": False, "rmse": 12.0, "run_id": "run_3", "model_uri": "gs://bucket/rf_3"},
        ]

        with open(summary_path, "w") as f:
            json.dump(summary, f)

        with patch("backend.regmodel.app.model_registry_summary._download_gcs_dir") as mock_download:
            with patch("backend.regmodel.app.model_registry_summary.RFPipeline.load") as mock_load:
                test_dir = str(tmp_path / "model_rmse")
                os.makedirs(test_dir, exist_ok=True)
                mock_download.return_value = test_dir
                mock_load.return_value = MagicMock()

                from backend.regmodel.app.model_registry_summary import get_best_model_from_summary

                model = get_best_model_from_summary(
                    model_type="rf",
                    summary_path=summary_path,
                    env="prod",
                    metric="rmse",
                    test_mode=False
                )

                # Should select run_2 (rmse=10.0, lowest)
                call_args = mock_download.call_args[0]
                assert "rf_2" in call_args[0]
