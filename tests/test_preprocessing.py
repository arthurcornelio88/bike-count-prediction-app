"""Tests for preprocessing transformers."""

import pytest
import pandas as pd
import numpy as np
from app.classes import RawCleanerTransformer, TimeFeatureTransformer


class TestRawCleanerTransformer:
    """Test RawCleanerTransformer (preprocessing step 1)."""

    def test_transformer_instantiation(self):
        """Test transformer can be created."""
        transformer = RawCleanerTransformer(keep_compteur=True)
        assert transformer is not None
        assert transformer.keep_compteur is True

    def test_column_name_standardization(self, sample_bike_data):
        """Test column names are standardized (lowercase, underscores)."""
        transformer = RawCleanerTransformer(keep_compteur=True)
        df_clean = transformer.fit_transform(sample_bike_data)

        # Check standardized column names
        assert 'nom_du_compteur' in df_clean.columns
        assert 'Nom du compteur' not in df_clean.columns

        # Check no leading/trailing underscores
        for col in df_clean.columns:
            assert not col.startswith('_')
            assert not col.endswith('_')

    def test_datetime_parsing(self, sample_bike_data):
        """Test datetime parsing and timezone conversion."""
        transformer = RawCleanerTransformer(keep_compteur=True)
        df_clean = transformer.fit_transform(sample_bike_data)

        # Original datetime column should be removed
        assert 'date_et_heure_de_comptage' not in df_clean.columns

        # Temporal features should be created
        assert 'heure' in df_clean.columns
        assert 'jour_mois' in df_clean.columns
        assert 'mois' in df_clean.columns
        assert 'annee' in df_clean.columns
        assert 'jour_semaine' in df_clean.columns

        # Check heure range (0-23)
        assert df_clean['heure'].min() >= 0
        assert df_clean['heure'].max() <= 23

        # Check jour_semaine encoded as int (0-6)
        assert df_clean['jour_semaine'].dtype == 'int8'
        assert df_clean['jour_semaine'].min() >= 0
        assert df_clean['jour_semaine'].max() <= 6

    def test_geographic_coordinates_parsing(self, sample_bike_data):
        """Test lat/lon extraction from 'Coordonnées géographiques'."""
        transformer = RawCleanerTransformer(keep_compteur=True)
        df_clean = transformer.fit_transform(sample_bike_data)

        # Lat/lon columns should exist
        assert 'latitude' in df_clean.columns
        assert 'longitude' in df_clean.columns

        # Original column should be removed
        assert 'coordonnées_géographiques' not in df_clean.columns

        # Check data types
        assert df_clean['latitude'].dtype == 'float32'
        assert df_clean['longitude'].dtype == 'float32'

        # Check Paris coordinates range
        assert df_clean['latitude'].between(48.8, 48.9).all()
        assert df_clean['longitude'].between(2.2, 2.4).all()

    def test_unwanted_columns_removed(self, sample_bike_data):
        """Test that unwanted columns are dropped."""
        transformer = RawCleanerTransformer(keep_compteur=True)
        df_clean = transformer.fit_transform(sample_bike_data)

        # Columns that should be removed
        unwanted = [
            'mois_annee_comptage',
            'identifiant_du_site_de_comptage',
            'identifiant_du_compteur',
            'nom_du_site_de_comptage',
        ]

        for col in unwanted:
            assert col not in df_clean.columns

    def test_compteur_column_handling(self, sample_bike_data):
        """Test keep_compteur parameter behavior."""
        # Test with keep_compteur=True
        transformer_keep = RawCleanerTransformer(keep_compteur=True)
        df_keep = transformer_keep.fit_transform(sample_bike_data)
        assert 'nom_du_compteur' in df_keep.columns
        assert df_keep['nom_du_compteur'].dtype.name == 'category'

        # Test with keep_compteur=False
        transformer_drop = RawCleanerTransformer(keep_compteur=False)
        df_drop = transformer_drop.fit_transform(sample_bike_data)
        assert 'nom_du_compteur' not in df_drop.columns

    def test_column_sorting(self, sample_bike_data):
        """Test that output columns are sorted alphabetically."""
        transformer = RawCleanerTransformer(keep_compteur=True)
        df_clean = transformer.fit_transform(sample_bike_data)

        columns = list(df_clean.columns)
        assert columns == sorted(columns)

    def test_transformer_fit_transform_idempotent(self, sample_bike_data):
        """Test that fit_transform is idempotent (same result on re-run)."""
        transformer = RawCleanerTransformer(keep_compteur=True)

        df_clean1 = transformer.fit_transform(sample_bike_data.copy())
        df_clean2 = transformer.fit_transform(sample_bike_data.copy())

        pd.testing.assert_frame_equal(df_clean1, df_clean2)


class TestTimeFeatureTransformer:
    """Test TimeFeatureTransformer (cyclical encoding)."""

    @pytest.fixture
    def temporal_data(self):
        """Create data with temporal features."""
        return pd.DataFrame({
            'heure': np.arange(24),  # 0 to 23 (24 rows)
            'mois': np.tile(np.arange(1, 13), 2),  # 1 to 12, repeated (24 rows)
            'jour_mois': np.arange(1, 25),  # 1 to 24 (24 rows)
            'annee': [2024] * 12 + [2025] * 12,  # 24 rows
            'jour_semaine': np.tile(np.arange(7), 4)[:24]  # 24 rows
        })

    def test_transformer_instantiation(self):
        """Test transformer can be created."""
        transformer = TimeFeatureTransformer()
        assert transformer is not None

    def test_cyclical_encoding_heure(self, temporal_data):
        """Test hour cyclical encoding (sin/cos)."""
        transformer = TimeFeatureTransformer()
        df_transformed = transformer.fit_transform(temporal_data)

        # Check new columns exist
        assert 'heure_sin' in df_transformed.columns
        assert 'heure_cos' in df_transformed.columns

        # Original 'heure' should be removed
        assert 'heure' not in df_transformed.columns

        # Check sin/cos range [-1, 1]
        assert df_transformed['heure_sin'].between(-1, 1).all()
        assert df_transformed['heure_cos'].between(-1, 1).all()

        # Check midnight (0h) and noon (12h) encoding
        # At midnight: heure_sin = sin(0) = 0, heure_cos = cos(0) = 1
        # At noon: heure_sin = sin(π) = 0, heure_cos = cos(π) = -1
        midnight_row = df_transformed.iloc[0]
        assert np.isclose(midnight_row['heure_sin'], 0, atol=1e-10)
        assert np.isclose(midnight_row['heure_cos'], 1, atol=1e-10)

    def test_cyclical_encoding_mois(self, temporal_data):
        """Test month cyclical encoding (sin/cos)."""
        transformer = TimeFeatureTransformer()
        df_transformed = transformer.fit_transform(temporal_data)

        # Check new columns exist
        assert 'mois_sin' in df_transformed.columns
        assert 'mois_cos' in df_transformed.columns

        # Original 'mois' should be removed
        assert 'mois' not in df_transformed.columns

        # Check sin/cos range
        assert df_transformed['mois_sin'].between(-1, 1).all()
        assert df_transformed['mois_cos'].between(-1, 1).all()

    def test_annee_mapping(self, temporal_data):
        """Test year mapping (2024→0, 2025→1)."""
        transformer = TimeFeatureTransformer()
        df_transformed = transformer.fit_transform(temporal_data)

        # Check annee values are mapped
        assert set(df_transformed['annee'].unique()) == {0, 1}
        assert (df_transformed['annee'].iloc[:12] == 0).all()  # 2024 rows
        assert (df_transformed['annee'].iloc[12:] == 1).all()  # 2025 rows

    def test_output_columns_preserved(self, temporal_data):
        """Test that non-transformed columns are preserved."""
        transformer = TimeFeatureTransformer()
        df_transformed = transformer.fit_transform(temporal_data)

        # These should still exist
        assert 'jour_mois' in df_transformed.columns
        assert 'jour_semaine' in df_transformed.columns

    def test_transformer_with_pipeline(self, sample_bike_data):
        """Test TimeFeatureTransformer after RawCleaner in pipeline."""
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ('raw', RawCleanerTransformer(keep_compteur=False)),
            ('time', TimeFeatureTransformer())
        ])

        df_transformed = pipeline.fit_transform(sample_bike_data)

        # Check cyclical features exist
        assert 'heure_sin' in df_transformed.columns
        assert 'heure_cos' in df_transformed.columns
        assert 'mois_sin' in df_transformed.columns
        assert 'mois_cos' in df_transformed.columns

        # Original temporal columns should be gone
        assert 'heure' not in df_transformed.columns
        assert 'mois' not in df_transformed.columns


class TestPreprocessingEdgeCases:
    """Test edge cases and error handling."""

    def test_raw_cleaner_with_missing_coordinates(self):
        """Test RawCleaner handles missing coordinates."""
        data = {
            'Nom du compteur': ['Compteur A', 'Compteur B', 'Compteur C'],
            'Date et heure de comptage': pd.date_range(
                '2024-04-01', periods=3, freq='h', tz='Europe/Paris'
            ).astype(str),
            'Coordonnées géographiques': ['48.8, 2.3', None, '48.9, 2.4']
        }
        df = pd.DataFrame(data)

        transformer = RawCleanerTransformer(keep_compteur=True)
        df_clean = transformer.fit_transform(df)

        # Should have latitude/longitude for valid rows
        assert 'latitude' in df_clean.columns
        assert 'longitude' in df_clean.columns

    def test_time_transformer_with_edge_hours(self):
        """Test TimeFeatureTransformer with boundary hours."""
        data = {
            'heure': [0, 6, 12, 18, 23],  # Midnight, dawn, noon, dusk, late
            'mois': [1, 3, 6, 9, 12],
            'annee': [2024, 2024, 2025, 2025, 2025],
            'jour_mois': [1, 15, 15, 20, 31],
            'jour_semaine': [0, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data)

        transformer = TimeFeatureTransformer()
        df_transformed = transformer.fit_transform(df)

        # Check all transformations succeed
        assert len(df_transformed) == 5
        assert 'heure_sin' in df_transformed.columns

    def test_raw_cleaner_column_order_consistency(self):
        """Test that column order is consistent across runs."""
        data1 = {
            'Nom du compteur': ['A'] * 5,
            'Date et heure de comptage': pd.date_range(
                '2024-04-01', periods=5, freq='h', tz='Europe/Paris'
            ).astype(str),
            'Coordonnées géographiques': ['48.8, 2.3'] * 5
        }

        data2 = {
            'Coordonnées géographiques': ['48.8, 2.3'] * 5,  # Different order
            'Date et heure de comptage': pd.date_range(
                '2024-04-01', periods=5, freq='h', tz='Europe/Paris'
            ).astype(str),
            'Nom du compteur': ['A'] * 5,
        }

        transformer = RawCleanerTransformer(keep_compteur=True)

        df1 = transformer.fit_transform(pd.DataFrame(data1))
        df2 = transformer.fit_transform(pd.DataFrame(data2))

        # Column order should be identical (sorted)
        assert list(df1.columns) == list(df2.columns)
