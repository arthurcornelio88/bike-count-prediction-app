"""
Normalize reference_data.csv column names to match BigQuery schema
Fixes schema drift detection issue
"""

import pandas as pd
import os

# Column mapping from French (original) to normalized (BigQuery schema)
COLUMN_MAPPING = {
    "Identifiant du compteur": "identifiant_du_compteur",
    "Nom du compteur": "nom_du_compteur",
    "Identifiant du site de comptage": "identifiant_du_site_de_comptage",
    "Nom du site de comptage": "nom_du_site_de_comptage",
    "Comptage horaire": "comptage_horaire",
    "Date et heure de comptage": "date_et_heure_de_comptage",
    "Date d'installation du site de comptage": "date_installation_du_site_de_comptage",
    "Lien vers photo du site de comptage": "lien_vers_photo_du_site_de_comptage",
    "Coordonnées géographiques": "coordonnées_géographiques",
    "Identifiant technique compteur": "identifiant_technique_compteur",
    "ID Photos": "id_photos",
    "test_lien_vers_photos_du_site_de_comptage_": "test_lien_vers_photos_du_site_de_comptage",
    "id_photo_1": "id_photo_1",
    "url_sites": "url_sites",
    "type_dimage": "type_dimage",
    "mois_annee_comptage": "mois_annee_comptage",
}


def normalize_reference_data(
    input_path: str, output_path: str = None, backup: bool = True
):
    """
    Normalize column names in reference_data.csv

    Args:
        input_path: Path to original reference_data.csv
        output_path: Path to save normalized file (default: overwrite input)
        backup: Whether to create .bak backup (default: True)
    """
    if output_path is None:
        output_path = input_path

    print(f"📥 Loading reference data from {input_path}")
    df = pd.read_csv(input_path, sep=";")
    print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")

    print("\n📋 Original columns:")
    for col in df.columns:
        print(f"   - {col}")

    # Create backup if requested
    if backup and input_path == output_path:
        backup_path = input_path + ".bak"
        print(f"\n💾 Creating backup: {backup_path}")
        df.to_csv(backup_path, sep=";", index=False)

    # Rename columns
    print("\n🔄 Normalizing column names...")
    renamed_count = 0
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in df.columns:
            print(f"   ✓ '{old_name}' → '{new_name}'")
            renamed_count += 1

    df.rename(columns=COLUMN_MAPPING, inplace=True)

    print("\n📋 Normalized columns:")
    for col in df.columns:
        print(f"   - {col}")

    # Save normalized file
    print(f"\n💾 Saving normalized file to {output_path}")
    df.to_csv(output_path, sep=";", index=False)

    print("\n✅ Normalization complete!")
    print(f"   - Renamed {renamed_count} columns")
    print(f"   - Saved to: {output_path}")
    if backup and input_path == output_path:
        print(f"   - Backup: {backup_path}")


if __name__ == "__main__":
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Normalize reference data
    reference_path = os.path.join(project_root, "data", "reference_data.csv")

    if not os.path.exists(reference_path):
        print(f"❌ Reference data not found: {reference_path}")
        exit(1)

    normalize_reference_data(reference_path, backup=True)
