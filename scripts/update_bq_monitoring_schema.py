#!/usr/bin/env python3
"""
Update BigQuery monitoring_audit.logs table schema to add double evaluation columns.
This script adds new columns without dropping the table (preserves existing data).
"""

import os
from google.cloud import bigquery

# Configuration
PROJECT_ID = os.getenv("BQ_PROJECT", "datascientest-460618")
DATASET_ID = "monitoring_audit"
TABLE_ID = "logs"
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# New columns to add (double evaluation fields)
NEW_COLUMNS = [
    bigquery.SchemaField("double_evaluation_enabled", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("baseline_regression", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("r2_baseline", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("r2_current", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("r2_train", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("deployment_decision", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("model_uri", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("run_id", "STRING", mode="NULLABLE"),
]


def main():
    print(f"🔧 Updating BigQuery table schema: {FULL_TABLE_ID}")

    # Initialize client
    client = bigquery.Client(project=PROJECT_ID)

    # Get current table
    try:
        table = client.get_table(FULL_TABLE_ID)
        print(f"✅ Table found: {FULL_TABLE_ID}")
        print(f"📊 Current schema has {len(table.schema)} columns")
    except Exception as e:
        print(f"❌ Error: Table not found - {e}")
        print("💡 Run the DAG once to create the table first")
        return

    # Get current schema
    current_schema = list(table.schema)
    current_field_names = [field.name for field in current_schema]

    print("\n📋 Current columns:")
    for field in current_schema:
        print(f"   - {field.name} ({field.field_type})")

    # Check which new columns are missing
    missing_columns = [
        field for field in NEW_COLUMNS if field.name not in current_field_names
    ]

    if not missing_columns:
        print("\n✅ Schema is already up-to-date! All columns present.")
        return

    print(f"\n🆕 Adding {len(missing_columns)} new columns:")
    for field in missing_columns:
        print(f"   + {field.name} ({field.field_type})")

    # Add new columns to schema
    new_schema = current_schema + missing_columns

    # Update table schema
    table.schema = new_schema

    try:
        table = client.update_table(table, ["schema"])
        print("\n✅ Schema updated successfully!")
        print(f"📊 New schema has {len(table.schema)} columns")

        print("\n📋 Updated columns:")
        for field in table.schema:
            print(f"   - {field.name} ({field.field_type})")

        print(f"\n🎉 Done! Table {FULL_TABLE_ID} is ready for double evaluation.")

    except Exception as e:
        print(f"\n❌ Failed to update schema: {e}")
        print("\n💡 Alternative: Delete and recreate the table")
        print(f"   bq rm -f {FULL_TABLE_ID}")
        print("   Then run the DAG again to create table with new schema")


if __name__ == "__main__":
    main()
