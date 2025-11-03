# test_secrets.py
from google.cloud import secretmanager


def test_secrets():
    project_id = "datascientest-460618"
    client = secretmanager.SecretManagerServiceClient()

    secrets = [
        "gcs-bucket-bike",
        "bq-project-bike",
        "bq-raw-dataset-bike",
        "bq-predict-dataset-bike",
        "bq-location",
        "prod-bike-api-url",
        "bike-api-key-secret",
    ]

    for secret_id in secrets:
        try:
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            value = response.payload.data.decode("UTF-8")
            print(f"✅ {secret_id}: {value[:20]}...")
        except Exception as e:
            print(f"❌ {secret_id}: {e}")


if __name__ == "__main__":
    test_secrets()
