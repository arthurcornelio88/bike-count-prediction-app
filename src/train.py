import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import os
import shutil
import numpy as np
from app.classes import RFPipeline, NNPipeline, AffluenceClassifierPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import app.app_config as _  # forcer le sys.path side effect
from app.model_registry_summary import update_summary
from datetime import datetime


def setup_environment(env: str, model_test: bool):
    if env == "dev":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("traffic_cycliste_experiment")
        data_path = "data/comptage-velo-donnees-compteurs_test.csv" if model_test else "data/comptage-velo-donnees-compteurs.csv"
        artifact_path = "models/"
        os.makedirs(artifact_path, exist_ok=True)
    elif env == "prod":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("traffic_cycliste_experiment")
        data_path = "gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs_test.csv" if model_test else "gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs.csv"
        artifact_path = "models/"
        os.makedirs(artifact_path, exist_ok=True)

        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
            gcp_credentials_path = "./mlflow-trainer.json"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path
            if not os.path.exists(gcp_credentials_path):
                raise FileNotFoundError(f"Clé GCP manquante : {gcp_credentials_path}")
    else:
        raise ValueError("Environnement invalide")
    
    return data_path, artifact_path


def load_and_clean_data(path: str, preserve_target=False):
    df = pd.read_csv(path, sep=";")
    df[['latitude', 'longitude']] = df['Coordonnées géographiques'].str.split(',', expand=True).astype(float)
    df_clean = df.dropna(subset=['latitude', 'longitude'])

    if preserve_target:
        return df_clean

    X = df_clean.drop(columns='Comptage horaire')
    y = df_clean['Comptage horaire']
    return X, y



def log_and_export_model(model_type, model_obj, temp_model_path, rmse, r2, env, test_mode, run_id):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_subdir = f"{model_type}/{timestamp}"
    gcs_model_uri = f"gs://df_traffic_cyclist1/models/{model_subdir}/"

    if env == "prod":
        cp_command = f"gsutil -m cp -r {os.path.join(temp_model_path, model_type)} {gcs_model_uri}"
        cp_success = os.system(cp_command) == 0

        if cp_success:
            update_summary(
                summary_path="gs://df_traffic_cyclist1/models/summary.json",
                model_type=model_type,
                run_id=run_id,
                rmse=rmse,
                r2=r2,
                model_uri=gcs_model_uri,
                env=env,
                test_mode=test_mode
            )
            print(f"📤 Modèle {model_type} exporté vers {gcs_model_uri}")
        else:
            print(f"❌ Erreur lors de la copie du modèle {model_type} vers GCS")

    shutil.rmtree(temp_model_path)
    print(f"🧹 Répertoire temporaire supprimé : {temp_model_path}")

def train_rf(X, y, env, test_mode):
    y = y.to_numpy()
    run_name = f"RandomForest_Train_{env}" + ("_TEST" if test_mode else "")
    print("📡 Tracking URI :", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        mlflow.set_tag("mode", env)
        mlflow.set_tag("test_mode", test_mode)
        mlflow.log_metric("test_mode", int(test_mode))
        mlflow.sklearn.autolog(disable=True)

        rf = RFPipeline()
        rf.fit(X, y)

        mlflow.log_param("rf_n_estimators", rf.model.n_estimators)
        mlflow.log_param("rf_max_depth", rf.model.max_depth)

        y_pred = rf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        mlflow.log_metric("rf_rmse_train", rmse)
        mlflow.log_metric("rf_r2_train", r2)

        print(f"🎯 RF – RMSE : {rmse:.2f} | R² : {r2:.4f}")
        # mlflow.log_artifacts("tmp_rf_model", artifact_path="rf_model")
        
        if env == "prod":
            model_dir = os.path.join("tmp_rf_model", "rf")
        else:
            model_dir = os.path.join("models", "rf_prod")

        os.makedirs(model_dir, exist_ok=True)
        rf.save(model_dir)

        # ✅ log après .save()
        if env == "prod":
            mlflow.log_artifacts(model_dir, artifact_path="rf_model")
            log_and_export_model("rf", rf, "tmp_rf_model", rmse, r2, env, test_mode, run.info.run_id)

            if os.path.exists("tmp_rf_model"):
                shutil.rmtree("tmp_rf_model")
                print("🧹 Dossier temporaire supprimé : tmp_rf_model")
        else:
            mlflow.log_artifacts(model_dir, artifact_path="rf_model")
            print(f"✅ Modèle RF sauvegardé localement dans : {model_dir}")

def train_nn(X, y, env, test_mode):
    y = y.to_numpy(dtype="float32")
    run_name = f"NeuralNet_Train_{env}" + ("_TEST" if test_mode else "")

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        mlflow.set_tag("mode", env)
        mlflow.set_tag("test_mode", test_mode)
        mlflow.log_metric("test_mode", int(test_mode))
        mlflow.tensorflow.autolog(disable=True)

        nn = NNPipeline()
        epochs = 50
        batch_size = 128
        nn.fit(X, y, epochs=epochs, batch_size=batch_size)

        mlflow.log_param("nn_epochs", epochs)
        mlflow.log_param("nn_batch_size", batch_size)
        mlflow.log_param("nn_embedding_dim", nn.embedding_dim)

        total_params = nn.model.count_params()
        mlflow.log_metric("nn_total_params", total_params)

        y_pred = nn.predict(X).flatten()
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        mlflow.log_metric("nn_rmse_train", rmse)
        mlflow.log_metric("nn_r2_train", r2)

        print(f"🎯 NN – RMSE : {rmse:.2f} | R² : {r2:.4f} | Params: {total_params}")
        #mlflow.log_artifacts("tmp_nn_model", artifact_path="nn_model")

        if env == "prod":
            model_dir = os.path.join("tmp_nn_model", "nn")
        else:
            model_dir = os.path.join("models", "nn_prod")

        os.makedirs(model_dir, exist_ok=True)
        nn.save(model_dir)

        # ✅ log après .save()
        if env == "prod":
            mlflow.log_artifacts(model_dir, artifact_path="nn_model")
            log_and_export_model("nn", nn, "tmp_nn_model", rmse, r2, env, test_mode, run.info.run_id)

            if os.path.exists("tmp_nn_model"):
                shutil.rmtree("tmp_nn_model")
                print("🧹 Dossier temporaire supprimé : tmp_nn_model")
        else:
            mlflow.log_artifacts(model_dir, artifact_path="nn_model")
            print(f"✅ Modèle NN sauvegardé localement dans : {model_dir}")



def train_rfc(X_raw, y_unused, env, test_mode):
    run_name = f"AffluenceClassifier_Train_{env}" + ("_TEST" if test_mode else "")
    print("📡 Tracking URI :", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        mlflow.set_tag("mode", env)
        mlflow.set_tag("test_mode", test_mode)
        mlflow.log_metric("test_mode", int(test_mode))
        mlflow.sklearn.autolog(disable=True)

        # Entraînement
        clf = AffluenceClassifierPipeline()
        clf.fit(X_raw)

        # Évaluation
        y_true = clf.y_test
        y_pred = clf.model.predict(clf.X_test)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        mlflow.log_metric("rfc_accuracy", acc)
        mlflow.log_metric("rfc_precision", prec)
        mlflow.log_metric("rfc_recall", rec)
        mlflow.log_metric("rfc_f1_score", f1)

        print(f"🎯 RFC – Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        # Sauvegarde locale
        model_dir = os.path.join("tmp_rfc_model", "rf_class") if env == "prod" else os.path.join("models", "rf_class")
        os.makedirs(model_dir, exist_ok=True)
        clf.save(model_dir)

        # Logging MLflow
        mlflow.log_artifacts(model_dir, artifact_path="rfc_model")

        if env == "prod":
            # Export vers GCS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gcs_model_uri = f"gs://df_traffic_cyclist1/models/rf_class/{timestamp}/rf_class/"
            cp_command = f"gsutil -m cp -r {model_dir} {gcs_model_uri}"
            cp_success = os.system(cp_command) == 0

            if cp_success:
                update_summary(
                    summary_path="gs://df_traffic_cyclist1/models/summary.json",
                    model_type="rf_class",
                    run_id=run.info.run_id,
                    model_uri=gcs_model_uri,
                    env=env,
                    test_mode=test_mode,
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    f1_score=f1
                )
                print(f"📤 Modèle Affluence exporté vers {gcs_model_uri}")
            else:
                print(f"❌ Échec upload modèle Affluence vers GCS")

            if os.path.exists("tmp_rfc_model"):
                shutil.rmtree("tmp_rfc_model")
                print("🧹 Dossier temporaire supprimé : tmp_rfc_model")
        else:
            print(f"✅ Modèle RFC sauvegardé localement dans : {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bike count models (RF + NN)")
    parser.add_argument('--model_test', action='store_true', help="Use 1000 samples for fast training")
    parser.add_argument('--env', default="dev", choices=["dev", "prod"], help="Choose 'dev' or 'prod'")
    args = parser.parse_args()

    data_path, artifact_path = setup_environment(args.env, args.model_test)
    print(f"✅ Environnement {args.env} configuré. Données : {data_path}")

    X, y = load_and_clean_data(data_path)
    print(f"📊 Données chargées : {X.shape[0]} lignes")

    train_rf(X, y, args.env, args.model_test)
    train_nn(X, y, args.env, args.model_test)

    df_raw = load_and_clean_data(data_path, preserve_target=True)
    train_rfc(df_raw, None, args.env, args.model_test)

    print(f"🏁 Entraînement terminé ({args.env})")
