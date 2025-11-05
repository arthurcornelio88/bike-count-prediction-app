-- Add champion_r2_baseline column to BigQuery monitoring_audit.logs table
-- Run this BEFORE deploying the updated airflow_exporter and DAG

-- Option 1: Add column to existing table (preserves data)
ALTER TABLE `datascientest-460618.monitoring_audit.logs`
ADD COLUMN IF NOT EXISTS champion_r2_baseline FLOAT64;

-- Option 2: If ALTER fails, recreate table (WARNING: DELETES DATA!)
-- DROP TABLE IF EXISTS `datascientest-460618.monitoring_audit.logs`;
--
-- CREATE TABLE `datascientest-460618.monitoring_audit.logs` (
--   timestamp TIMESTAMP NOT NULL,
--   drift_detected BOOLEAN NOT NULL,
--   rmse FLOAT64 NOT NULL,
--   r2 FLOAT64 NOT NULL,
--   fine_tune_triggered BOOLEAN NOT NULL,
--   fine_tune_success BOOLEAN NOT NULL,
--   model_improvement FLOAT64 NOT NULL,
--   env STRING NOT NULL,
--   error_message STRING,
--   double_evaluation_enabled BOOLEAN,
--   baseline_regression BOOLEAN,
--   r2_baseline FLOAT64,
--   r2_current FLOAT64,
--   r2_train FLOAT64,
--   deployment_decision STRING,
--   champion_r2_baseline FLOAT64,
--   model_uri STRING,
--   run_id STRING
-- );
