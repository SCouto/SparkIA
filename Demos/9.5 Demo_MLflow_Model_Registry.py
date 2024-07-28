# Databricks notebook source
# DBTITLE 0,--i18n-2cf41655-1957-4aa5-a4f0-b9ec55ea213b
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # MLflow Lab Model Registry
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 0,--i18n-a5e8f936-8a26-47c1-b388-476c16d4addb
# MAGIC %md 
# MAGIC ## Steps
# MAGIC
# MAGIC
# MAGIC 1. Load in Airbnb dataset, and save both training dataset and test dataset as Delta tables
# MAGIC 1. Train an MLlib linear regression model using all the listing features and tracking parameters, metrics artifacts and Delta table version to MLflow
# MAGIC 1. Register this initial model and move it to staging using MLflow Model Registry
# MAGIC 1. Add a new column, **`log_price`** to both our train and test table and update the corresponding Delta tables
# MAGIC 1. Train a second MLlib linear regression model, this time using **`log_price`** as our target and training on all features, tracking to MLflow 
# MAGIC 1. Compare the performance of the different runs by looking at the underlying data versions for both models
# MAGIC 1. Move the better performing model to production in MLflow model registry

# COMMAND ----------


train_delta_path = "dbfs:/FileStore/output/airbnb/train.delta"
test_delta_path = "dbfs:/FileStore/output/airbnb/test.delta"


dbutils.fs.rm(train_delta_path, True)
dbutils.fs.rm(test_delta_path, True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Let's load the clean Airbnb dataset in again 
# MAGIC
# MAGIC We created it in the previous notebook, it should exists in `dbfs:/FileStore/output/airbnb/clean_data`

# COMMAND ----------

file_path = f"dbfs:/FileStore/output/airbnb/clean_data"
airbnb_df = spark.read.format("delta").load(file_path)

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# DBTITLE 0,--i18n-197ad07c-dead-4444-82de-67353d81dcb0
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##  Step 1. Creating Delta Tables
# MAGIC
# MAGIC We will create a Delta table for each of the datasets

# COMMAND ----------


# In case paths already exists
dbutils.fs.rm(train_delta_path, True)
dbutils.fs.rm(test_delta_path, True)

train_df.write.mode("overwrite").format("delta").save(train_delta_path)
test_df.write.mode("overwrite").format("delta").save(test_delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's read the first version of each delta table

# COMMAND ----------

data_version = 0
train_delta = spark.read.format("delta").option("versionAsOf", data_version).load(train_delta_path)  
test_delta = spark.read.format("delta").option("versionAsOf", data_version).load(test_delta_path)

# COMMAND ----------

# DBTITLE 0,--i18n-2bf375c9-fb36-47a3-b973-82fa805e8b22
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Review Delta Table History
# MAGIC All the transactions for this table are stored within this table including the initial set of insertions, update, delete, merge, and inserts.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`"))

# COMMAND ----------

# DBTITLE 0,--i18n-ffe159a7-e5dd-49fd-9059-e399237005a7
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Step 2. Log Initial Run to MLflow
# MAGIC
# MAGIC Let's first log a run to MLflow where we use all features. We use the same approach with RFormula as before. This time however, let's also log both the version of our data and the data path to MLflow.

# COMMAND ----------


import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import RFormula

with mlflow.start_run(run_name="lr_model") as run:
    # Log parameters
    mlflow.log_param("label", "price-all-features")
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_path", train_delta_path)    

    # Create pipeline
    r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")
    lr = LinearRegression(labelCol="price", featuresCol="features")
    pipeline = Pipeline(stages = [r_formula, lr])
    model = pipeline.fit(train_delta)

    # Log pipeline
    mlflow.spark.log_model(model, "model")

    # Create predictions and metrics
    pred_df = model.transform(test_delta)
    regression_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
    rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 0,--i18n-3bac0fef-149d-4c25-9db1-94fbcd63ba13
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Step 3. Register Model and Move to Staging Using MLflow Model Registry
# MAGIC
# MAGIC We are happy with the performance of the above model and want to move it to staging. Let's create the model and register it to the MLflow model registry.

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"

suffix = "scc"
model_name = f"mllib-lr_{suffix}"
print(f"Model Name: {model_name}\n")

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# DBTITLE 0,--i18n-78b33d27-0815-4d31-80a0-5e110aa96224
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Transition model to staging.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# COMMAND ----------

# Define a utility method to wait until the model is ready
def wait_for_model(model_name, version, stage="None", status="READY", timeout=300):
    import time

    last_stage = "unknown"
    last_status = "unknown"

    for i in range(timeout):
        model_version_details = client.get_model_version(name=model_name, version=version)
        last_stage = str(model_version_details.current_stage)
        last_status = str(model_version_details.status)
        if last_status == str(status) and last_stage == str(stage):
            return

        time.sleep(1)

    raise Exception(f"The model {model_name} v{version} was not {status} after {timeout} seconds: {last_status}/{last_stage}")

# COMMAND ----------

# Force our notebook to block until the model is ready
wait_for_model(model_name, 1, stage="Staging")

# COMMAND ----------

# DBTITLE 0,--i18n-b5f74e40-1806-46ab-9dd0-97b82d8f297e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Add a model description using <a href="https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.update_registered_model" target="_blank">update_registered_model</a>.

# COMMAND ----------


client.update_registered_model(
    name=model_details.name,
    description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

# COMMAND ----------

wait_for_model(model_details.name, 1, stage="Staging")

# COMMAND ----------

# DBTITLE 0,--i18n-03dff1c0-5c7b-473f-83ec-4a8283427280
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##  Step 4. Feature Engineering: Evolve Data Schema
# MAGIC
# MAGIC We now want to do some feature engineering with the aim of improving model performance; we can use Delta Lake to track older versions of the dataset. 
# MAGIC
# MAGIC We will add **`log_price`** as a new column and update our Delta table with it.

# COMMAND ----------

from pyspark.sql.functions import col, log, exp

# Create a new log_price column for both train and test datasets
train_new = train_delta.withColumn("log_price", log(col("price")))
test_new = test_delta.withColumn("log_price", log(col("price")))

# COMMAND ----------

# DBTITLE 0,--i18n-565313ed-2bca-4cc6-af87-1c0d509c0a69
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Save the updated DataFrames to **`train_delta_path`** and **`test_delta_path`**, respectively, passing the **`mergeSchema`** option to safely evolve its schema. 
# MAGIC
# MAGIC Take a look at this <a href="https://databricks.com/blog/2019/09/24/diving-into-delta-lake-schema-enforcement-evolution.html" target="_blank">blog</a> on Delta Lake for more information about **`mergeSchema`**.

# COMMAND ----------

train_new.write.option("mergeSchema", "true").format("delta").mode("overwrite").save(train_delta_path)
test_new.write.option("mergeSchema", "true").format("delta").mode("overwrite").save(test_delta_path)

# COMMAND ----------

# DBTITLE 1,--i18n-0c7c986b-1346-4ff1-a4e2-ee190891a5bf
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Reviewing delta history

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`"))

# COMMAND ----------

data_version = 1
train_delta_new = spark.read.format("delta").option("versionAsOf", data_version).load(train_delta_path)  
test_delta_new = spark.read.format("delta").option("versionAsOf", data_version).load(test_delta_path)

# COMMAND ----------

# DBTITLE 0,--i18n-f29c99ca-b92c-4f74-8bf5-c74070a8cd50
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Step 5. Use **`log_price`** as Target and Track Run with MLflow
# MAGIC
# MAGIC Retrain the model on the updated data and compare its performance to the original, logging results to MLflow.

# COMMAND ----------

with mlflow.start_run(run_name="lr_log_model") as run:
    # Log parameters
    mlflow.log_param("label", "log-price")
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_path", train_delta_path)    

    # Create pipeline
    r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip")  
    lr = LinearRegression(labelCol="log_price", predictionCol="log_prediction")
    pipeline = Pipeline(stages = [r_formula, lr])
    pipeline_model = pipeline.fit(train_delta_new)

    # Log model and update the registered model
    mlflow.spark.log_model(
        spark_model=pipeline_model,
        artifact_path="log-model",
        registered_model_name=model_name
    )  

    # Create predictions and metrics
    pred_df = pipeline_model.transform(test_delta)
    exp_df = pred_df.withColumn("prediction", exp(col("log_prediction")))
    rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)  

    run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 0,--i18n-e5bd7bfb-f445-44b5-a272-c6ae2849ac9f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Step 6. Compare Performance Across Runs by Looking at Delta Table Versions 
# MAGIC
# MAGIC Use MLflow's <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs" target="_blank">**`mlflow.search_runs`**</a> API to identify runs according to the version of data the run was trained on. Let's compare our runs according to our data versions.
# MAGIC
# MAGIC Filter based on **`params.data_path`** and **`params.data_version`**.

# COMMAND ----------

data_version = 0

mlflow.search_runs(filter_string=f"params.data_path='{train_delta_path}' and params.data_version='{data_version}'")

# COMMAND ----------


data_version = 1

mlflow.search_runs(filter_string=f"params.data_path='{train_delta_path}' and params.data_version='{data_version}'")

# COMMAND ----------

# DBTITLE 0,--i18n-3056bfcc-7623-4410-8b1b-82cba24ae3dd
# MAGIC %md 
# MAGIC
# MAGIC ## Step 7. Move the Best Performing Model to Production Using MLflow Model Registry
# MAGIC
# MAGIC Get the most recent model version and move it to production.

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="This model version was built using a MLlib Linear Regression model with all features and log_price as predictor."
)

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=new_model_version)
model_version_details.status

# COMMAND ----------

wait_for_model(model_name, new_model_version)

# COMMAND ----------


client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production"
)

# COMMAND ----------

wait_for_model(model_name, new_model_version, "Production")

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Archived"
)

# COMMAND ----------

wait_for_model(model_name, 1, "Archived")

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived"
)

# COMMAND ----------

wait_for_model(model_name, 2, "Archived")

# COMMAND ----------

client.delete_registered_model(model_name)
