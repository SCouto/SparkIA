# Databricks notebook source
# DBTITLE 0,--i18n-b9944704-a562-44e0-8ef6-8639f11312ca
# MAGIC %md 
# MAGIC # XGBoost
# MAGIC
# MAGIC We'll use in this notebook an external library
# MAGIC  
# MAGIC

# COMMAND ----------

# DBTITLE 0,--i18n-3e08ca45-9a00-4c6a-ac38-169c7e87d9e4
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Load Data & Preparation
# MAGIC
# MAGIC Let's load the clean Airbnb dataset in again , but this time we will split in 3, so we have a validation set as well as a training and test sets
# MAGIC
# MAGIC We created it in a previous notebook, it should exists in `dbfs:/FileStore/output/airbnb/clean_data` 
# MAGIC
# MAGIC Also, let's index all of our categorical features, and set our label to be **`log(price)`**.

# COMMAND ----------

# DBTITLE 1,Load Data
from pyspark.sql.functions import log, col
from pyspark.ml.feature import StringIndexer, VectorAssembler

file_path = "dbfs:/FileStore/output/airbnb/clean_data"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.withColumn(<TODO>, <TODO>).randomSplit([.8, .2], seed=42)

#Select all categorical columns and index them 
categorical_cols = <TODO>
index_output_cols = <TODO>

string_indexer = StringIndexer(inputCols=<TODO>, outputCols=<TODO>, handleInvalid="skip")

#Select all numeric columns except price and label
numeric_cols = <TODO>

assembler_inputs = <TODO>
vec_assembler = VectorAssembler(inputCols=<TODO>, outputCol=<TODO>)

# COMMAND ----------

# DBTITLE 0,--i18n-733cd880-143d-42c2-9f29-602e48f60efe
# MAGIC %md 
# MAGIC ### Distributed Training of XGBoost Models
# MAGIC
# MAGIC We create an SparkXGBRegressor with the following params as a json
# MAGIC
# MAGIC * n_estimators: 100
# MAGIC * learning_rate: 0.1
# MAGIC * max_depth: 4
# MAGIC * random_state:42
# MAGIC * missing:0
# MAGIC

# COMMAND ----------

# DBTITLE 1,Regressor
from xgboost.spark import SparkXGBRegressor
from pyspark.ml import Pipeline

params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42, "missing": 0}

xgboost = SparkXGBRegressor(**params)

pipeline = Pipeline(stages=[<TODO>])
pipeline_model = pipeline.fit(<TODO>)

# COMMAND ----------

# DBTITLE 0,--i18n-8d5f8c24-ee0b-476e-a250-95ce2d73dd28
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate Model Performance
# MAGIC
# MAGIC * Remember to exponentiate the label column so we can evaluate the real prediction
# MAGIC * Set the exp(prediction) in a new column called just prediction
# MAGIC

# COMMAND ----------

# DBTITLE 1,Exponentiate prediction
from pyspark.sql.functions import exp, col

log_pred_df = pipeline_model.transform(<TODO>)

exp_xgboost_df = log_pred_df.withColumn(<TODO>, <TODO>)

display(exp_xgboost_df.select(<TODO>, <TODO>))

# COMMAND ----------

# DBTITLE 1,Evaluate performance
from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = <TODO>

rmse = regression_evaluator.evaluate(<TODO>)
r2 = regression_evaluator.setMetricName(<TODO>).evaluate(<TODO>)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")