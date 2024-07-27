# Databricks notebook source
# DBTITLE 0,--i18n-b9944704-a562-44e0-8ef6-8639f11312ca
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # XGBoost
# MAGIC
# MAGIC We'll use in this notebook an external library
# MAGIC  
# MAGIC
# MAGIC ### Requeriments
# MAGIC
# MAGIC
# MAGIC Make sure to use a ML Cluster, for instance 12.2 LTS ML

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

from pyspark.sql.functions import log, col
from pyspark.ml.feature import StringIndexer, VectorAssembler

file_path = "dbfs:/FileStore/output/airbnb/clean_data"
airbnb_df = spark.read.format("delta").load(file_path)

#split in train, val and test
train_df, val_df, test_df = airbnb_df.filter("price < 250").withColumn("label", log(col("price"))).randomSplit([.6, .2, .2], seed=42)

#Extract, prepare and index Categorical Columns
categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

#Extract numeric columns except the label and the price
numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price") & (field != "label"))]

#Join all columns together and vector assemble them
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
#Create Regression Evauator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")


# COMMAND ----------

# DBTITLE 0,--i18n-733cd880-143d-42c2-9f29-602e48f60efe
# MAGIC %md 
# MAGIC ### Distributed Training of XGBoost Models
# MAGIC
# MAGIC To create our distributed XGBoost model. We will use `xgboost`'s PySpark estimator. 
# MAGIC
# MAGIC We need to specify two additional parameters:
# MAGIC
# MAGIC * `num_workers`: The number of workers to distribute over.
# MAGIC * `use_gpu`: Enable to utilize GPU based training for faster performance.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Defining base params
from xgboost.spark import SparkXGBRegressor
from pyspark.ml import Pipeline

base_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42, "missing": 0}


# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperopt
# MAGIC

# COMMAND ----------

# DBTITLE 1,training function
from hyperopt import hp
from hyperopt import fmin, tpe, Trials
import numpy as np
from pyspark.sql.functions import exp
from pyspark.ml.evaluation import RegressionEvaluator


def objective_function(params):    
    
    # set the hyperparameters that we want to tune
    max_depth = int(params["max_depth"])
    n_estimators = int(params["n_estimators"])
    learning_rate = int(params["learning_rate"])


    updated_params = base_params.copy()
    updated_params.update({'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': learning_rate})
    

    xgboost = SparkXGBRegressor(**updated_params)
    estimator = Pipeline(stages=[string_indexer, vec_assembler, xgboost])

    model = estimator.fit(train_df)

    preds = model.transform(val_df).withColumn("prediction", exp(col("prediction")))
    rmse = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse").evaluate(preds)

    return rmse
  

# COMMAND ----------

# DBTITLE 1,Defining SearchSpace
search_space = {
    "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
    "max_depth": hp.quniform("max_depth", 2, 10, 1),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3)  
}

# COMMAND ----------

# DBTITLE 1,Calculating best hyperparams

num_evals = 5
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))


# COMMAND ----------


print(best_hyperparam)


best_params = base_params.copy()

best_params.update({'max_depth': int(best_hyperparam["max_depth"]),
                        'n_estimators': int(best_hyperparam["n_estimators"]),
                         'learning_rate': best_hyperparam["learning_rate"]})

#Create Regressor
xgboost = SparkXGBRegressor(**best_params)

#Create pipeline
pipeline = Pipeline(stages=[string_indexer, vec_assembler, xgboost])

#Combine train and val_df
combined_df = train_df.union(val_df) # Combine train & validation together


#Train model
pipeline_model = pipeline.fit(combined_df)


# COMMAND ----------

pred_df = pipeline_model.transform(test_df).withColumn("prediction", exp(col("prediction")))

rmse = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse").evaluate(pred_df)
r2 = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="r2").evaluate(pred_df)

print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
