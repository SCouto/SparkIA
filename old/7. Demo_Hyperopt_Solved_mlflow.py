# Databricks notebook source
# DBTITLE 0,--i18n-1fa7a9c8-3dad-454e-b7ac-555020a4bda8
# MAGIC %md 
# MAGIC # Hyperopt
# MAGIC
# MAGIC
# MAGIC Hyperopt is used in ML to optimize hyperparameters 
# MAGIC

# COMMAND ----------

# DBTITLE 0,--i18n-2340cdf4-9753-41b4-a613-043b90f0f472
# MAGIC %md 
# MAGIC
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Let's load the clean Airbnb dataset in again , but this time we will split in 3, so we have a validation set as well as a training and test sets
# MAGIC
# MAGIC We created it in a previous notebook, it should exists in `dbfs:/FileStore/output/airbnb/clean_data`

# COMMAND ----------

# DBTITLE 1,Load Data
file_path = f"dbfs:/FileStore/output/airbnb/clean_data"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, val_df, test_df = airbnb_df.randomSplit([.6, .2, .2], seed=42)

# COMMAND ----------

# DBTITLE 1,Building Model Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

#Extract, prepare and index Categorical Columns
categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

#Extract numeric columns except the label
numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]

#Join all columns together and vector assemble them
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

#Create RandomForest Regressor
rf = RandomForestRegressor(labelCol="price", maxBins=250, seed=42)

#Create Pipeline
pipeline = Pipeline(stages=[string_indexer, vec_assembler, rf])

#Create Regression Evauator
regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price")

# COMMAND ----------

# DBTITLE 0,--i18n-e4627900-f2a5-4f65-881e-1374187dd4f9
# MAGIC %md 
# MAGIC
# MAGIC ## Hyperopt 
# MAGIC
# MAGIC ###Define *Objective Function*
# MAGIC
# MAGIC
# MAGIC First, define our **training objective function**. The objective function has two primary requirements:
# MAGIC
# MAGIC 1. An **input** **`params`** including hyperparameter values to use when training the model (`max_depth` & `num_trees`)
# MAGIC 2. An **output** containing a loss metric on which to optimize, we will use `rmse`
# MAGIC
# MAGIC We can copy the pipeline and updating some of the parameters
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Training Function
def objective_function(params):    
    
    # set the hyperparameters that we want to tune
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]

    with mlflow.start_run():
        estimator = pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees})
        model = estimator.fit(train_df)

        preds = model.transform(val_df)
        rmse = regression_evaluator.evaluate(preds)
        mlflow.log_metric("rmse", rmse)

    return rmse

# COMMAND ----------

# DBTITLE 0,--i18n-d4f9dd2b-060b-4eef-8164-442b2be242f4
# MAGIC %md
# MAGIC Create a search space for
# MAGIC
# MAGIC * max_depth: min=2, max=10, q=1
# MAGIC * num_trees: min=10, max=200, q=1
# MAGIC
# MAGIC **This will fail if you are not in a ML cluster**

# COMMAND ----------

# DBTITLE 1,Search Space
from hyperopt import hp

#Defining search space
#hp.quniform("name", min, max, q),

search_space = {
    "max_depth": hp.quniform("max_depth", 2, 10, 1),
    "num_trees": hp.quniform("num_trees", 10, 200, 1)
}

# COMMAND ----------

# DBTITLE 0,--i18n-27891521-e481-4734-b21c-b2c5fe1f01fe
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC **`fmin()`** generates new hyperparameter configurations to use for your **`objective_function`**. It will evaluate as many models as max_evals parameters, using the information from the previous models to make a more informative decision for the next hyperparameter to try. 
# MAGIC

# COMMAND ----------

from hyperopt import fmin, tpe, Trials
import numpy as np
import mlflow
import mlflow.spark
mlflow.pyspark.ml.autolog(log_models=False)

num_evals = 5
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))

# Retrain model on train & validation dataset and evaluate on test dataset
with mlflow.start_run():
    best_max_depth = best_hyperparam["max_depth"]
    best_num_trees = best_hyperparam["num_trees"]
    print(f'best_max_depth: {best_max_depth}')
    print(f'best_num_trees: {best_num_trees}')

    estimator = pipeline.copy({rf.maxDepth: best_max_depth, rf.numTrees: best_num_trees})
    combined_df = train_df.union(val_df) # Combine train & validation together

    pipeline_model = estimator.fit(combined_df)
    pred_df = pipeline_model.transform(test_df)
    rmse = regression_evaluator.evaluate(pred_df)

    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_max_depth)
    mlflow.log_param("numTrees", best_num_trees)
    mlflow.log_metric("rmse", rmse)
    mlflow.spark.log_model(pipeline_model, "model")
