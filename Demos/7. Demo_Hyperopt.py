# Databricks notebook source
# DBTITLE 0,--i18n-1fa7a9c8-3dad-454e-b7ac-555020a4bda8
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Hyperopt
# MAGIC
# MAGIC Hyperopt is used in ML to optimize hyperparameters 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Make sure to use a *ML Cluster*, for instance *12.2 LTS ML*

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
categorical_cols = <TODO>
index_output_cols = <TODO>
string_indexer = <TODO>

#Extract numeric columns except the label
numeric_cols = <TODO>

#Join all columns together and vector assemble them
assembler_inputs = <TODO>
vec_assembler = VectorAssembler(inputCols=<TODO>, outputCol="features")

#Create RandomForest Regressor
rf = RandomForestRegressor(labelCol=<TODO>, maxBins=250, seed=42)

#Create Pipeline
pipeline = Pipeline(stages=[<TODO>])

#Create Regression Evauator
regression_evaluator = RegressionEvaluator(predictionCol=<TODO>, labelCol=<TODO>)

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

# DBTITLE 1,Defining Training Function
def objective_function(params):    
    
    # get the hyperparameters value from params dictionary
    max_depth = <TODO>
    num_trees = <TODO>

    #Set the value in the pipeline
    estimator = pipeline.copy({rf.maxDepth: <TODO>, rf.numTrees: <TODO>})
    model = estimator.fit(<TODO>)

    #Train model with val_df
    preds_df = model.transform(<TODO>)
    rmse = regression_evaluator.evaluate(<TODO>)

    #return the metric
    return rmse

# COMMAND ----------

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
    "max_depth": <TODO>,
    "num_trees": <TODO>
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

num_evals = 50
trials = Trials()
best_hyperparam = fmin(fn=<TODO>, 
                       space=<TODO>,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))


#Retrieve best hyperparams
best_max_depth = <TODO>
best_num_trees = <TODO>

print(f'best_max_depth: {best_max_depth}')
print(f'best_num_trees: {best_num_trees}')

#Recreate pipeline with them
estimator = pipeline.copy({rf.maxDepth: <TODO>, rf.numTrees: <TODO>})

# Combine train & validation together
combined_df = train_df.union(val_df) 

#Train model with the desired hyperparams
pipeline_model = estimator.fit(<TODO>)

#Apply model to test set
pred_df = pipeline_model.transform(<TODO>)

#Evaluate model
rmse = regression_evaluator.evaluate(<TODO>)
r2 = regression_evaluator.setMetricName("r2").evaluate(<TODO>)

print(f"R2 is {r2}")
print(f"RMSE is {rmse}")
