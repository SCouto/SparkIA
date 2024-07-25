# Databricks notebook source
# DBTITLE 0,--i18n-2ab084da-06ed-457d-834a-1d19353e5c59
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Random Forests and Hyperparameter Tuning
# MAGIC
# MAGIC We'll check now how to tune RandomForest
# MAGIC
# MAGIC
# MAGIC ### Requeriments
# MAGIC
# MAGIC
# MAGIC Make sure to use a ML Cluster, for instance 12.2 LTS ML

# COMMAND ----------

# DBTITLE 0,--i18n-67393595-40fc-4274-b9ed-40f8ef4f7db1
# MAGIC %md 
# MAGIC
# MAGIC ## Build a Model Pipeline
# MAGIC
# MAGIC
# MAGIC Let's load the clean Airbnb dataset in again 
# MAGIC
# MAGIC We created it in a previous notebook, it should exists in `dbfs:/FileStore/output/airbnb/clean_data`

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

file_path = f"dbfs:/FileStore/output/airbnb/clean_data"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=250)
stages = [string_indexer, vec_assembler, rf]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

# DBTITLE 0,--i18n-4561938e-90b5-413c-9e25-ef15ba40e99c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## ParamGrid
# MAGIC
# MAGIC

# COMMAND ----------

print(rf.explainParams())

# COMMAND ----------

# DBTITLE 0,--i18n-819de6f9-75d2-45df-beb1-6b59ecd2cfd2
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
# MAGIC
# MAGIC We can define a grid of hyperparameters to test:
# MAGIC   - **`maxDepth`**: max depth of each decision tree between **`2 and 5`**)
# MAGIC   - **`numTrees`**: number of decision trees to train between **`5 and 10`**)
# MAGIC
# MAGIC **`addGrid()`** requires the name of the parameter as first input (e.g. **`rf.maxDepth`**), and a list of the possible values (e.g. **`[2, 5]`**).

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5])
              .addGrid(rf.numTrees, [5, 10])
              .build())

# COMMAND ----------

# DBTITLE 0,--i18n-9f043287-11b8-482d-8501-2f7d8b1458ea
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Cross Validation
# MAGIC
# MAGIC We are also going to use 3-fold cross validation to identify the optimal hyperparameters.
# MAGIC
# MAGIC ![crossValidation](https://files.training.databricks.com/images/301/CrossValidation.png)
# MAGIC
# MAGIC With 3-fold cross-validation: 
# MAGIC *  We train on 2/3 of the data
# MAGIC * We evaluate in the other third 
# MAGIC
# MAGIC
# MAGIC Each fold gets the chance to act as the validation set. We then average the results of the three rounds.

# COMMAND ----------

# DBTITLE 0,--i18n-ec0440ab-071d-4201-be86-5eeedaf80a4f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC We pass in the **`estimator`** (pipeline), **`evaluator`**, and **`estimatorParamMaps`** to <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator" target="_blank">CrossValidator</a> so that it knows:
# MAGIC - Which model to use
# MAGIC - How to evaluate the model
# MAGIC - What hyperparameters to set for the model
# MAGIC
# MAGIC We can also set the number of folds we want to split our data into (3), as well as setting a seed so we all have the same split in the data.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

cv = CrossValidator(estimator=pipeline,
                    evaluator=evaluator,
                    estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)

# COMMAND ----------

# DBTITLE 0,--i18n-673c9261-a861-4ace-b008-c04565230a8e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Question**: How many models are we training right now?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This param grid will cause to evaluate the model with the following hyperparameters combination:
# MAGIC
# MAGIC * maxDepth: 2 numTrees: 5
# MAGIC * maxDepth: 2 numTrees: 10
# MAGIC * maxDepth: 5 numTrees: 5
# MAGIC * maxDepth: 5 numTrees: 10
# MAGIC
# MAGIC So four combinations

# COMMAND ----------

cv_model = cv.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-2d00b40f-c5e7-4089-890b-a50ccced34c6
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Question**: Should we put the pipeline in the cross validator, or the cross validator in the pipeline?
# MAGIC
# MAGIC Since we have things like StringIndexer (an estimator) in the pipeline, it will be recalculated entirely if the pipeline is put in the cross validator. And this is not of importance for the cross validation step.
# MAGIC
# MAGIC * We can put the only piece that changes (Regressor itself) in the cross validation
# MAGIC
# MAGIC The safest thing is to put the pipeline inside the CV, not the other way. CV first splits the data and then .fit() the pipeline. If it is placed at the end of the pipeline, we potentially can leak the info from hold-out set to train set.

# COMMAND ----------

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)

pipeline = Pipeline(stages=[string_indexer, vec_assembler, cv])

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-dede990c-2551-4c07-8aad-d697ae827e71
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can look at the model with the best hyperparameter configuration by checking it's metrics

# COMMAND ----------

list(zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics))

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

rmse = evaluator.evaluate(pred_df)
r2 = evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
