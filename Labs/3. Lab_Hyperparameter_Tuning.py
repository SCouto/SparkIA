# Databricks notebook source
# DBTITLE 0,--i18n-b778c8d0-84e6-4192-a921-b9b60fd20d9b
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Hyperparameter Tuning with Random Forests
# MAGIC
# MAGIC In this lab, you will convert the Airbnb problem to a classification dataset, build a random forest classifier, and tune some hyperparameters of the random forest.

# COMMAND ----------

# DBTITLE 0,--i18n-02dc0920-88e1-4f5b-886c-62b8cc02d1bb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Classification
# MAGIC
# MAGIC In this lab, you will classify the listings between **high and low price.**  
# MAGIC
# MAGIC The **`class`** column will be:
# MAGIC
# MAGIC - **`0`** for a low cost listing of under $150
# MAGIC - **`1`** for a high cost listing of $150 or more
# MAGIC
# MAGIC The main difference with before is:
# MAGIC * We create a new column priceClass  with the values from before
# MAGIC * We drop the price column

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

file_path = f"dbfs:/FileStore/output/airbnb/clean_data"

airbnb_df = (spark
            .read
            .format("delta")
            .load(file_path)
            .withColumn(<TODO>)
            .drop(<TODO>)
           )

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

categorical_cols = <TODO>
index_output_cols = <TODO>

string_indexer = StringIndexer(inputCols=<TODO>, outputCols=<TODO>, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "priceClass"))]

assembler_inputs = <TODO>
vec_assembler = VectorAssembler(inputCols=<TODO>, outputCol="features")

# COMMAND ----------

# DBTITLE 0,--i18n-0e9bdc2f-0d8d-41cb-9509-47833d66bc5e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Random Forest
# MAGIC
# MAGIC Create a Random Forest classifer called **`rf`** with the **`labelCol=priceClass`**, **`maxBins=250`**, and **`seed=42`** (for reproducibility).
# MAGIC
# MAGIC It's under **`pyspark.ml.classification.RandomForestClassifier`** in Python.

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(<TODO>)


# COMMAND ----------

# DBTITLE 0,--i18n-7f3962e7-51b8-4477-9599-2465ab94a049
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Grid Search
# MAGIC
# MAGIC
# MAGIC Let's define a grid of hyperparameters to test:
# MAGIC   - maxDepth: max depth of the decision tree (Use **`2, 5, 10`**)
# MAGIC   - numTrees: number of decision trees (Use **`10, 20, 100`**)
# MAGIC
# MAGIC **`addGrid()`** accepts the name of the parameter (e.g. **`rf.maxDepth`**), and a list of the possible values (e.g. **`[2, 5, 10]`**).

# COMMAND ----------

# TODO
from pyspark.ml.tuning import ParamGridBuilder

param_grid = (ParamGridBuilder()
              <TODO>
              <TODO>
              .build())

# COMMAND ----------

# DBTITLE 0,--i18n-e1862bae-e31e-4f5a-ab0e-926261c4e27b
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluator
# MAGIC
# MAGIC In the past, we used a **`RegressionEvaluator`**.  For classification, we can use a <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html?highlight=binaryclass#pyspark.ml.evaluation.BinaryClassificationEvaluator" target="_blank">BinaryClassificationEvaluator</a> if we have two classes or <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html?highlight=multiclass#pyspark.ml.evaluation.MulticlassClassificationEvaluator" target="_blank">MulticlassClassificationEvaluator</a> for more than two classes.
# MAGIC
# MAGIC Create a **`BinaryClassificationEvaluator`** with **`areaUnderROC`** as the metric.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic" target="_blank">Read more on ROC curves here.</a>  In essence, it compares true positive and false positives.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator 

evaluator = BinaryClassificationEvaluator(labelCol=<TODO>, rawPredictionCol=<TODO>,metricName="areaUnderROC")


# COMMAND ----------

# DBTITLE 0,--i18n-ea1c0e11-125d-4067-bd70-0bd6c7ca3cdb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Cross Validation
# MAGIC
# MAGIC We are going to do **3-Fold** cross-validation and set the **`seed`**=42 on the cross-validator for reproducibility.
# MAGIC
# MAGIC Put the Random Forest in the CV to speed up the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator" target="_blank">cross validation</a> (as opposed to the pipeline in the CV).

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

cv = CrossValidator(estimator=<TODO>,
                    evaluator=<TODO>,
                    estimatorParamMaps=<TODO>, 
                    numFolds=<TODO>, seed=42)


# COMMAND ----------

# DBTITLE 0,--i18n-1f8cebd5-673c-4513-b73b-b64b0a56297c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Pipeline
# MAGIC
# MAGIC Let's fit the pipeline with our cross validator to our training data (this may take a few minutes).

# COMMAND ----------

stages = [<TODO>]

pipeline = Pipeline(stages=<TODO>)

pipeline_model = pipeline.fit(<TODO>)

# COMMAND ----------

# DBTITLE 0,--i18n-70cdbfa3-0dd7-4f23-b755-afc0dadd7eb2
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Hyperparameter
# MAGIC
# MAGIC Which hyperparameter combination performed the best?

# COMMAND ----------

cv_model = pipeline_model.stages[-1]
rf_model = cv_model.bestModel

# list(zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics))

print(rf_model.explainParams())

# COMMAND ----------

# DBTITLE 0,--i18n-11e6c47a-ddb1-416d-92a5-2f61340f9a5e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Feature Importance

# COMMAND ----------

import pandas as pd

pandas_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), rf_model.featureImportances)), columns=["feature", "importance"])
top_features = pandas_df.sort_values(["importance"], ascending=False)
top_features

# COMMAND ----------

# DBTITLE 0,--i18n-ae7e312e-d32b-4b02-97ff-ad4d2c737892
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Do those features make sense? Would you use those features when picking an Airbnb rental?

# COMMAND ----------

# DBTITLE 0,--i18n-950eb40f-b1d2-4e7f-8b07-76faff6b8186
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply Model to Test Set

# COMMAND ----------

pred_df = pipeline_model.transform(<TODO>)
area_under_roc = evaluator.evaluate(<TODO>)
print(f"Area under ROC is {area_under_roc:.2f}")

#I got 0.77
