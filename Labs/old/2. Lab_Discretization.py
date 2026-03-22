# Databricks notebook source
# DBTITLE 0,--i18n-4e1b9835-762c-42f2-9ff8-75164cb1a702
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Categorical Variables Discretization
# MAGIC
# MAGIC * Include categorical variables in the model

# COMMAND ----------

# DBTITLE 0,--i18n-1500312a-d027-42d0-a787-0dea4f8d7d03
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

# DBTITLE 0,--i18n-a427d25c-591f-4899-866a-14064eff40e3
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## RFormula
# MAGIC
# MAGIC To avoid manually specifying which columns are categorical to the StringIndexer and OneHotEncoder [RFormula](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.RFormula.html?highlight=rformula#pyspark.ml.feature.RFormula) can do that by itself
# MAGIC
# MAGIC Any String column will be consider a Categorial Feature, indexed and one-hot encoded. Then it will combine them with the byumeric features into a single vector called features
# MAGIC

# COMMAND ----------

# DBTITLE 1,RFormula Example
from pyspark.ml.feature import RFormula

dataset = spark.createDataFrame(
    [(7, "US", 18, 1.0),
     (8, "CA", 12, 0.0),
     (9, "NZ", 15, 0.0)],
    ["id", "country", "hour", "clicked"])

print("original")
dataset.show()
formula = RFormula(
    formula="clicked ~ country + hour",
    featuresCol="features",
    labelCol="label")

output = formula.fit(dataset).transform(dataset)
print("result")
output.select("features", "label").show()

# COMMAND ----------

# TODO
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

#Create the RFormula object with the following  parameters:
# formula= "price ~ ." That means that price is the label to predict based on all (.) the other columns
# features="features" Output name for the features
# handleInvalid = "skip" so errors are ignored
# labelCol= "price"
r_formula = RFormula(<TODO>)

#Create the linear regresor object indicating only the labelCol, since all the rest will be default
lr = <TODO>

#Create the pipeline, now there will only be 2 stages, since all is automated in the r_fomula, the other stage is the lr itself
pipeline = Pipeline(<TODO>)

#Train the model with the pipeline using the fit method with the proper dataframe
pipeline_model = <TODO>

#Apply the model to the proper testing dataframe with the transform method of the pipeline
pred_df = <TODO>

#Create a regresor evaluator with the predictionCol and the labelCol
regression_evaluator = <TODO>

rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

display(pred_df.select("price", "prediction"))
