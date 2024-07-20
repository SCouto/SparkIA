# Databricks notebook source
# DBTITLE 0,--i18n-62811f6d-e550-4c60-8903-f38d7ed56ca7
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Regression: Predicting Rental Price
# MAGIC
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices 
# MAGIC

# COMMAND ----------

# DBTITLE 0,--i18n-b44be11f-203c-4ea4-bc3e-20e696cabb0e
# MAGIC %md 
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Let's load the clean Airbnb dataset in again 
# MAGIC
# MAGIC We created it in the previous notebook, it should exists in `dbfs:/FileStore/output/airbnb/clean_data`

# COMMAND ----------

# DBTITLE 1,Load Data
file_path = f"dbfs:/FileStore/output/airbnb/clean_data"
airbnb_df = spark.read.format("delta").load(file_path)


# COMMAND ----------

# DBTITLE 1,Spliting in train and test
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------


from pyspark.sql.functions import avg, lit, median
from pyspark.ml.evaluation import RegressionEvaluator


avg_price = train_df.select(avg("price")).first()[0]
median_price = train_df.select(median("price")).first()[0]

pred_df = (test_df
          .withColumn("avgPrediction", lit(avg_price))
          .withColumn("medianPrediction", lit(median_price)))

regression_evaluatorAVG = RegressionEvaluator(predictionCol="avgPrediction", labelCol="price", metricName="rmse")
regression_evaluatorMedian = RegressionEvaluator(predictionCol="medianPrediction", labelCol="price", metricName="rmse")


rmseAVG = regression_evaluatorAVG.evaluate(pred_df)
print(f"RMSE for AVG is: {rmseAVG}")
rmseMedian = regression_evaluatorMedian.evaluate(pred_df)
print(f"RMSE for MEDIAN is {rmseMedian}")


r2Avg = regression_evaluatorAVG.setMetricName("r2").evaluate(pred_df)
print(f"R2 for AVG is {r2Avg}")
r2Median = regression_evaluatorMedian.setMetricName("r2").evaluate(pred_df)
print(f"R2 for Median is {r2Median}")

# COMMAND ----------

# DBTITLE 0,--i18n-5b96c695-717e-4269-84c7-8292ceff9d83
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Linear Regression
# MAGIC
# MAGIC Check **`price`** and **`bedrooms`** relations with a visualization
# MAGIC

# COMMAND ----------

# DBTITLE 1,price/bedroom visualization
display(train_df.select("price", "bedrooms"))

# COMMAND ----------

# DBTITLE 1,Summary
display(train_df.select("price", "bedrooms").summary())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Our dataset has a lot of columns, we'll be using only two of them for this notebook for the sake of simplicity
# MAGIC
# MAGIC * bedrooms: Feature
# MAGIC * price: Label

# COMMAND ----------

# DBTITLE 0,--i18n-4171a9ae-e928-41e3-9689-c6fcc2b3d57c
# MAGIC %md 
# MAGIC
# MAGIC We will use [LinearRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html?highlight=linearregression#pyspark.ml.regression.LinearRegression) to build the model.
# MAGIC
# MAGIC We will also use [VectorAssembler](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler) to build the feature column to the proper type
# MAGIC

# COMMAND ----------

# DBTITLE 1,Vector Assembler Sample
#Sample Vector Assembler

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

dataset = spark.createDataFrame(
    [(0, 18, 1.0, 1.0),(1, 22, 3.0, 5.0)],
    ["id", "hour", "mobile", "clicked"])

assembler = VectorAssembler(
    inputCols=["hour", "mobile"],
    outputCol="features")

print("original dataset")
dataset.show(truncate=False)

output = assembler.transform(dataset)

print("result dataset")
output.select("id", "features", "clicked").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Creación columna feature
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)

# COMMAND ----------

# DBTITLE 1,Priting schema
vec_train_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Creating and training model
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(vec_train_df)

# COMMAND ----------

# DBTITLE 1,Checking model params
print(lr.explainParams())

# COMMAND ----------

# DBTITLE 0,--i18n-ab8f4965-71db-487d-bbb3-329216580be5
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Inspect the Model
# MAGIC
# MAGIC We can extract the formula for the lineal regression where:
# MAGIC
# MAGIC * Formula: y = Coefficient * X + Intercept

# COMMAND ----------

m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

# DBTITLE 0,--i18n-ae6dfaf9-9164-4dcc-a699-31184c4a962e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply Model to Test Set
# MAGIC
# MAGIC * First transform the test_df with Vector Assembler as we did with train_df
# MAGIC * Instead of fit method, which is used to training, we use transform method from the model, it will create a column named prediction

# COMMAND ----------

# DBTITLE 1,Transform test_set as we did with train_set
vec_test_df = vec_assembler.transform(test_df)

pred_df = lr_model.transform(vec_test_df)


# COMMAND ----------

# DBTITLE 1,Checking prediction
display(pred_df.select("price", "prediction"))

# COMMAND ----------

# DBTITLE 0,--i18n-8d73c3ee-34bc-4f8b-b2ba-03597548680c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate the Model
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"R2 is {r2}")



#Baseline
#RMSE for AVG is: 57.72840841939142
#RMSE for MEDIAN is 58.37921776470596

#R2 for AVG is -5.960639805380197e-05
#R2 for Median is -0.022735334682391084


# COMMAND ----------

# DBTITLE 0,--i18n-703fbf0b-a2e1-4086-b002-8f63e06afdd8
# MAGIC %md 
# MAGIC
# MAGIC ### Result
# MAGIC
# MAGIC
# MAGIC
# MAGIC It is just a little better than the baselines
# MAGIC
# MAGIC 1523 vs 1528 (median)
# MAGIC 1523 vs 1525 (mean)
