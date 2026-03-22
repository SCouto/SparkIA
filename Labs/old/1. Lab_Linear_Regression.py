# Databricks notebook source
# DBTITLE 0,--i18n-45bb1181-9fe0-4255-b0f0-b42637fc9591
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC #Linear Regression Lab
# MAGIC
# MAGIC In the previous lesson, we predicted price using just one variable: bedrooms. Now, we want to predict price given a few other features.
# MAGIC
# MAGIC Steps:
# MAGIC 1. Create Vector for **`bedrooms`**, **`bathrooms`**, **`bathrooms_na`**, **`minimum_nights`**, and **`number_of_reviews`** with VectorAssembler
# MAGIC 1. Build a Linear Regression Model
# MAGIC 1. Evaluate the **`RMSE`** and the **`R2`**.
# MAGIC

# COMMAND ----------

# DBTITLE 0,--i18n-6b124bfe-8486-4607-b911-9fc433159633
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

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

#Create vec assembler and apply to both train_df and test_df
vec_assembler = <TODO>


#Create Linear Regression object and train with the vectorized train_df 
lr_model = <TODO>

#Apply model to the vectorized test_df
pred_df = <TODO>

#Create a regresion evaluator with the proper predictionCol, the proper labelCol and MetricName rmse
regression_evaluator = <TODO>

#Calculate RMSE
rmse = <TODO>

#Calculate R2 updating Metric Name of the previous evaluator to R2
r2 = <TODO>


print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# DBTITLE 1,Gathering coefficients
for col, coef in zip(vec_assembler.getInputCols(), lr_model.coefficients):
    print(col, coef)
  
print(f"intercept: {lr_model.intercept}")
