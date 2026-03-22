# Databricks notebook source
# DBTITLE 0,--i18n-3bdc2b9e-9f58-4cb7-8c55-22bade9f79df
# MAGIC %md 
# MAGIC # Decision Trees
# MAGIC
# MAGIC Another method different than linear regression
# MAGIC
# MAGIC
# MAGIC ### Requirements
# MAGIC
# MAGIC Make sure to use a ML Cluster, for instance 12.2 LTS ML
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Let's load the clean Airbnb dataset in again 
# MAGIC
# MAGIC We created it in the previous notebook, it should exists in `dbfs:/FileStore/output/airbnb/clean_data`

# COMMAND ----------

# DBTITLE 1,Load Dataset
file_path = f"dbfs:/FileStore/output/airbnb/clean_data"
airbnb_df = spark.read.format("delta").load(file_path)

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# DBTITLE 1,Handling Categorical features
from pyspark.ml.feature import StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

# COMMAND ----------

# DBTITLE 0,--i18n-35e2f231-2ebb-4889-bc55-089200dd1605
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## VectorAssembler
# MAGIC
# MAGIC Let's use the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a> to combine all of our categorical and numeric inputs.

# COMMAND ----------

# DBTITLE 1,Vector Assembler
from pyspark.ml.feature import VectorAssembler

# Filter for just numeric columns (and exclude price, the target column)
numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]

# Combine output of StringIndexer defined above and numeric columns
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

# DBTITLE 0,--i18n-2096f7aa-7fab-4807-b45f-fcbd0424a3e8
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Decision Tree
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Regresor
from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(labelCol="price")

# COMMAND ----------

# DBTITLE 0,--i18n-506ab7fa-0952-4c55-ad9b-afefb6469380
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Training model with Pipeline
# MAGIC
# MAGIC The following cell is expected to error, but we subsequently fix this.

# COMMAND ----------

# DBTITLE 1,Training to fail
from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [string_indexer, vec_assembler, dt]
pipeline = Pipeline(stages=stages)

# Train model with the train set
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 1,Checking which feature
assembler_inputs

# COMMAND ----------

# DBTITLE 1,set maxBins
dt.setMaxBins(250)

# COMMAND ----------

# DBTITLE 1,training for real
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-2426e78b-9bd2-4b7d-a65b-52054906e438
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Feature Importance
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,visualizing tree
dt_model = pipeline_model.stages[-1]
display(dt_model)

# COMMAND ----------

# DBTITLE 1,featureImportance
dt_model.featureImportances

# COMMAND ----------

# DBTITLE 0,--i18n-823c20ff-f20b-4853-beb0-4b324debb2e6
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Interpreting Feature Importance
# MAGIC
# MAGIC It's complicated to interprete features by number, let's zip it with vec_assembler to name them

# COMMAND ----------

# DBTITLE 1,Beautify Feature importance
import pandas as pd

features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), dt_model.featureImportances)), columns=["feature", "importance"])
features_df

# COMMAND ----------

# DBTITLE 0,--i18n-1fe0f603-add5-4904-964b-7288ae98b2e8
# MAGIC %md 
# MAGIC
# MAGIC # Only a handful of features are > 0
# MAGIC
# MAGIC this is because default **`maxDepth`** is 5, so there are only a few features that where considered
# MAGIC

# COMMAND ----------

# DBTITLE 1,Top 5 features
top_n = 5

top_features = features_df.sort_values(["importance"], ascending=False)[:top_n]["feature"].values
print(top_features)

# COMMAND ----------

# DBTITLE 0,--i18n-d9525bf7-b871-45c8-b0f9-dca5fd7ae825
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Scale Invariant
# MAGIC
# MAGIC With decision trees, the scale of the features does not matter. For example, it will split 1/3 of the data if that split point is 100 or if it is normalized to be .33. The only thing that matters is how many data points fall left and right of that split point - not the absolute value of the split point.
# MAGIC
# MAGIC This is not true for linear regression, and the default in Spark is to standardize first. Think about it: If you measure shoe sizes in American vs European sizing, the corresponding weight of those features will be very different even those those measures represent the same thing: the size of a person's foot!

# COMMAND ----------

# DBTITLE 0,--i18n-bad0dd6d-05ba-484b-90d6-cfe16a1bc11e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply model to test set

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

pred_df.select("features", "price", "prediction").orderBy("price", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can filter out those outliers
# MAGIC
# MAGIC

# COMMAND ----------

display(pred_df.select("features", "price", "prediction").orderBy("price", ascending=False).filter("prediction < 2000").filter("price <2000"))

# COMMAND ----------

# DBTITLE 0,--i18n-094553a3-10c0-4e08-9a58-f94430b4a512
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Pitfall
# MAGIC
# MAGIC What if we get a massive Airbnb rental? It was 20 bedrooms and 20 bathrooms. What will a decision tree predict?
# MAGIC
# MAGIC It turns out decision trees cannot predict any values larger than they were trained on. The max value in our training set was $10,000, so we can't predict any values larger than that.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
