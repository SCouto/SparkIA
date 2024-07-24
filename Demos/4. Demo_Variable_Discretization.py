# Databricks notebook source
# DBTITLE 0,--i18n-60a5d18a-6438-4ee3-9097-5145dc31d938
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Linear Regression: Improving the Model with Categorical Variables
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cleaning output
model_output = "dbfs:/FileStore/output/airbnb/models/lr-model/"
dbutils.fs.rm(model_output, True)

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
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)


# COMMAND ----------

# DBTITLE 0,--i18n-09003d63-70c1-4fb7-a4b7-306101a88ae3
# MAGIC %md 
# MAGIC
# MAGIC ### One Hot Encoder
# MAGIC
# MAGIC * Extract all the string variables
# MAGIC * Create a column with Index for each of them to serve as the output of the StringIndexer
# MAGIC * Create a column with OHE for each of them to serve as the output of the One-Hot Encoder
# MAGIC * You need to use [StringIndexer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer) in order to map a string column of labels to an ML column of label indices.
# MAGIC *  Then, apply the [OneHotEncoder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html?highlight=onehotencoder#pyspark.ml.feature.OneHotEncoder) to the output of the StringIndexer.

# COMMAND ----------

# DBTITLE 1,String Indexer Example
from pyspark.ml.feature import StringIndexer

df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])

print("original")
df.show()
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
print("indexed")
indexed.show()

# COMMAND ----------

# DBTITLE 1,One Hot Encoder Example
from pyspark.ml.feature import OneHotEncoder

df = spark.createDataFrame([
    (0.0, 1.0),
    (1.0, 0.0),
    (2.0, 1.0),
    (0.0, 2.0),
    (0.0, 1.0),
    (2.0, 0.0)
], ["categoryIndex1", "categoryIndex2"])

print("original")
df.show()

encoder = OneHotEncoder(inputCols=["categoryIndex1", "categoryIndex2"],
                        outputCols=["categoryVec1", "categoryVec2"])
model = encoder.fit(df)
encoded = model.transform(df)
encoded.show()

# COMMAND ----------

# DBTITLE 1,One hot encoding variables
from pyspark.ml.feature import OneHotEncoder, StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=<TODO>, outputCols=<TODO>, handleInvalid="skip")
ohe_encoder = OneHotEncoder(inputCols=<TODO>, outputCols=<TODO>)

# COMMAND ----------

# DBTITLE 0,--i18n-dedd7980-1c27-4f35-9d94-b0f1a1f92839
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC Now you should combine our OHE categorical features with our numeric features.
# MAGIC  * Extract numeric columns (except price, since it is the target one)
# MAGIC  * Use Vector Assembler once again to create the features vector

# COMMAND ----------

# DBTITLE 1,Vectorizing features
from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = ohe_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=<TODO>, outputCol=<TODO>)

# COMMAND ----------

# DBTITLE 0,--i18n-fb06fb9b-5dac-46df-aff3-ddee6dc88125
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Linear Regression
# MAGIC
# MAGIC * Build Linear Regression object with price as label

# COMMAND ----------

# DBTITLE 1,Creating LR object
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol=<TODO>, featuresCol=<TODO>)

# COMMAND ----------

# DBTITLE 0,--i18n-a7aabdd1-b384-45fc-bff2-f385cc7fe4ac
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Pipeline
# MAGIC
# MAGIC  A [Pipeline](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html?highlight=pipeline#pyspark.ml.Pipeline) is a way of organizing all of the steps.
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Pipelining
from pyspark.ml import Pipeline

#stages should be indexer, ohe, vectorizing and finally linear regression
stages = [<TODO>]
pipeline = Pipeline(stages=<TODO>)

pipeline_model = pipeline.fit(<TODO>)


# COMMAND ----------

# DBTITLE 0,--i18n-c7420125-24be-464f-b609-1bb4e765d4ff
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Saving Models
# MAGIC
# MAGIC Training a model may be costly, so we can save it in case our cluster goes down so we don't have to recompute our results.

# COMMAND ----------

# DBTITLE 1,Saving model
pipeline_model.write().overwrite().save(model_output)

# COMMAND ----------

model_output

# COMMAND ----------

# DBTITLE 0,--i18n-15f4623d-d99a-42d6-bee8-d7c4f79fdecb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Loading models
# MAGIC
# MAGIC
# MAGIC If all your transformers/estimators are set into a Pipeline, and stored like that, you can always load the generic `PipelineModel` back in. Otherwise, you may need to know which kind of model was stored (LinearRegression, LogisticRegression etc...)

# COMMAND ----------

# DBTITLE 1,Loading Model
from pyspark.ml import PipelineModel

saved_pipeline_model = PipelineModel.load(<TODO>)

# COMMAND ----------

# DBTITLE 0,--i18n-1303ef7d-1a57-4573-8afe-561f7730eb33
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply the Model to Test Set

# COMMAND ----------

# DBTITLE 1,Applying model
pred_df = saved_pipeline_model.transform(<TODO>)

display(pred_df.select("features", "price", "prediction"))

# COMMAND ----------

# DBTITLE 0,--i18n-9497f680-1c61-4bf1-8ab4-e36af502268d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate the Model
# MAGIC

# COMMAND ----------

# DBTITLE 1,Evaluating model
display(pred_df.select("price", "prediction"))

# COMMAND ----------

# DBTITLE 1,Mathematical evaluation
from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol=<TODO>, labelCol=<TODO>, metricName=<TODO>)

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName(<TODO>).evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
