# Databricks notebook source
# DBTITLE 0,--i18n-8c6d3ef3-e44b-4292-a0d3-1aaba0198525
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC # Data Cleansing
# MAGIC
# MAGIC We will be using Spark to perform exploratory analysis and cleaning Data from a public Airbnb Dataset.

# COMMAND ----------

# DBTITLE 1,Cleaning output data from previous executions
output_path = 'dbfs:/FileStore/output/airbnb/clean_data/'
dbutils.fs.rm(output_path, True)

# COMMAND ----------

# DBTITLE 0,--i18n-969507ea-bffc-4255-9a99-2306a594625f
# MAGIC %md 
# MAGIC
# MAGIC # Input Data
# MAGIC
# MAGIC Let's load the Airbnb dataset in.  
# MAGIC
# MAGIC You can download data from any city from <a href="http://insideairbnb.com/get-the-data.html" target="_blank">Airbnb website.</a> The dataset we are going to use is the Detailed Listings data called listings.csv.gz
# MAGIC
# MAGIC The one provided is the one from New York City, i recommend to store it in a folder such as:
# MAGIC
# MAGIC ```dbfs:/FileStore/input/airbnb/listingsNY.csv```
# MAGIC
# MAGIC Check the course documentation to know how to upload it.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Load Data
# MAGIC
# MAGIC First you need to load Data to a Dataframe, there are several options to consider: 
# MAGIC
# MAGIC * header: true if the data contains a header , otherwise the header will be considered Data (Check input file to know which value to use)
# MAGIC * inferSchema: true to let spark infer the schema instead of provider it's types mannually. Set it to true
# MAGIC * multiLine: true to allow multiline texts in records. Set it to true
# MAGIC * escape: Ignore special characters in data, set it to '\"'
# MAGIC * quote: Character that encloses each value, set it to '\"'
# MAGIC
# MAGIC
# MAGIC ## Expected output
# MAGIC
# MAGIC * 38199 rows for the provided Dataset (may differ if you downloaded another version of another city)
# MAGIC * Typed dataset (Ensure not all of the fields are strings)
# MAGIC * 75 columns

# COMMAND ----------

# DBTITLE 1,Load Data
from pyspark.sql.functions import split, col


file_path = f"dbfs:/FileStore/input/airbnb/listingsNY.csv"

raw_df = spark.read.csv(file_path,
                         header="true", 
                          inferSchema="true", 
                          multiLine="true", 
                          escape='\"', 
                          quote='\"')

raw_df.count()

# COMMAND ----------

# DBTITLE 1,Checking columns
raw_df.columns

# COMMAND ----------

# DBTITLE 0,--i18n-94856418-c319-4915-a73e-5728fcd44101
# MAGIC %md 
# MAGIC
# MAGIC # Cleaning columns
# MAGIC Keep only the following columns from this dataset.  Solo las que consideramos features y las que queremos predecir (precio)
# MAGIC
# MAGIC ```
# MAGIC columns_to_keep = [
# MAGIC     "host_is_superhost",
# MAGIC     "instant_bookable",
# MAGIC     "host_total_listings_count",
# MAGIC     "neighbourhood_cleansed",
# MAGIC     "latitude",
# MAGIC     "longitude",
# MAGIC     "property_type",
# MAGIC     "room_type",
# MAGIC     "accommodates",
# MAGIC     "bathrooms",
# MAGIC     "bedrooms",
# MAGIC     "beds",
# MAGIC     "minimum_nights",
# MAGIC     "number_of_reviews",
# MAGIC     "review_scores_rating",
# MAGIC     "review_scores_accuracy",
# MAGIC     "review_scores_cleanliness"`
# MAGIC     "review_scores_checkin",
# MAGIC     "review_scores_communication",
# MAGIC     "review_scores_location",
# MAGIC     "review_scores_value",
# MAGIC     "price"
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC ## Expected output
# MAGIC * 22 columns

# COMMAND ----------

# DBTITLE 1,Deleting unneceary columns
columns_to_keep = [
    "host_is_superhost",
    "instant_bookable",
    "host_total_listings_count",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "price"
]

base_df = raw_df.select(columns_to_keep)
base_df.cache().count()
display(base_df)

# COMMAND ----------

# DBTITLE 0,--i18n-a12c5a59-ad1c-4542-8695-d822ec10c4ca
# MAGIC %md 
# MAGIC
# MAGIC # Fixing Data Types
# MAGIC
# MAGIC If you check the schema from the previous dataframe, you will be able to see that the **`price`** field was understand as a price. This is a pretty commons error when dealing with monetary amounts due to currency sumbols. We need it to be a number. So:
# MAGIC
# MAGIC * Remove dollar sign with `translate` method  which replace the second parameter for the third one in the column indicated in the first parameter. In this case we should do `translate(col("price"), "$,", "")`
# MAGIC * cast column to double
# MAGIC * Use withColumn method for all that as follows: 
# MAGIC   *  `withColumn("price", translate(col("price), "$,", "").cast("double"))`
# MAGIC   
# MAGIC
# MAGIC
# MAGIC Once fixed the data type, we can build an histogram over the dataframe to see that there is a huge skew over the price column. Skew is one of the main issues in Spark.
# MAGIC
# MAGIC Skew is when data is not balanced and it doesnt follows a normal distribution

# COMMAND ----------

# DBTITLE 1,Fixing price
from pyspark.sql.functions import col, translate

fixed_price_df = base_df.withColumn("price", translate(col("price"), "$,", "").cast("double"))

display(fixed_price_df)

# COMMAND ----------

# DBTITLE 0,--i18n-4ad08138-4563-4a93-b038-801832c9bc73
# MAGIC %md 
# MAGIC
# MAGIC # Data statistics
# MAGIC
# MAGIC Two options:
# MAGIC * **`describe`**: Providers count, mean, stddev, min, max
# MAGIC * **`summary`**: Providers the ones from describe and the interquartile range (IQR) (25-50-75%)

# COMMAND ----------

# DBTITLE 1,describe
display(fixed_price_df.describe())

# COMMAND ----------

# DBTITLE 1,summary
display(fixed_price_df.summary())

# COMMAND ----------

# DBTITLE 0,--i18n-e9860f92-2fbe-4d23-b728-678a7bb4734e
# MAGIC %md 
# MAGIC
# MAGIC # Getting rid of extreme values
# MAGIC
# MAGIC Select only the `price` column and describe the result Dataframe in order to check the *min* and *max* values of the said column.

# COMMAND ----------

# DBTITLE 1,Describing Outliers
display(fixed_price_df.select("price").describe())

# COMMAND ----------

# DBTITLE 0,--i18n-bf195d9b-ea4d-4a3e-8b61-372be8eec327
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now only keep rows with a strictly positive *price*.

# COMMAND ----------

# DBTITLE 1,Deleting null prices
pos_prices_df = fixed_price_df.filter(col("price").isNotNull()).filter(col('price')> 0).filter(col('price') < 250)

# COMMAND ----------

# DBTITLE 0,--i18n-dc8600db-ebd1-4110-bfb1-ce555bc95245
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the *minimum_nights* column:

# COMMAND ----------

# DBTITLE 1,Same with nights
display(pos_prices_df.select("minimum_nights").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Group by minimum nights to check how many for each of them there are

# COMMAND ----------

# DBTITLE 1,groupBy & order
display(pos_prices_df
        .groupBy("minimum_nights").count()
        .orderBy(col("count").desc(), col("minimum_nights"))
       )

# COMMAND ----------

# DBTITLE 0,--i18n-5aa4dfa8-d9a1-42e2-9060-a5dcc3513a0d
# MAGIC %md 
# MAGIC
# MAGIC A minimum stay of one year seems to be a reasonable limit here. Let's filter out those records where the *minimum_nights* is greater than 365.

# COMMAND ----------

# DBTITLE 1,Deleting min nights outliers
min_nights_df = pos_prices_df.filter(col("minimum_nights") <= 365)

display(min_nights_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Check null values

# COMMAND ----------

# DBTITLE 1,null min nights
print(f'total: {min_nights_df.count()}')
print(f'not nulls: {min_nights_df.na.drop().count()}')

#We have a lot with missing values, so not a good idea to remove them

# COMMAND ----------

# DBTITLE 0,--i18n-25a35390-d716-43ad-8f51-7e7690e1c913
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Handling Null Values
# MAGIC
# MAGIC There are many different ways to handle null value, for instance you may just remove them, but sometines, null can actually be a key indicator of the thing you are trying to predict.
# MAGIC
# MAGIC Some ways to handle nulls:
# MAGIC * Drop any records that contain nulls
# MAGIC * Numeric: Replace them with mean/median/zero/etc.
# MAGIC * Categorical: Replace them with the mode or create a special category for null (i.e. "absent")
# MAGIC * Use techniques like ALS (Alternating Least Squares) which are designed to impute missing values
# MAGIC   
# MAGIC **If you do ANY imputation techniques for categorical/numerical features, you MUST include an additional field specifying that field was imputed.**
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark Imputer 
# MAGIC
# MAGIC  Spark imputer allows to impute values from null to another one, considering different strategies, it does not support imputation for categorical (string) features.
# MAGIC
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html?highlight=imputer#pyspark.ml.feature.Imputer" target="_blank">Documentation.</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/ml-features.html#imputer" target="_blank">Programming guide.</a>

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Transformers and Estimators
# MAGIC
# MAGIC Two key concepts introduced by the Spark ML API: **`transformers`** and **`estimators`**.
# MAGIC
# MAGIC **Transformer**: Transforms one DataFrame into another DataFrame. It accepts a DataFrame as input, and returns a new DataFrame with one or more columns appended to it. Transformers do not learn any parameters from your data and simply apply rule-based transformations. It has a **`.transform()`** method.
# MAGIC
# MAGIC **Estimator**: An algorithm which can be fit on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model. It has a **`.fit()`** method because it learns (or "fits") parameters from your DataFrame.

# COMMAND ----------

# DBTITLE 1,Imputer Sample
#Imputer sample
from pyspark.ml.feature import Imputer

df = spark.createDataFrame([
    (1.0, float("nan")),
    (2.0, float("nan")),
    (float("nan"), 3.0),
    (4.0, 4.0),
    (5.0, 5.0)
], ["a", "b"])


#Strategy mean (default), median, mode
imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
model = imputer.fit(df)

model.transform(df).show()

# COMMAND ----------

# DBTITLE 1,explain params
print(imputer.explainParams())

# COMMAND ----------

# DBTITLE 0,--i18n-83e56fca-ce6d-4e3c-8042-0c1c7b9eaa5a
# MAGIC %md 
# MAGIC
# MAGIC ### Impute: Cast to Double
# MAGIC
# MAGIC  SparkML's Imputer requires all fields to be of type double so we should cast all integer fields to double.

# COMMAND ----------

# DBTITLE 1,Casting to Double
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

integer_columns = [x.name for x in min_nights_df.schema.fields if x.dataType == IntegerType()]
doubles_df = min_nights_df

for c in integer_columns:
    doubles_df = doubles_df.withColumn(c, col(c).cast("double"))
    
columns = "\n - ".join(integer_columns)
print(f"Columns converted from Integer to Double:\n - {columns}")

# COMMAND ----------

# DBTITLE 0,--i18n-69b58107-82ad-4cec-8984-028a5df1b69e
# MAGIC %md 
# MAGIC
# MAGIC Add an extra column to marks the presence of null values before imputing (i.e. 1.0 = Yes, 0.0 = No). This way we we'll have a way to know if a column was "fixed" or nor

# COMMAND ----------

# DBTITLE 1,Adding na column
from pyspark.sql.functions import when

impute_cols = [
    'bedrooms',
    'bathrooms',
    'beds',
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value'
]

for c in impute_cols:
    doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))
# COMMAND ----------

# DBTITLE 1,describe
display(doubles_df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Applying estimator
# MAGIC
# MAGIC Now that all fields are double and we have an extra column for each of them indicating transformation we can apply the estimator
# MAGIC
# MAGIC
# MAGIC * Create an imputer with:
# MAGIC   * `"median"` strategy 
# MAGIC   * inputCols = impute_cols
# MAGIC   * outputCols same as inpute_cols so they are overriden
# MAGIC
# MAGIC * Fit the doubles_df datarame to the imputer using the `fit` method
# MAGIC   * This will return a model
# MAGIC *  Use the previous model to transform the `doubles_df` to another `imputed_df` using the `transform` method

# COMMAND ----------

# DBTITLE 1,applying estimator
from pyspark.ml.feature import Imputer

imputer = Imputer(strategy='median', inputCols=impute_cols, outputCols=impute_cols) 

imputer_model = imputer.fit(doubles_df) # compute strategy (median)
imputed_df = imputer_model.transform(doubles_df) # Apply calculation to the desired DF

# COMMAND ----------

# DBTITLE 0,--i18n-4df06e83-27e6-4cc6-b66d-883317b2a7eb
# MAGIC %md 
# MAGIC
# MAGIC ## Saving output
# MAGIC
# MAGIC
# MAGIC Our data is cleansed now. Let's save this DataFrame to Delta using override

# COMMAND ----------

# DBTITLE 1,Writing output
imputed_df.write.format('delta').mode('overwrite').save(output_path)

# COMMAND ----------

# DBTITLE 1,describe final
display(imputed_df.describe())
