# Databricks notebook source
# DBTITLE 1,Deleting Data
#Uncomment and run if needed

#dbutils.fs.rm("dbfs:/FileStore/input/wrongPath", True)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC # Data Cleansing & Anonymization
# MAGIC
# MAGIC We will be using Spark to load data from a dataset about people in NSS
# MAGIC
# MAGIC We are going to clean data from a file that contains data about different employees, with several columns, among them:
# MAGIC
# MAGIC * first name
# MAGIC * middle name
# MAGIC * last name
# MAGIC * ssn (Social security number)
# MAGIC * birthdate
# MAGIC * salary
# MAGIC * company
# MAGIC * department
# MAGIC * role
# MAGIC
# MAGIC
# MAGIC There are several issues:
# MAGIC
# MAGIC * Name format: first name may not be right, for instance we may have Rachel or CAROL
# MAGIC * SSN format: Some of the records come without hyphens. for instance  628-40-9043 and 324469083
# MAGIC * Due to the format issues, data come with duplicates, your mission is to fix it.
# MAGIC
# MAGIC Once the issues are fixed, records are guaranteed to match, so you can remove duplicates easily
# MAGIC
# MAGIC
# MAGIC * Also you should anonymize SSN 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load Data
from pyspark.sql.functions import split, col

file_path = f"dbfs:/FileStore/input/employees/"


raw_df = spark.read.csv(file_path,
                         header="true", 
                          inferSchema="true")

raw_df.count()

# COMMAND ----------

# DBTITLE 1,Cleaning SSN
from pyspark.sql.functions import concat, substring, lit

# A function that fix SSN to proper format XXX-XX-XXXX
def format_ssn(ssn):
    return concat(substring(ssn, 1, 3), lit('-'), substring(ssn, 4, 2), lit('-'), substring(ssn, 6, 4))

# Filter Columns not containing ssn
# Create a new column with the same name fixing the ssn column
wrong_ssn_df = raw_df.filter(~col("ssn").contains("-"))\ 
                .withColumn("ssn", format_ssn(col("ssn"))) 

# Filter again the original dataframe to keep only the rows containing a - (good rows)
# Union it with the dataset with the fixed SSN
fixed_ssn_df = raw_df.filter(col("ssn").contains("-")) \
.union(wrong_ssn_df) 

# COMMAND ----------

# DBTITLE 1,Cleaning Name
from pyspark.sql.functions import  initcap

#Use initcap function, that giving a String returns it with inly the first letter in uppercase
#No need to bifurcate dataset this time since good rows are not afected by the fix, they stay the same
fixed_name_df = fixed_ssn_df \
.withColumn("first_name", initcap(col("first_name")))


# COMMAND ----------

# DBTITLE 1,Removing duplicates
#Now that all the rows are clear, we can safely remove duplicates
fixed_df = fixed_name_df.drop_duplicates()

display(fixed_df)

# COMMAND ----------

# DBTITLE 1,Anonymizing
from pyspark.sql.functions import sha2

#Now we want to anonymize the salary and the ssn
#Use sha2 function with the column and the number of bits(256) inside withColumn
#Beware that it only applies on String, you need to cast salary to string first (or at the time)
anon_df = fixed_df \
  .withColumn("salary", sha2(col("salary").cast("string"), 256)) \
  .withColumn("ssn", sha2(col("ssn"), 256))


display(anon_df)


# COMMAND ----------

# MAGIC %md
# MAGIC
