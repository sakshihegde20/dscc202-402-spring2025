# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC ## M&M Example
# MAGIC Used in `class`

# COMMAND ----------

from pyspark.sql.functions import *

mnm_file = "/databricks-datasets/learning-spark-v2/mnm_dataset.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read from the CSV and infer the schema

# COMMAND ----------

mnm_df = spark.read \
          .format("csv") \
          .option("header", "true") \
          .option("inferSchema", "true") \
          .load(mnm_file)

# COMMAND ----------

display(mnm_df)

# COMMAND ----------

mnm_df = (spark
          .read
          .format("csv")
          .option("header", "true")
          .option("inferSchema", "true")
          .load(mnm_file))

display(mnm_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Repartition the dataframe to have 12 partitions to be consistent with our illustration in class

# COMMAND ----------

# Determine the number of partitions you want
desired_partitions = 12

# Using repartition() (full shuffle)
mnm_df_repartitioned = mnm_df.repartition(desired_partitions)

# Verify the number of partitions
num_partitions_repartitioned = mnm_df_repartitioned.rdd.getNumPartitions()
print(f"Number of partitions after repartition: {num_partitions_repartitioned}")

# Verify the number of partitions of original dataframe
num_partitions_original = mnm_df.rdd.getNumPartitions()
print(f"Number of partitions of original df: {num_partitions_original}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter out the brown M&Ms from the data!

# COMMAND ----------

mnm_df_repartitioned_no_brown = mnm_df_repartitioned.filter(mnm_df.Color != "Brown")
display(mnm_df_repartitioned_no_brown)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Repartition the dataframe to have 4 partitions to be consistent with our illustration in class

# COMMAND ----------

# Option 2: Using coalesce() (reduces partitions efficiently, if possible)
mnm_df_coalesced_no_brown = mnm_df_repartitioned_no_brown.coalesce(4) # Use only if you are reducing the number of partitions.

# Verify the number of partitions
num_partitions_coalesced = mnm_df_coalesced_no_brown.rdd.getNumPartitions()
print(f"Number of partitions after repartition: {num_partitions_repartitioned}")

# Verify the number of partitions of original dataframe
num_partitions_original = mnm_df_coalesced_no_brown.rdd.getNumPartitions()
print(f"Number of partitions of original df: {num_partitions_original}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count the number of non-brown M&Ms and compare to the original data.  How many brown M&Ms were there?

# COMMAND ----------

display(mnm_df_coalesced_no_brown)

# COMMAND ----------

total_no_brown = mnm_df_coalesced_no_brown.count()
print(f"Number of non-brown M&Ms: {total_no_brown}")

# COMMAND ----------

total = mnm_df.count()
print(f"Number of M&Ms: {total}")

# COMMAND ----------

print(f"The total number of brown M&Ms is: {total - total_no_brown}")
