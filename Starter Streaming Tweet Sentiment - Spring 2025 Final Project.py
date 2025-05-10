# Databricks notebook source
# MAGIC %md
# MAGIC ## DSCC202-402 Data Science at Scale Final Project
# MAGIC ### Tracking Tweet sentiment at scale using a pretrained transformer (classifier)
# MAGIC <p>Consider the following illustration of the end to end system that you will be building.  Each student should do their own work.  The project will demonstrate your understanding of Spark Streaming, the medalion data architecture using Delta Lake, Spark Inference at Scale using an MLflow packaged model as well as Exploritory Data Analysis and System Tracking and Monitoring.</p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/pipeline.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You will be pulling an updated copy of the course GitHub repositiory: <a href="https://github.com/lpalum/dscc202-402-spring2025">The Repo</a>.  
# MAGIC
# MAGIC Once you have updated your fork of the repository you should see the following template project that is resident in the final_project directory.
# MAGIC </p>
# MAGIC
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/notebooks.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You can then pull your project into the Databrick Workspace using the <a href="https://github.com/apps/databricks">Databricks App on Github</a> or by cloning the repo to your laptop and then uploading the final_project directory and its contents to your workspace using file imports.  Your choice.
# MAGIC
# MAGIC <p>
# MAGIC Work your way through this notebook which will give you the steps required to submit a complete and compliant project.  The following illustration and associated data dictionary specifies the transformations and data that you are to generate for each step in the medallion pipeline.
# MAGIC </p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/dataframes.drawio.png">
# MAGIC
# MAGIC #### Bronze Data - raw ingest
# MAGIC - date - string in the source json
# MAGIC - user - string in the source json
# MAGIC - text - tweet string in the source json
# MAGIC - sentiment - the given sentiment of the text as determined by an unknown model that is provided in the source json
# MAGIC - source_file - the path of the source json file the this row of data was read from
# MAGIC - processing_time - a timestamp of when you read this row from the source json
# MAGIC
# MAGIC #### Silver Data - Bronze Preprocessing
# MAGIC - timestamp - convert date string in the bronze data to a timestamp
# MAGIC - mention - every @username mentioned in the text string in the bronze data gets a row in this silver data table.
# MAGIC - cleaned_text - the bronze text data with the mentions (@username) removed.
# MAGIC - sentiment - the given sentiment that was associated with the text in the bronze table.
# MAGIC
# MAGIC #### Gold Data - Silver Table Inference
# MAGIC - timestamp - the timestamp from the silver data table rows
# MAGIC - mention - the mention from the silver data table rows
# MAGIC - cleaned_text - the cleaned_text from the silver data table rows
# MAGIC - sentiment - the given sentiment from the silver data table rows
# MAGIC - predicted_score - score out of 100 from the Hugging Face Sentiment Transformer
# MAGIC - predicted_sentiment - string representation of the sentiment
# MAGIC - sentiment_id - 0 for negative and 1 for postive associated with the given sentiment
# MAGIC - predicted_sentiment_id - 0 for negative and 1 for positive assocaited with the Hugging Face Sentiment Transformer
# MAGIC
# MAGIC #### Application Data - Gold Table Aggregation
# MAGIC - min_timestamp - the oldest timestamp on a given mention (@username)
# MAGIC - max_timestamp - the newest timestamp on a given mention (@username)
# MAGIC - mention - the user (@username) that this row pertains to.
# MAGIC - negative - total negative tweets directed at this mention (@username)
# MAGIC - neutral - total neutral tweets directed at this mention (@username)
# MAGIC - positive - total positive tweets directed at this mention (@username)
# MAGIC
# MAGIC When you are designing your approach, one of the main decisions that you will need to make is how you are going to orchestrate the streaming data processing in your pipeline.  There are several valid approaches to triggering your steams and how you will gate the execution of your pipeline.  Think through how you want to proceed and ask questions if you need guidance. The following references may be helpful:
# MAGIC - [Spark Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
# MAGIC - [Databricks Autoloader - Cloudfiles](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
# MAGIC - [In class examples - Spark Structured Streaming Performance](https://dbc-f85bdc5b-07db.cloud.databricks.com/editor/notebooks/2638424645880316?o=1093580174577663)
# MAGIC
# MAGIC ### Be sure your project runs end to end when *Run all* is executued on this notebook! (7 points)
# MAGIC
# MAGIC ### This project is worth 25% of your final grade.
# MAGIC - DSCC-202 Students have 55 possible points on this project (see points above and the instructions below)
# MAGIC - DSCC-402 Students have 60 possible points on this project (one extra section to complete)

# COMMAND ----------

# DBTITLE 1,Pull in the Includes & Utiltites
# MAGIC %run ./includes/includes

# COMMAND ----------

# DBTITLE 1,Notebook Control Widgets (maybe helpful)
"""
Adding a widget to the notebook to control the clearing of a previous run.
or stopping the active streams using routines defined in the utilities notebook
"""
dbutils.widgets.removeAll()

dbutils.widgets.dropdown("clear_previous_run", "No", ["No","Yes"])
if (getArgument("clear_previous_run") == "Yes"):
    clear_previous_run()
    print("Cleared all previous data.")

dbutils.widgets.dropdown("stop_streams", "No", ["No","Yes"])
if (getArgument("stop_streams") == "Yes"):
    stop_all_streams()
    print("Stopped all active streams.")

dbutils.widgets.dropdown("optimize_tables", "No", ["No","Yes"])
if (getArgument("optimize_tables") == "Yes"):
    # Suck up those small files that we have been appending.
    # Optimize the tables
    optimize_table(BRONZE_DELTA)
    optimize_table(SILVER_DELTA)
    optimize_table(GOLD_DELTA)
    print("Optimized all of the Delta Tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.0 Import your libraries here (2 points)
# MAGIC - Are your shuffle partitions consistent with your cluster and your workload?
# MAGIC - Do you have the necessary libraries to perform the required operations in the pipeline/application?

# COMMAND ----------

# ENTER YOUR CODE HERE
%pip install ydata-profiling==4.5.1

# COMMAND ----------

!pip install 'pydantic<2'

# COMMAND ----------

# MAGIC %pip install transformers

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

!pip install 'anyio<4.0.0'

# COMMAND ----------

# ENTER YOUR CODE HERE
spark.sparkContext.defaultParallelism

# COMMAND ----------

#selecting the partition
spark.conf.set("spark.sql.shuffle.partitions", "16")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.0 Define and execute utility functions (3 points)
# MAGIC - Read the source file directory listing
# MAGIC - Count the source files (how many are there?)
# MAGIC - print the contents of one of the files

# COMMAND ----------

# List all files in the Tweet source directory
files = dbutils.fs.ls(TWEET_SOURCE_PATH)

# Count the number of files
print(f"Number of source files: {len(files)}")

# Print the content of one file (first 500 characters)
if files:
    sample_file = files[0].path
    sample_content = dbutils.fs.head(sample_file, 500)
    print("Sample content from first file:\n", sample_content)
else:
    print("No files found in TWEET_SOURCE_PATH.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.0 Transform the Raw Data to Bronze Data using a stream  (8 points)
# MAGIC - define the schema for the raw data
# MAGIC - setup a read stream using cloudfiles and the source data format
# MAGIC - setup a write stream using delta lake to append to the bronze delta table
# MAGIC - enforce schema
# MAGIC - allow a new schema to be merged into the bronze delta table
# MAGIC - Use the defined BRONZE_CHECKPOINT and BRONZE_DELTA paths defined in the includes
# MAGIC - name your raw to bronze stream as bronze_stream
# MAGIC - transform the raw data to the bronze data using the data definition at the top of the notebook

# COMMAND ----------

# ENTER YOUR CODE HERE
from pyspark.sql.types import StructType, StringType
from pyspark.sql.functions import input_file_name, current_timestamp

# 1. Define schema for raw JSON data (as per spec)
bronze_schema = StructType() \
    .add("date", StringType()) \
    .add("user", StringType()) \
    .add("text", StringType()) \
    .add("sentiment", StringType())

# 2. Set up a read stream using Auto Loader
raw_stream_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .schema(bronze_schema)  # Enforce schema
    .load(TWEET_SOURCE_PATH)
    .withColumn("source_file", input_file_name())
    .withColumn("processing_time", current_timestamp())
)

# 3. Write stream to Bronze Delta table
bronze_stream = (
    raw_stream_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", BRONZE_CHECKPOINT)
    .option("mergeSchema", "true")  # Allow new columns if schema evolves
    .queryName("bronze_stream")     # Name the stream
    .start(BRONZE_DELTA) )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.0 Transform the Bronze Data to Silver Data using a stream (5 points)
# MAGIC - setup a read stream on your bronze delta table
# MAGIC - setup a write stream to append to the silver delta table
# MAGIC - Use the defined SILVER_CHECKPOINT and SILVER_DELTA paths in the includes
# MAGIC - name your bronze to silver stream as silver_stream
# MAGIC - transform the bronze data to the silver data using the data definition at the top of the notebook

# COMMAND ----------

#Transforming bronze data to silver
from pyspark.sql.functions import col, regexp_replace, explode, split, trim, to_timestamp
from pyspark.sql.types import StringType
 
#Read from Bronze Delta table
bronze_df = (
    spark.readStream
    .format("delta")
    .load(BRONZE_DELTA)
)
 
#Transform into Silver Data
silver_df = (
    bronze_df
    .filter(col("text").isNotNull() & col("sentiment").isNotNull())
    
    #Convert the date to timestamp
    .withColumn("timestamp", to_timestamp(col("date")))
    
    #Extract mentions as individual rows
    .withColumn("mention", explode(split(col("text"), " ")))  # Split by space
    .filter(col("mention").startswith("@"))  # Keep only tokens that are mentions
    
    #Remove all mentions from the original text to form cleaned_text
    .withColumn("cleaned_text", regexp_replace(col("text"), "@\\w+", ""))
    
    #Clean extra whitespace
    .withColumn("cleaned_text", trim(col("cleaned_text")))
    
    #Select and rename columns to match Silver schema
    .select(
        "timestamp",
        "mention",
        "cleaned_text",
        col("sentiment").alias("Sentiment")
    )
)
 
#Write to Silver Delta table
silver_stream = (
    silver_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", SILVER_CHECKPOINT)
    .queryName("silver_stream")
    .start(SILVER_DELTA)
)
 
print("🚀 Silver stream started.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.0 Transform the Silver Data to Gold Data using a stream (7 points)
# MAGIC - setup a read stream on your silver delta table
# MAGIC - setup a write stream to append to the gold delta table
# MAGIC - Use the defined GOLD_CHECKPOINT and GOLD_DELTA paths defines in the includes
# MAGIC - name your silver to gold stream as gold_stream
# MAGIC - transform the silver data to the gold data using the data definition at the top of the notebook
# MAGIC - Load the pretrained transformer sentiment classifier from the MODEL_NAME at the production level from the MLflow registry
# MAGIC - Use a spark UDF to parallelize the inference across your silver data

# COMMAND ----------

# Transforming the Silver data to Gold data
# Loading model from MLflow
import mlflow.pyfunc
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
 
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StringType, FloatType, StructType, StructField
 
sentiment_schema = StructType([
    StructField("score", FloatType()),
    StructField("label", StringType())
])
 
@udf(sentiment_schema)
def sentiment_predict_udf(text):
    import pandas as pd
    try:
        if not text or not isinstance(text, str):
            return (0.0, "UNKNOWN")
        df = pd.DataFrame([text], columns=["text"])
        result = model.predict(df)
        return (float(result.iloc[0]["score"]) * 100, result.iloc[0]["label"].upper())
    except Exception:
        return (0.0, "ERROR")
 
# starting the silver stream 
silver_stream = (
    spark.readStream
         .format("delta")
         .load(SILVER_DELTA)
)
 
# applying the UDF and building the Gold schema 
gold_ready = (
    silver_stream
        .withColumn("prediction", sentiment_predict_udf(col("cleaned_text")))
        .withColumn("predicted_score", col("prediction.score"))
        .withColumn("predicted_sentiment", col("prediction.label"))
        .withColumn("sentiment_id", when(col("sentiment").isin("negative", "NEG"), 0).otherwise(1))
        .withColumn("predicted_sentiment_id", when(col("predicted_sentiment").isin("NEGATIVE", "NEG"), 0).otherwise(1))
        .select(
            col("timestamp"),
            col("mention"),
            col("cleaned_text"),
            col("sentiment"),
            col("predicted_score"),
            col("predicted_sentiment"),
            col("sentiment_id"),
            col("predicted_sentiment_id")
        )
)
 
# writng to gold delta table and starting the stream
gold_output_stream = (
    gold_ready.writeStream
              .queryName("gold_stream")
              .format("delta")
              .option("checkpointLocation", GOLD_CHECKPOINT)
              .outputMode("append")
              .start(GOLD_DELTA)
)
 
print("✅ Gold stream started with correct schema.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.0 Monitor your Streams (5 points)
# MAGIC - Setup a loop that runs at least every 10 seconds
# MAGIC - Print a timestamp of the monitoring query along with the list of streams, rows processed on each, and the processing time on each
# MAGIC - Run the loop until all of the data is processed (0 rows read on each active stream)
# MAGIC - Plot a line graph that shows the data processed by each stream over time
# MAGIC - Plot a line graph that shows the average processing time on each stream over time

# COMMAND ----------

# ENTER YOUR CODE HERE
# List all active streams
for stream in spark.streams.active:
    print(f"🔄 Stream Name: {stream.name}")
    print(f"  Is Active: {stream.isActive}")
    print(f"  Status: {stream.status['message']}")
    print("-" * 40)

# COMMAND ----------

import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Initialize tracking
monitoring_data = []
idle_checks = 0
max_idle_checks = 3  # allow 3 idle loops before stopping
check_interval = 10  # seconds between checks

print("📡 Monitoring active streams...")
while True:
    active_queries = spark.streams.active
    timestamp = datetime.now().strftime("%H:%M:%S")

    all_idle = True

    for query in active_queries:
        progress = query.lastProgress
        if progress:
            num_input_rows = progress["numInputRows"]
            proc_time = float(progress["durationMs"].get("addBatch", 0)) / 1000
            name = query.name

            print(f"[{timestamp}] Stream: {name} | Rows: {num_input_rows} | Time: {proc_time:.2f}s")

            monitoring_data.append({
                "timestamp": timestamp,
                "stream": name,
                "rows": num_input_rows,
                "time": proc_time
            })

            if num_input_rows > 0:
                all_idle = False

    if all_idle:
        idle_checks += 1
        print(f"⚠️ No data processed. Idle check {idle_checks}/{max_idle_checks}")
    else:
        idle_checks = 0

    if idle_checks >= max_idle_checks:
        print("✅ Streams idle for 3 consecutive checks. Stopping monitoring.")
        break

    time.sleep(check_interval)

# COMMAND ----------

#Gold record count
gold_count = spark.read.format("delta").load(GOLD_DELTA).count()
print(f"Gold table records : {gold_count}  (expect ~100 000)")

# COMMAND ----------

#Bronze record count
bronze_count = spark.read.format("delta").load(BRONZE_DELTA).count()
print(f"BRONZE table records : {bronze_count}  (expect ~100 000)")

# COMMAND ----------

#Silver record count
silver_count = spark.read.format("delta").load(SILVER_DELTA).count()
print(f"Silver table records : {silver_count}  (expect ~100 000)")

# COMMAND ----------

# Convert to Pandas DataFrame
df_monitor = pd.DataFrame(monitoring_data)

# Plot: Rows processed over time
plt.figure(figsize=(25, 15))
for stream in df_monitor["stream"].unique():
    subset = df_monitor[df_monitor["stream"] == stream]
    plt.plot(subset["timestamp"], subset["rows"], marker="o", label=stream)

plt.title("Rows Processed Over Time")
plt.xlabel("Time")
plt.ylabel("Rows Processed")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------

# Plot: Processing time over time
plt.figure(figsize=(25, 15))
for stream in df_monitor["stream"].unique():
    subset = df_monitor[df_monitor["stream"] == stream]
    plt.plot(subset["timestamp"], subset["time"], marker="s", label=stream)

plt.title("Average Processing Time Over Time")
plt.xlabel("Time")
plt.ylabel("Processing Time (s)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.0 Bronze Data Exploratory Data Analysis (5 points)
# MAGIC - How many tweets are captured in your Bronze Table?
# MAGIC - Are there any columns that contain Nan or Null values?  If so how many and what will you do in your silver transforms to address this?
# MAGIC - Count the number of tweets by each unique user handle and sort the data by descending count.
# MAGIC - How many tweets have at least one mention (@) how many tweet have no mentions (@)
# MAGIC - Plot a bar chart that shows the top 20 tweeters (users)
# MAGIC

# COMMAND ----------

# ENTER YOUR CODE HERE
# Load Bronze data (batch read)
bronze_df = spark.read.format("delta").load(BRONZE_DELTA)

# COMMAND ----------

tweet_count = bronze_df.count()
print(f"Total number of tweets in Bronze Table: {tweet_count}")

# COMMAND ----------

from pyspark.sql.functions import col, isnan

null_summary = bronze_df.select([
    col(c).isNull().cast("int").alias(c) for c in bronze_df.columns
]).groupBy().sum().toPandas().T

null_summary.columns = ["Null Count"]
print(null_summary)

# COMMAND ----------

from pyspark.sql.functions import desc

user_tweet_counts = bronze_df.groupBy("user").count().orderBy(desc("count"))
user_tweet_counts.show(10)

# COMMAND ----------

from pyspark.sql.functions import col

with_mentions = bronze_df.filter(col("text").contains("@")).count()
without_mentions = bronze_df.filter(~col("text").contains("@")).count()

print(f"Tweets with @ mentions: {with_mentions}")
print(f"Tweets without @ mentions: {without_mentions}")

# COMMAND ----------

# Convert top 20 tweeters to Pandas for plotting
top20_users = user_tweet_counts.limit(20).toPandas()

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.barh(top20_users["user"], top20_users["count"], color='skyblue')
plt.xlabel("Tweet Count")
plt.title("Top 20 Tweeters")
plt.gca().invert_yaxis()  # Highest on top
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.0 Capture the accuracy metrics from the gold table in MLflow  (4 points)
# MAGIC Store the following in an MLflow experiment run:
# MAGIC - Store the precision, recall, and F1-score as MLflow metrics
# MAGIC - Store an image of the confusion matrix as an MLflow artifact
# MAGIC - Store the model name and the MLflow version that was used as an MLflow parameters
# MAGIC - Store the version of the Delta Table (input-silver) as an MLflow parameter

# COMMAND ----------

# === Imports ===
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

# === Step 1: Load Gold Delta Table ===
gold_df = spark.read.format("delta").load(GOLD_DELTA).toPandas()

# === Step 2: Drop missing predictions or labels ===
gold_df = gold_df.dropna(subset=["sentiment_id", "predicted_sentiment_id"])

# === Step 3: Convert to integers ===
y_true = gold_df["sentiment_id"].astype(int)
y_pred = gold_df["predicted_sentiment_id"].astype(int)

# === Step 4: Compute metrics ===
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

# === Step 5: Plot confusion matrix ===
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save the plot
conf_matrix_path = "/tmp/confusion_matrix.png"
plt.savefig(conf_matrix_path)
plt.close()

# === Step 6: Get model version from MLflow Registry ===
client = MlflowClient()
model_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0].version

# === Step 7: Get Delta Table version of Silver input ===
silver_version = DeltaTable.forPath(spark, SILVER_DELTA).history().select("version").first()["version"]

# === Step 8: Start MLflow run and log everything ===
with mlflow.start_run(run_name="Gold Evaluation Metrics"):
    # Log metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log parameters
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("model_version", model_version)
    mlflow.log_param("silver_table_version", silver_version)

    # Log confusion matrix plot
    mlflow.log_artifact(conf_matrix_path)

print("✅ Evaluation complete and all metrics/artifacts logged to MLflow.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.0 Application Data Processing and Visualization (6 points)
# MAGIC - How many mentions are there in the gold data total?
# MAGIC - Count the number of neutral, positive and negative tweets for each mention in new columns
# MAGIC - Capture the total for each mention in a new column
# MAGIC - Sort the mention count totals in descending order
# MAGIC - Plot a bar chart of the top 20 mentions with positive sentiment (the people who are in favor)
# MAGIC - Plot a bar chart of the top 20 mentions with negative sentiment (the people who are the vilians)
# MAGIC
# MAGIC *note: A mention is a specific twitter user that has been "mentioned" in a tweet with an @user reference.

# COMMAND ----------

# Load Gold Delta table as batch
gold_df = spark.read.format("delta").load(GOLD_DELTA)

# COMMAND ----------

from pyspark.sql.functions import upper, trim

# COMMAND ----------

gold_df.select("Predicted_sentiment").distinct().show(truncate=False)


# COMMAND ----------

mention_summary = (
    gold_df.groupBy("mention")
    .agg(
        count("*").alias("total"),
        count(when(upper(trim(col("Predicted_sentiment"))) == "POS", True)).alias("positive"),
        count(when(upper(trim(col("Predicted_sentiment"))) == "NEU", True)).alias("neutral"),
        count(when(upper(trim(col("Predicted_sentiment"))) == "NEG", True)).alias("negative")
    )
    .orderBy(col("total").desc())
)


# COMMAND ----------

total_mentions = mention_summary.count()
print(f"Total unique mentions in the gold data: {total_mentions}")

# COMMAND ----------

mention_pdf = mention_summary.toPandas()

# COMMAND ----------

mention_pdf[['mention', 'positive', 'negative']].sort_values(by='positive', ascending=False).head(10)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Top 20 positive
top_pos = mention_pdf.sort_values(by="positive", ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_pos, x="mention", y="positive", palette="Greens_r")
plt.xticks(rotation=45, ha="right")
plt.title("Top 20 Mentions with Positive Sentiment")
plt.tight_layout()
plt.show()

# Top 20 negative
top_neg = mention_pdf.sort_values(by="negative", ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_neg, x="mention", y="negative", palette="Reds_r")
plt.xticks(rotation=45, ha="right")
plt.title("Top 20 Mentions with Negative Sentiment")
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.0 Clean up and completion of your pipeline (3 points)
# MAGIC - using the utilities what streams are running? If any.
# MAGIC - Stop all active streams
# MAGIC - print out the elapsed time of your notebook. Note: In the includes there is a variable START_TIME that captures the starting time of the notebook.

# COMMAND ----------

# ENTER YOUR CODE HERE
# List all currently active streams
print("🔍 Active Streams:")
for query in spark.streams.active:
    print(f"- {query.name} | Is Active: {query.isActive} | Status: {query.status['message']}")

# COMMAND ----------

print("🛑 Stopping all active streams...")
for query in spark.streams.active:
    print(f"Stopping stream: {query.name}")
    query.stop()

# COMMAND ----------

from datetime import datetime

# Convert START_TIME from float to datetime object
start_dt = datetime.fromtimestamp(START_TIME)
end_dt = datetime.now()

elapsed = end_dt - start_dt
print(f"⏱️ Notebook Elapsed Time: {elapsed}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.0 How Optimized is your Spark Application (Grad Students Only) (5 points)
# MAGIC Graduate students (registered for the DSCC-402 section of the course) are required to do this section.  This is a written analysis using the Spark UI (link to screen shots) that support your analysis of your pipelines execution and what is driving its performance.
# MAGIC Recall that Spark Optimization has 5 significant dimensions of considertation:
# MAGIC - Spill: write to executor disk due to lack of memory
# MAGIC - Skew: imbalance in partition size
# MAGIC - Shuffle: network io moving data between executors (wide transforms)
# MAGIC - Storage: inefficiency due to disk storage format (small files, location)
# MAGIC - Serialization: distribution of code segments across the cluster
# MAGIC
# MAGIC Comment on each of the dimentions of performance and how your impelementation is or is not being affected.  Use specific information in the Spark UI to support your description.  
# MAGIC
# MAGIC Note: you can take sreenshots of the Spark UI from your project runs in databricks and then link to those pictures by storing them as a publicly accessible file on your cloud drive (google, one drive, etc.)
# MAGIC
# MAGIC References:
# MAGIC - [Spark UI Reference Reference](https://spark.apache.org/docs/latest/web-ui.html#web-ui)
# MAGIC - [Spark UI Simulator](https://www.databricks.training/spark-ui-simulator/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ENTER YOUR MARKDOWN HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.0 Spark Application Performance Analysis 
# MAGIC
# MAGIC ### 1. **Spill (Memory Spill to Disk)**  
# MAGIC > Spark spills data when memory is insufficient for shuffle operations.
# MAGIC
# MAGIC - Observed that Stage 1034, which likely corresponds to the Bronze stream or a lightweight processing stage, completed without any memory or disk spill. 
# MAGIC - Task deserialization was minimal (2ms), and no shuffle or serialization delays were present. 
# MAGIC - This indicates that the Spark executor had sufficient memory to process the task entirely  in-memory
# MAGIC
# MAGIC *See screenshot below*  
# MAGIC https://drive.google.com/file/d/1CtvHSfpE2RjQjop0oVfwdNSZS151dNfy/view?usp=drive_link
# MAGIC https://drive.google.com/file/d/1QY3o5F1sRNv-DP7mV8T5_doCg-pnsiFc/view?usp=drive_link
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. **Skew (Data Partition Skew)**  
# MAGIC > Skew occurs when some partitions are much larger than others.
# MAGIC
# MAGIC - Analyzed Stage 1026, which included a shuffle and grouped data by mention.
# MAGIC - The Event Timeline shows four parallel tasks executing with nearly identical duration.
# MAGIC - Task metrics confirm uniform distribution of records and compute load.
# MAGIC - No skew observed.
# MAGIC
# MAGIC *See screenshot below*  
# MAGIC https://drive.google.com/file/d/1pmcVaBYpmUANNTr8atlXRonKgRhQ5Biu/view?usp=drive_link
# MAGIC https://drive.google.com/file/d/1EJc0F2OE-o7eB0IPPOh_16BOIGOaLpyd/view?usp=drive_link
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. **Shuffle (Wide Transformations)**  
# MAGIC > Shuffle occurs during wide transformations like `groupBy`, `join`.
# MAGIC
# MAGIC - Stage 1026 triggered a shuffle due to `groupBy(mention)` during aggregation.
# MAGIC - Shuffle write sizes per task were moderate (up to ~250 KiB), with no signs of slowdown.
# MAGIC - All tasks completed in roughly the same time (0.2s), indicating good parallelism and no imbalance.
# MAGIC
# MAGIC *See screenshot below* <br>
# MAGIC https://drive.google.com/file/d/1pmcVaBYpmUANNTr8atlXRonKgRhQ5Biu/view?usp=drive_link <br>
# MAGIC https://drive.google.com/file/d/1EJc0F2OE-o7eB0IPPOh_16BOIGOaLpyd/view?usp=drive_link
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4. **Storage (Delta Format & Small Files)**  
# MAGIC > Efficient storage improves performance, especially with Delta Lake.
# MAGIC
# MAGIC - Used Delta format across Bronze, Silver, and Gold layers.
# MAGIC - Streaming writes succeeded with moderate file sizes and no job failures.
# MAGIC - Not observed performance issues due to small files.
# MAGIC - Job 451 shows 3 stages completed with successful Delta writes and consistent output sizes.
# MAGIC
# MAGIC *See screenshot below* <br>
# MAGIC https://drive.google.com/file/d/1PBKtfk4ydLwEqQfbaoIzglv1cBJY6q4E/view?usp=drive_link
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5. **Serialization (UDF + Model Overhead)**  
# MAGIC > Serialization overhead comes from UDFs or large Python objects.
# MAGIC
# MAGIC - Used a Spark UDF (`@udf(...)`) that wraps an MLflow sentiment model.
# MAGIC - The model was loaded once per executor, avoiding per-row overhead.
# MAGIC - Stage 921 shows 4 tasks completing with consistent durations and 0 ms `Result Serialization Time`.
# MAGIC - No task failures or lag observed, confirming efficient serialization.
# MAGIC
# MAGIC *See screenshot below* <br>
# MAGIC https://drive.google.com/file/d/18tjT-hKuuyqzg6MwWa3SMqhUFZ1U6E-B/view?usp=drive_link <br>
# MAGIC https://drive.google.com/file/d/153iWmx6CFyhWM7YVRb92sol6OdACar8p/view?usp=drive_link
# MAGIC
# MAGIC
# MAGIC