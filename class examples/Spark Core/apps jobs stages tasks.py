# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/1.png" width="640">

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Consider this illustration of a spark cluster with a driver (all clusters have one) where your applications runs 
# MAGIC and 4 worker/executors and 8 cores.  Note: that the numbers of workers and cores are configurable based on the size of cluster you are working with.
# MAGIC It could be 100s/1000s down to 1 worker coresident with the driver like the Databricks Community edition.</p><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/2.png" width="640">
# MAGIC <p>When we are running in the context of a notebook, each cell of the notebook can comprise one or more jobs, stages and tasks that are 
# MAGIC   scheduled to run on the workers of the cluster [Physcial Job Planning].  </p>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/3.png" width="800">
# MAGIC   
# MAGIC <p>Consider the following definitions:</p>
# MAGIC - **Job**
# MAGIC  - The work required to compute an RDD
# MAGIC - **Stage**
# MAGIC  - A series of work within a job to produce one or  more pipelined RDD’s
# MAGIC - **Tasks**
# MAGIC  - A unit of work within a stage, corresponding to one  RDD partition
# MAGIC - **Shuffle**
# MAGIC  - The transfer of data between stages
# MAGIC
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/4.png" width="640">
# MAGIC
# MAGIC
# MAGIC In the next cell we can try a simple example job that includes a set of *transforms*:
# MAGIC - **parallelize** (create an RDD)
# MAGIC - **union** (combine RDD)
# MAGIC - **groupBy** (group by first letter of the word)
# MAGIC - **filter**     (remove the first letter of )
# MAGIC
# MAGIC followed by a single *action*:
# MAGIC - **collect** 
# MAGIC
# MAGIC This illustration outlines the steps that spark follows to build and schedule this kind of job<br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/5.png" width="800">
# MAGIC
# MAGIC Shuffling is worth a specific mention here.  There are two kinds of shuffling between stages of the DAG.  Wide shuffling (grouping) is more expensive that the narrow shuffling (filtering):
# MAGIC <table broder=0><tr><td>
# MAGIC   <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/6.png" width="400">
# MAGIC   </td>
# MAGIC   <td><img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/7.png" width="400">
# MAGIC   </td></tr></table>

# COMMAND ----------

# DBTITLE 1,Example of a simple spark job
# creating RDD x with fruits and names
a = spark.sparkContext.parallelize(["Apple", "Orange", "Pineapple", "Kiwi", "Banana", "Grape",  "Date", "Pomeganate", "Mango"], 3)
b = spark.sparkContext.parallelize(["Allan", "Oliver", "Paula", "Karen", "James", "Cory", "Christine", "Jackeline", "Juan"], 3)

# take the union of the two RDD
x = a.union(b)

# Applying groupBy operation on x
y = x.groupBy(lambda name: name[0])

# filter out names that start with J and P
z = y.filter(lambda name: name[0]!='J' and name[0]!='P')

for t in z.collect():
    print((t[0],[i for i in t[1]]))
 

#TODO After you run this cell click on the "view" link next to the Job to see the spark UI that outlines how the job was planned and scheduled on the cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-stage pipelines
# MAGIC Let's look at a little more complicated example of a multi-stage Job that reads in the works of Sherlock Holmes and counts the occurance of each word in the text.
# MAGIC <p>We are going to look at two different ways to perform this job.<p>
# MAGIC   - Resilient Distributed Data (RDD) based job (low level spark api)<br>
# MAGIC   <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/10.png" width="800">
# MAGIC
# MAGIC
# MAGIC Dataframe (DF) based job (higher level api that we will use primarily in ths class)

# COMMAND ----------

# DBTITLE 1,Read the sherlock.txt file into the temp dir and move to dbfs
# MAGIC %sh
# MAGIC # note curl is a command line utility to read data over https
# MAGIC curl https://data-science-at-scale.s3.amazonaws.com/spark-on-docker/shared-workspace/data/sherlock.txt > /tmp/sherlock.txt
# MAGIC ls /tmp/sherlock.txt

# COMMAND ----------

dbutils.fs.mv("file:/tmp/sherlock.txt", "dbfs:/FileStore/sherlock.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC lines = spark.read.text("dbfs:/FileStore/sherlock.txt").rdd.map(lambda r: r[0])
# MAGIC ```
# MAGIC
# MAGIC **1. `spark.read.text("dbfs:/FileStore/sherlock.txt")`**
# MAGIC
# MAGIC    *   `spark`: This refers to your `SparkSession` object, which is the entry point to Spark functionality. It's where you access various Spark APIs.
# MAGIC    *   `read`: This is a method within the `SparkSession` that allows you to read data from various sources.
# MAGIC    *   `text(...)`: This specific method is used to read a text file into a Spark DataFrame. It assumes that each line in the text file represents a single record.
# MAGIC    *   `"dbfs:/FileStore/sherlock.txt"`: This is the path to the text file you're reading. `dbfs` indicates that the file is located in Databricks File System (DBFS). `FileStore/sherlock.txt` would be the location in DBFS where the file is stored.
# MAGIC    * **Output:**  This part of the code reads the text file and creates a **DataFrame** with a single column named "value". Each row in this DataFrame would be a line from the text file.
# MAGIC
# MAGIC **2. `.rdd`**
# MAGIC
# MAGIC    *   This is a method called on the DataFrame, `.rdd()`, to convert a DataFrame to a Resilient Distributed Dataset (RDD). It transforms the data from a structured table into an RDD that contains Row objects, in this case, where each Row object corresponds to a row in the DataFrame.
# MAGIC    * **Output:** An RDD containing Row objects from the previous step.
# MAGIC
# MAGIC **3. `.map(lambda r: r[0])`**
# MAGIC
# MAGIC    *   `map(...)`: This is a transformation on RDDs. The `map` transformation applies a function to every element in the RDD, and returns a new RDD.
# MAGIC    *   `lambda r: r[0]`: This is an anonymous function (lambda function).
# MAGIC         * `r`: Represents each element of the RDD. Since the input is an RDD created from a DataFrame, each element of the RDD, or in this case each `r`, is an object of type `Row` which in this case has only one field named `value` by default.
# MAGIC         *   `r[0]`: Accesses the **first** element (index 0) within each Row object. In this case, the `value` field of the Row object is in index 0 by default.  This extracts the actual string (line from file) itself from the row object.
# MAGIC    * **Output:** A new RDD where each element is a line of text extracted from the original file, and the original Row objects are discarded.
# MAGIC
# MAGIC **In Summary**
# MAGIC
# MAGIC This line of code:
# MAGIC
# MAGIC 1. Reads a text file from DBFS.
# MAGIC 2. Creates a DataFrame with a single column, containing each line of the text as a row.
# MAGIC 3. Converts this DataFrame into an RDD of Row objects.
# MAGIC 4. Applies a `map` transformation to the RDD to extract the text line from each Row object (the `value` field at index 0).
# MAGIC 5. Assigns the resulting RDD, which now contains strings, to the `lines` variable.
# MAGIC
# MAGIC **Essentially, the code reads a text file and stores each line of the file in a Spark RDD as a string.**
# MAGIC
# MAGIC **Why do this?**
# MAGIC
# MAGIC *   While Spark generally encourages using DataFrames, there may be situations where you need to work directly with RDDs for lower-level control or access to features only available on RDDs.
# MAGIC *   This specific code might be the first step of a larger operation that needs to perform some special manipulations on strings, and might be using RDD API for such operations.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Read the file into a RDD
# Read in the works of sherlock holmes
lines = spark.read.text("dbfs:/FileStore/sherlock.txt").rdd.map(lambda r: r[0])
lines.take(30)

# COMMAND ----------

# MAGIC %md
# MAGIC Okay, let's break down this PySpark code snippet step by step. This code performs a word count operation and then retrieves the top 10 most frequent words:
# MAGIC
# MAGIC ```python
# MAGIC lines.flatMap(lambda x: x.split(' ')) \
# MAGIC               .filter(lambda x: len(x) >= 1) \
# MAGIC               .map(lambda x: (x, 1)) \
# MAGIC               .reduceByKey(lambda a,b: a+b) \
# MAGIC               .sortBy(lambda x: x[1], False) \
# MAGIC               .take(10)
# MAGIC ```
# MAGIC
# MAGIC We are assuming that `lines` refers to a Spark RDD where each element is a line of text (as in our previous discussion).
# MAGIC
# MAGIC **1. `lines.flatMap(lambda x: x.split(' '))`**
# MAGIC
# MAGIC    *   `flatMap(...)`: This is a transformation on RDDs. Like `map`, `flatMap` applies a function to each element of the RDD, but `flatMap` flattens the resulting lists (or other collections) into a single RDD.
# MAGIC    *   `lambda x: x.split(' ')`: This is an anonymous function (lambda function).
# MAGIC        *   `x`: represents each element of the RDD, which is a line of text.
# MAGIC        *   `x.split(' ')`: Splits each line of text into a list of words, using a space (" ") as the delimiter.
# MAGIC    * **Output:** A new RDD, where each element is an individual word from all the lines in the original `lines` RDD.
# MAGIC
# MAGIC **2. `.filter(lambda x: len(x) >= 1)`**
# MAGIC
# MAGIC    *   `filter(...)`: This is a transformation on RDDs that filters out elements based on a condition.
# MAGIC    *   `lambda x: len(x) >= 1`: This lambda function checks if the length of a given word (`x`) is greater than or equal to 1. This means we keep only words that are not empty strings after the splitting of lines.
# MAGIC    * **Output:** A new RDD that contains only non-empty words.
# MAGIC
# MAGIC **3. `.map(lambda x: (x, 1))`**
# MAGIC
# MAGIC    *   `map(...)`: This is a transformation on RDDs that applies a function to each element.
# MAGIC    *   `lambda x: (x, 1)`: This lambda function transforms each word `x` into a key-value pair `(x, 1)`. This means for every word, we will associate the number 1 with it. This forms the basis for counting the number of times this word appears.
# MAGIC    * **Output:** A new RDD of key-value pairs, where the keys are the words and values are all initialized to `1`. This is in preparation for the counting of the words.
# MAGIC
# MAGIC **4. `.reduceByKey(lambda a,b: a+b)`**
# MAGIC
# MAGIC    *   `reduceByKey(...)`: This transformation is specific to RDDs of key-value pairs. It groups values with the same key and applies the given reduction function to them.
# MAGIC    *   `lambda a,b: a+b`: This lambda function takes two values `a` and `b` (both integers, representing counts) and returns their sum (`a + b`).  This will add all the ones for the same word, giving you the word count for every word in the RDD.
# MAGIC    * **Output:** A new RDD of key-value pairs, where each key is a word, and the value is the number of times it occurred in the text.
# MAGIC
# MAGIC **5. `.sortBy(lambda x: x[1], False)`**
# MAGIC
# MAGIC    *   `sortBy(...)`: This transformation sorts the RDD based on a specified key.
# MAGIC    *   `lambda x: x[1]`: This lambda function accesses the second element (`x[1]`) of the key-value pair `x`, which represents the word count. We want to sort based on word counts.
# MAGIC    *   `False`: This boolean parameter indicates that the sort should be in descending order (from the most frequent to the least frequent).
# MAGIC    * **Output:** A new RDD, sorted by the number of times each word occurred, in descending order.
# MAGIC
# MAGIC **6. `.take(10)`**
# MAGIC
# MAGIC    *   `take(...)`: This is an action (not a transformation) that returns a specified number of elements from the RDD to the driver.
# MAGIC    *   `10`: Returns the first 10 elements. In this case, those will be the 10 most frequent words and their counts.
# MAGIC    * **Output:** A list of the top 10 most frequent words (and their counts) in the form of a list of tuples, `[(word1, count1), (word2, count2), ...]`.
# MAGIC
# MAGIC **In Summary:**
# MAGIC
# MAGIC This code snippet performs the following:
# MAGIC
# MAGIC 1.  **Splits** each line of text into individual words.
# MAGIC 2.  **Removes** any empty strings from the list of words.
# MAGIC 3.  **Creates key-value pairs** (word, 1) for each word.
# MAGIC 4.  **Counts** occurrences of each word.
# MAGIC 5.  **Sorts** the words by their count in descending order.
# MAGIC 6.  **Retrieves** the top 10 most frequent words and their counts and returns that list to the driver.
# MAGIC
# MAGIC This is a very common pattern for performing a word count on text data using Spark. The resulting output, if printed, would look like a list of tuples, for instance: `[('the', 123), ('and', 98), ('to', 75), ..., ('word10', 12)]`.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Perform the RDD based job
"""
Steps
=====
- split each line in the file up into individual words (separated by spaces)
- filter out words that are less than 1 character
- create a tuple for each word that looks like (word, 1)
- for each unique word add up all of the occurances (reduce)
- sort the results in descending order
- take a look at the highest 10 occuring words
"""
lines.flatMap(lambda x: x.split(' ')) \
              .filter(lambda x: len(x) >= 1) \
              .map(lambda x: (x, 1)) \
              .reduceByKey(lambda a,b: a+b) \
              .sortBy(lambda x: x[1], False) \
              .take(10)

# COMMAND ----------


