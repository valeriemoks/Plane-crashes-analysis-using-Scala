// Databricks notebook source
// MAGIC %md
// MAGIC
// MAGIC ## Overview
// MAGIC
// MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
// MAGIC
// MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

// COMMAND ----------

// MAGIC %python
// MAGIC # File location and type
// MAGIC file_location = "/FileStore/tables/Plane_Crashes-2.csv"
// MAGIC file_type = "csv"
// MAGIC
// MAGIC # CSV options
// MAGIC infer_schema = "false"
// MAGIC first_row_is_header = "True"
// MAGIC delimiter = ","
// MAGIC
// MAGIC # The applied options are for CSV files. For other file types, these will be ignored.
// MAGIC df = spark.read.format(file_type) \
// MAGIC   .option("inferSchema", infer_schema) \
// MAGIC   .option("header", first_row_is_header) \
// MAGIC   .option("sep", delimiter) \
// MAGIC   .load(file_location)
// MAGIC
// MAGIC display(df)

// COMMAND ----------

// File location and type
val fileLocation = "/FileStore/tables/Plane_Crashes-2.csv"
val fileType = "csv"

// CSV options
val inferSchema = "false"
val firstRowIsHeader = "true"
val delimiter = ","

// Read CSV file into a DataFrame
val df = spark.read.format(fileType)
  .option("inferSchema", inferSchema)
  .option("header", firstRowIsHeader)
  .option("sep", delimiter)
  .load(fileLocation)

// Display the DataFrame
df.show()


// COMMAND ----------

// Assuming 'df' is your DataFrame
val dfRenamed = df.withColumnRenamed("Flight no.", "FlightNo")
dfRenamed.show()


// COMMAND ----------

// Drop specified columns
val columnsToDrop = Seq("Time", "Aircraft", "Operator", "Registration", "MSN", "Flight no.", "YOM", "Circumstances", "Schedule")
val dfWithoutSpecifiedColumns = df.drop(columnsToDrop: _*)

// Show the modified DataFrame
dfWithoutSpecifiedColumns.show()


// COMMAND ----------

// Drop rows with missing values
val dfWithoutMissingRows = dfWithoutSpecifiedColumns.na.drop()

// Show the DataFrame after dropping rows with missing values
dfWithoutMissingRows.show()


// COMMAND ----------


import org.apache.spark.sql.functions._

// Convert 'Date' column to Year
val dfWithYear = dfWithoutMissingRows.withColumn("Year", year(col("Date")))

// Group by Year and count the number of crashes
val crashesByYear = dfWithYear.groupBy("Year").count().orderBy("Year")

// Show the result
display(crashesByYear)



// COMMAND ----------


import org.apache.spark.sql.functions._

// Display count plot for Total Fatalities
val totalFatalitiesCounts = dfWithoutMissingRows.groupBy("Total fatalities").count().orderBy(desc("count"))
display(totalFatalitiesCounts)


// COMMAND ----------


import org.apache.spark.sql.functions._

// Display count plots for Flight phase
val flightPhaseCounts = dfWithoutMissingRows.groupBy("Flight phase").count().orderBy(desc("count"))
display(flightPhaseCounts)





// COMMAND ----------

// Display count plots for Flight type
val flightTypeCounts = dfWithoutMissingRows.groupBy("Flight type").count().orderBy(desc("count"))
display(flightTypeCounts)

// COMMAND ----------

// Display count plots for Crash site
val crashSiteCounts = dfWithoutMissingRows.groupBy("Crash site").count().orderBy(desc("count"))
display(crashSiteCounts)

// COMMAND ----------


// Display count plots for Crash cause
val crashCauseCounts = dfWithoutMissingRows.groupBy("Crash cause").count().orderBy(desc("count"))
display(crashCauseCounts)

// COMMAND ----------

import org.apache.spark.sql.functions.desc


// Extract the top 5 crash causes
val top5CrashCauses = crashCauseCounts.limit(5).select("Crash cause").collect().map(_.getString(0))

// Filter the DataFrame for the top 5 crash causes
val dfTop5CrashCauses = dfWithoutMissingRows.filter($"Crash cause".isin(top5CrashCauses: _*))

// Display count plots for the top 5 crash causes
val top5CrashCauseCounts = dfTop5CrashCauses.groupBy("Crash cause").count().orderBy(desc("count"))
display(top5CrashCauseCounts)


// COMMAND ----------

import org.apache.spark.sql.functions.{year, sum}

// Convert the "Date" column to a timestamp type
val dfWithTimestamp = dfWithoutMissingRows.withColumn("Date", to_timestamp($"Date", "yyyy-MM-dd"))

// Extract the year from the "Date" column
val dfWithYear = dfWithTimestamp.withColumn("Year", year($"Date"))

// Group by year and calculate the total fatalities
val totalFatalitiesByYear = dfWithYear.groupBy("Year").agg(sum("Total fatalities").alias("TotalFatalities"))

// Show the result
display(totalFatalitiesByYear)


// COMMAND ----------



// COMMAND ----------

dfWithoutMissingRows.show()

// COMMAND ----------


val columnTypes = dfWithoutMissingRows.dtypes

// Display the column names and their data types
columnTypes.foreach { case (columnName, columnType) =>
  println(s"Column: $columnName, Type: $columnType")
}


// COMMAND ----------

// Show unique values in the 'Survivors' column
dfWithoutMissingRows.select("Survivors").distinct().show()


// COMMAND ----------



// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
// Create a StringIndexer for the target column 'Survivors'
val labelIndexer = new StringIndexer()
  .setInputCol("Survivors")
  .setOutputCol("label")
  .setHandleInvalid("keep")

// List of categorical column names
val categoricalColumns = Array("Flight phase", "Flight type", "Crash site", "Country", "Region", "Crash cause", "Survivors")

// Create a StringIndexer for each categorical column with unique output column names
val indexers = categoricalColumns.map { colName =>
  new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(s"${colName}_index")
    .setHandleInvalid("keep")  // This option keeps unseen labels during transformation
}

// Create a pipeline with all the indexers
val pipeline = new Pipeline().setStages(indexers)

// Fit the pipeline on the DataFrame
val dfIndexed = pipeline.fit(dfWithoutMissingRows).transform(dfWithoutMissingRows)

// Show the resulting DataFrame with encoded columns
dfIndexed.show()



// COMMAND ----------


val maxCategories = dfIndexed.select(indexedColumns.map(col): _*).rdd
  .flatMap(row => row.toSeq.map(value => value.toString))
  .distinct()
  .count()
  .toInt

// COMMAND ----------

// Create a VectorAssembler to combine all features into a single vector column
val assembler = new VectorAssembler()
  .setInputCols(indexedColumns)
  .setOutputCol("features")

// Define the layers for the neural network
val layers = Array[Int](indexedColumns.length, 10, 5, 4) // Adjust the number of layers and neurons as needed

// Create a MultilayerPerceptronClassifier
val mlp = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)
  .setFeaturesCol("features")
  .setLabelCol("label")  // Using the indexed label column

// Create a pipeline with all the stages
val pipeline = new Pipeline().setStages(indexers :+ assembler :+ labelIndexer :+ mlp)

// Fit the pipeline on the DataFrame
val model = pipeline.fit(dfWithoutMissingRows)

// Make predictions on the DataFrame
val predictions = model.transform(dfWithoutMissingRows)

// Show the resulting DataFrame with predictions
predictions.select("Survivors", "prediction").show()

// COMMAND ----------


val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("Survivors_index")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

// Compute accuracy
val accuracy = evaluator.evaluate(predictions)
println(s"Accuracy: $accuracy")

