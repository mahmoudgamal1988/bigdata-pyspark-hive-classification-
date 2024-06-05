# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# # Initialize Spark session with Hive support
# spark = SparkSession.builder \
#     .appName('IrisClassifier') \
#     .config('spark.hadoop.fs.defaultFS', 'hdfs://namenode:8020') \
#     .config('spark.executor.memory', '1g') \
#     .config('spark.executor.cores', '1') \
#     .config('spark.yarn.am.memory', '1g') \
#     .config('hive.metastore.uris', 'thrift://hive-metastore:9083') \
#     .enableHiveSupport() \
#     .getOrCreate()

# # Load Iris dataset from Scikit-learn
# iris = load_iris()
# data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# data['target'] = iris.target

# # Split the data into training and testing sets
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# # Convert to Spark DataFrame
# train_df = spark.createDataFrame(train_data)
# test_df = spark.createDataFrame(test_data)

# # Save training and testing data to Hive
# train_df.write.mode('overwrite').saveAsTable('iris_train')
# test_df.write.mode('overwrite').saveAsTable('iris_test')

# # Read data from Hive
# train_df = spark.table('iris_train')
# test_df = spark.table('iris_test')

# # Prepare features
# assembler = VectorAssembler(inputCols=iris.feature_names, outputCol='features')
# train_df = assembler.transform(train_df)
# test_df = assembler.transform(test_df)

# # Initialize RandomForestClassifier
# rf = RandomForestClassifier(labelCol='target', featuresCol='features', numTrees=10)

# # Train the model
# model = rf.fit(train_df)

# # Make predictions
# predictions = model.transform(test_df)

# # Evaluate the model
# evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
# accuracy = evaluator.evaluate(predictions)
# print('Test Accuracy: {}'.format(accuracy))

# # Stop Spark session
# spark.stop()


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import Row
import matplotlib.pyplot as plt

# Initialize Spark session with Hive support
spark = SparkSession.builder \
    .appName('IrisClassifier') \
    .config('spark.hadoop.fs.defaultFS', 'hdfs://namenode:8020') \
    .config('spark.executor.memory', '1g') \
    .config('spark.executor.cores', '1') \
    .config('spark.yarn.am.memory', '1g') \
    .config('hive.metastore.uris', 'thrift://hive-metastore:9083') \
    .enableHiveSupport() \
    .getOrCreate()

# Load Iris dataset from Scikit-learn
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert to Spark DataFrame
train_df = spark.createDataFrame(train_data)
test_df = spark.createDataFrame(test_data)

# Save training and testing data to Hive
train_df.write.mode('overwrite').saveAsTable('iris_train')
test_df.write.mode('overwrite').saveAsTable('iris_test')

# Read data from Hive
train_df = spark.table('iris_train')
test_df = spark.table('iris_test')

# Prepare features
assembler = VectorAssembler(inputCols=iris.feature_names, outputCol='features')
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# Initialize RandomForestClassifier
rf = RandomForestClassifier(labelCol='target', featuresCol='features', numTrees=10)

# Train the model
model = rf.fit(train_df)

# Make predictions
predictions = model.transform(test_df)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print('Test Accuracy: {}'.format(accuracy))

# Extract feature importances
importances = model.featureImportances.toArray()
features = iris.feature_names

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='blue')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()

# Example MapReduce task: Count the number of instances for each class
def map_func(row):
    return Row(target=row['target'])

mapped_rdd = train_df.rdd.map(map_func)
reduced_rdd = mapped_rdd.map(lambda row: (row['target'], 1)).reduceByKey(lambda a, b: a + b)

# Collect and print the results
class_counts = reduced_rdd.collect()
print('Class counts in training data:', class_counts)

# Save the class counts to Hive
class_counts_df = spark.createDataFrame(class_counts, ['class', 'count'])
class_counts_df.write.mode('overwrite').saveAsTable('class_counts')

# Read and show the class counts from Hive
class_counts_df = spark.table('class_counts')
class_counts_df.show()

# Stop Spark session
spark.stop()
