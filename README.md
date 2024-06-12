# Hadoop Cluster with Hive Integration and Spark Machine Learning

## Contributers

   Mahmoud Gamal Ahmed Mostafa

   Mohamed Gamal Tawfik Mohamed

   Mohamed Abd El-Rady Korashy

   Ahmed Said Ahmed Ali Abo Hekal
   
   Ibrahim abdelmonem hussien
   
   Omar Mostafa Abdelmonem

## Overview
This project sets up a Hadoop cluster using Docker Compose, integrates Hive for data warehousing, and leverages Spark for machine learning tasks. The cluster consists of a NameNode, multiple DataNodes, ResourceManager, NodeManager, HistoryServer, Postgres database for Hive metastore, Hive metastore and server, and Spark for machine learning.

## Prerequisites
* Docker
* Docker Compose

## Services
The cluster setup includes the following services:

* **NameNode**: Manages the Hadoop distributed file system (HDFS) namespace and controls access to files by clients.
* **DataNodes**: Store the actual data in HDFS.
* **ResourceManager**: Manages resources and schedules applications running on YARN.
* **NodeManager**: Manages the individual nodes in a YARN cluster.
* **HistoryServer**: Provides a web UI to view completed jobs.
* **Postgres**: Serves as the database for Hive metastore.
* **Hive Metastore**: Stores metadata for Hive tables and partitions.
* **Hive Server**: Executes Hive queries.
* **Spark**: Used for running machine learning tasks.

## Setup
1. **Create a Docker Compose file**: Use the provided docker-compose.yaml file to define the services.

2. **Prepare Hive configurations**:
  * Create a hive-site.xml file with the necessary Hive configurations.
  * Ensure the PostgreSQL JDBC driver is available in the specified path.

3. **Prepare Spark Machine Learning Task**:
  Create a ml_task.py script

4. **Run the Docker Compose setup**:
  * Navigate to the directory containing your docker-compose.yaml file.
  * Run the command:
    ```sh
    docker-compose up
    ```
5. **Verify the setup**:
   * Access the Hadoop NameNode UI at http://localhost:9870.
   * Access the Hadoop ResourceManager UI at http://localhost:8088.
   * Access the Hive Metastore at thrift://localhost:9083.
   * Access the HiveServer2 at jdbc:hive2://localhost:10000.

6. **Execute the Spark Machine Learning Task**:
   * The Spark container runs the ml_task.py script which initializes a Spark session, loads the Iris dataset, trains a Random Forest classifier, and prints the accuracy.

## Conclusion
This setup provides a scalable Hadoop cluster with Hive for big data and Spark for machine learning, allowing you to perform distributed data processing and analysis.
