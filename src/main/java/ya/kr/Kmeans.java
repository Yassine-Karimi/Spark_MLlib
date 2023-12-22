package ya.kr;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Kmeans {
        public static void main(String[] args) {
            SparkSession ss = SparkSession.builder()
                    .appName("Kmeans App")
                    .master("local[*]")
                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    .getOrCreate();
            Dataset<Row> dataset = ss.read().option("inferSchema", true).option("header", true).csv("Mall_Customers.csv");
            VectorAssembler vectorAssemble = new VectorAssembler().setInputCols(
                    new String[]{"Age", "Annual Income (k$)", "Spending Score (1-100)"}
            ).setOutputCol("features");
            Dataset<Row> assembledDS = vectorAssemble.transform(dataset);
            MinMaxScaler scaler = new MinMaxScaler().setInputCol("features").setOutputCol("normalizedFeatures");
            Dataset<Row> normalizedDS = scaler.fit(assembledDS).transform(assembledDS);
            KMeans kMeans = new KMeans().setK(3).setFeaturesCol("normalizedFeatures").setPredictionCol("cluster");
            KMeansModel model = kMeans.fit(normalizedDS);
            Dataset<Row> prediction = model.transform(normalizedDS);
            prediction.show(100);


            ClusteringEvaluator evaluator = new ClusteringEvaluator()
                    .setPredictionCol("cluster")
                    .setFeaturesCol("normalizedFeatures");


            double score = evaluator.evaluate(prediction);
            System.out.println("Score = " + score);

        }
    }
