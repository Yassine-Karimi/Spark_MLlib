package ya.kr;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Main {
    public static void main(String[] args) {

        SparkSession ss=SparkSession.builder().appName("tp spark ml").master("local[*]").getOrCreate();

        Dataset<Row> dataset=ss.read().option("inferSchema",true).option("header",true).csv("advertising.csv");

        VectorAssembler vectorAssemble=new VectorAssembler().setInputCols(
                new String[]{"TV","Radio","Newspaper"}
        ).setOutputCol("Features");
        Dataset<Row> assembledDS=vectorAssemble.transform(dataset);
        Dataset<Row> splits[]=assembledDS.randomSplit(new double[]{0.8,0.2},123);
        Dataset<Row> train=splits[0];
        Dataset<Row> test=splits[1];
        LinearRegression lr=new LinearRegression().setLabelCol("Sales").setFeaturesCol("Features");
        LinearRegressionModel model=lr.fit(train);
        Dataset<Row> prediction=model.transform(test);
        System.out.println(model.getLoss());
        prediction.show();
        System.out.println("Intercept = "+model.intercept()+" Coeficients = "+model.coefficients());

    }
}