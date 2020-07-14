from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.functions as f
from pyspark.ml.feature import Word2Vec
from pyspark.sql import Window

spark = SparkSession.builder.appName("zim").master("local[*]").config("spark.sql.broadcastTimeout", "36000").config("spark.driver.memory", "4G").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

# Dataframe de transacciones con todas las unidades
df = spark.read.parquet("/data/zim_data/transacciones.parquet")

# Filtrar las bolsas plasticas de retail
df = df.filter(~f.col("PROD").startswith("BOLSA TIPO"))

# Agrupar los productos comprados por cada usuario y organizarlos por fecha
w = Window.partitionBy('USER').orderBy('DATE')
df = df.withColumn('PRODS', f.collect_list('PROD').over(w))
df = df.groupBy('USER').agg(f.max('PRODS').alias('PRODS'))

# Entrenar el modelo
model = Word2Vec(vectorSize=5, inputCol="PRODS", outputCol="model").fit(df)
model.findSynonymsArray("PAÑUELOS", 10)