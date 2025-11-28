from pyspark.context import SparkContext
from pyspark.sql.functions import col, lower, regexp_replace, trim, length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from awsglue.context import GlueContext
from awsglue.job import Job
import boto3
import logging
import time

cloudwatch = boto3.client("cloudwatch")
logger = logging.getLogger("glue")
logger.setLevel(logging.INFO)


def cloudwatch_metric(metric, value, unit="Count"):
    cloudwatch.put_metric_data(
        Namespace="Glue",
        MetricData=[{
            "MetricName": metric,
            "Value": float(value),
            "Unit": unit
        }]
    )
    logger.info(f"{metric}: {value}")


start_time = time.time()
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init("nlp_feature_job", {})

input_path = "s3://pj-cs6240/input/CDs_and_Vinyl_5.json"
output_path = "s3://pj-cs6240/output/"

# Load raw reviews
df = spark.read.json(input_path)

initial_count = df.count()
cloudwatch_metric("Initial_Rows_Count", initial_count)

# Filter nulls and verified reviews
df_clean = df.filter(
    (col("reviewText").isNotNull()) &
    (col("summary").isNotNull()) &
    (col("verified") == True)
)

filtered_count = df_clean.count()
cloudwatch_metric("Filtered_Rows_Count", filtered_count)

# Normalize text and overwrite original columns
df_clean = df_clean.withColumn("reviewText",
                               lower(trim(regexp_replace(col("reviewText"), "[^a-zA-Z0-9\\s]", "")))).withColumn(
    "summary", lower(trim(regexp_replace(col("summary"), "[^a-zA-Z0-9\\s]", ""))))
df_clean = df_clean.withColumn("review_length", length(col("reviewText"))).withColumn("summary_length",
                                                                                      length(col("summary")))

# Tokenization
tokenizer = Tokenizer(inputCol="reviewText", outputCol="tokens")
df_tokens = tokenizer.transform(df_clean)

# Stopword removal
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
df_tokens = remover.transform(df_tokens)

# Term Frequency – Inverse Document Frequency (TF-IDF) Feature Extraction
cv = CountVectorizer(inputCol="filtered_tokens", outputCol="raw_features", vocabSize=1000)
cv_model = cv.fit(df_tokens)

vocab_size = len(cv_model.vocabulary)
cloudwatch_metric("TFIDF_Vocab_Size", vocab_size)

df_cv = cv_model.transform(df_tokens)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(df_cv)
df_final = idf_model.transform(df_cv)

# Save cleaned review text and features
df_final.select("reviewerID", "asin", "overall", "verified", "review_length", "summary_length", "reviewText",
                "tfidf_features").write.mode("overwrite").parquet(output_path)

# Runtime Logging
runtime = round(time.time() - start_time, 2)
cloudwatch_metric("Total_Runtime", runtime, unit="Seconds")

job.commit()
