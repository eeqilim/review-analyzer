import sys
from pyspark.context import SparkContext
from pyspark.sql.functions import col, lower, regexp_replace, trim, length, lit, when, round # Added lit, when, round for feature engineering
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
    (col("overall").isNotNull()) & # Added check for 'overall' which is needed for later analysis
    (col("verified") == True)
)

filtered_count = df_clean.count()
cloudwatch_metric("Filtered_Rows_Count", filtered_count)

# ----------------------------
# 5. Normalize text and add basic length features
# ----------------------------

# Apply text normalization (original complex regex, watch for performance!)
df_clean = df_clean.withColumn("reviewText",
                             lower(trim(regexp_replace(col("reviewText"), "[^a-zA-Z0-9\\s]", "")))).withColumn(
    "summary", lower(trim(regexp_replace(col("summary"), "[^a-zA-Z0-9\\s]", ""))))

# Calculate review and summary lengths (basic features)
df_clean = df_clean.withColumn("review_length", length(col("reviewText"))).withColumn("summary_length",
                                                                                    length(col("summary")))

# -----------------------------------------------------------
# NEW STEP: INTEGRATE ANALYTICAL AND INFLUENCE FEATURES
# -----------------------------------------------------------

df_features = (
    df_clean
    # Feature 1: Influence Score (derived from the 'vote' field)
    # Cleans the string and casts to integer, handles non-numeric votes/nulls.
    .withColumn("votesCleaned", regexp_replace(col("vote"), "[^0-9]", ""))
    .withColumn("voteCount", col("votesCleaned").cast("int"))
    
    # Feature 2: Flag Highly Influential Reviews (e.g., those with 5 or more votes)
    .withColumn(
        "isHighInfluence", 
        when(col("voteCount") >= 5, lit(True)).otherwise(lit(False))
    )
    # Feature 3: Flag Short/Low-Effort Reviews (using the previously calculated 'review_length')
    .withColumn(
        "isShortReview",
        when(col("review_length") < 30, lit(True)).otherwise(lit(False))
    )
    .drop("votesCleaned") # Drop the intermediate cleaning column
)

# -----------------------------------------------------------
# CONTINUE NLP FEATURE EXTRACTION (Use df_features as input)
# -----------------------------------------------------------

# Tokenization
tokenizer = Tokenizer(inputCol="reviewText", outputCol="tokens")
df_tokens = tokenizer.transform(df_features) # Changed input to df_features

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

# Save all cleaned data and features
df_final.select("reviewerID", "asin", "overall", "verified", 
                "review_length", "summary_length", 
                "voteCount", "isHighInfluence", "isShortReview", # Included new features
                "reviewText", "tfidf_features")
        .write.mode("overwrite").parquet(output_path)

# Runtime Logging
runtime = round(time.time() - start_time, 2)
cloudwatch_metric("Total_Runtime", runtime, unit="Seconds")

job.commit()