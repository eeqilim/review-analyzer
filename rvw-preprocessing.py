import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, lower, regexp_replace, trim  
'''
ETL Script: Pre-normalization Pipeline for Amazon Review Data (AWS Glue + Spark)

This script performs an end-to-end preprocessing workflow for JSONL review data stored in S3.
'''
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


# 1. Define S3 paths
input_path = "s3://pj-jwc-eql/input/CDs_and_Vinyl_5.json"   
output_path = "s3://pj-jwc-eql/output/"                     


# 2. Read JSONL data from S3

df_raw = spark.read.json(input_path)


# 3. Data cleaning: remove null or missing values

df_clean = df_raw.filter(
    (col("reviewText").isNotNull()) & 
    (col("overall").isNotNull())
)

# ----------------------------
# 4. Filter only verified purchases
# ----------------------------
df_verified = df_clean.filter(col("verified") == True)

# 5. Normalize text fields
df_normalized = (
    df_verified
    .withColumn("reviewText", lower(trim(regexp_replace(col("reviewText"), "[^a-zA-Z0-9\\s]", ""))))
    .withColumn("summary", lower(trim(regexp_replace(col("summary"), "[^a-zA-Z0-9\\s]", ""))))
)

# 6. Select relevant columns

df_final = df_normalized.select(
    "reviewerID",
    "asin",
    col("overall").alias("rating"),
    "verified",
    "reviewText",
    "summary",
    "unixReviewTime"
)


# 7. Write cleaned data back to S3 (Parquet)

df_final.write.mode("overwrite").parquet(output_path)

job.commit()