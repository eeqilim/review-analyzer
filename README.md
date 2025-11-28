# Customer Reviews Analyzer

This project implements an end-to-end AWS pipeline that processes, analyzes, and visualizes Amazon product reviews.

## Architecture Overview

| Component          | Purpose                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| **Amazon S3**      | Data lake for raw input, ETL output, and Lambda results                       |
| **AWS Glue**       | Distributed ETL + NLP preprocessing (Spark), publishes CloudWatch ETL metrics |
| **AWS Athena**     | SQL access over processed review data                                         |
| **AWS Lambda**     | Real-time sentiment + keyphrase analysis                                      |
| **AWS Comprehend** | NLP services for sentiment and keyphrases                                     |
| **Streamlit**      | Frontend to explore ASIN-level analytics                                      |

## Prerequisites

- AWS Account with S3, Glue, Lambda, Athena, and IAM permissions
- Python 3.14+

## Step 0: Download Raw Review Data

Download the raw Amazon review datasets from:

**https://nijianmo.github.io/amazon/index.html#files**

After downloading, upload the JSON files to `s3://<bucket-name>/input/`

## Step 1: Create S3 Buckets

Create an S3 bucket with the following structure:

```
s3://<bucket-name>/
├── input/          # Raw JSON review files
├── output/         # Cleaned Parquet files from Glue ETL
└── lambda/         # Lambda outputs (sentiment + summaries)
```

**Instructions:**

1. Go to AWS S3 Console
2. Create bucket: `<bucket-name>`
3. Create the three folders listed above
4. Upload your downloaded review JSON files to the `input/` folder

## Step 2: Create IAM Roles

### 2.1 Glue Role

1. Navigate to **IAM → Roles → Create role**
2. Select **Glue** as the trusted entity
3. Attach the following policies:
   - `AmazonS3FullAccess`
   - `AWSGlueServiceRole`
   - `AmazonAthenaFullAccess`
4. Name the role: `GlueAmazonReviewsRole`
5. Click **Create role**

### 2.2 Lambda Role

1. Navigate to **IAM → Roles → Create role**
2. Select **Lambda** as the trusted entity
3. Attach the following policies:
   - `AmazonS3FullAccess`
   - `AmazonAthenaFullAccess`
   - `AWSGlueConsoleFullAccess`
   - `CloudWatchLogsFullAccess`
4. Name the role: `LambdaAmazonReviewsRole`
5. Click **Create role**

### 2.3 Add Inline Policy for AWS Comprehend

1. Go to the `LambdaAmazonReviewsRole`
2. Click the **Permissions** tab
3. Scroll down to **Inline policies** → Click **Add inline policy**
4. Switch to the **JSON** tab and paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Statement1",
      "Effect": "Allow",
      "Action": [
        "comprehend:DetectSentiment",
        "comprehend:BatchDetectSentiment",
        "comprehend:DetectKeyPhrases",
        "comprehend:BatchDetectKeyPhrases"
      ],
      "Resource": "*"
    }
  ]
}
```

5. Name the policy: `LambdaComprehendInvoke`
6. Click **Create policy**

## Step 3: Deploy Glue ETL Job

1. Navigate to **AWS Glue → ETL Jobs → Create job**
2. Configure the job:
   - **Name**: `preprocess_reviews`
   - **IAM Role**: `GlueAmazonReviewsRole`
   - **Type**: Spark (Python)
   - **Script**: Copy and paste the contents of `glue.py`
3. Save and run the job
4. Verify output in `s3://<bucket-name>/output/` (should contain Parquet files)

## Step 4: Create Glue Crawler

1. Navigate to **AWS Glue → Crawlers → Create crawler**
2. Configure the crawler:
   - **Name**: `crawler_reviews`
   - **Data source**: S3
   - **S3 path**: `s3://<bucket-name>/output/`
   - **IAM Role**: `GlueAmazonReviewsRole`
   - **Frequency**: Run on demand
   - **Output database**: `reviews_db` (create if it doesn't exist)
   - **Table prefix**: Leave blank
3. Click **Create** and then **Run crawler**

### Verify the Table

Query the table in Athena:

```sql
-- View sample data
SELECT *
FROM "AwsDataCatalog"."reviews_db"."output" LIMIT 10;

-- Show all tables
SHOW
TABLES IN `reviews_db`;

-- Describe table schema
DESCRIBE reviews_db.output;
```

## Step 5: Deploy Lambda Function

1. Navigate to **AWS Lambda → Create function**
2. Configure the function:
   - **Name**: `lambda_sentiment`
   - **Runtime**: Python 3.14
   - **Execution role**: Use existing role → `LambdaAmazonReviewsRole`
   - **Timeout**: 1 min (under Configuration → General configuration)
3. Copy and paste the code from `lambda.py`
4. Click **Deploy**

### Test the Lambda Function

Use the following example to test event:

```json
{
  "asin": "B00007IT8S"
}
```

Click **Test** to verify the function executes successfully.

## Step 6: Frontend Setup

### 6.1 Create IAM User for Streamlit

1. Navigate to **IAM → Users → Create user**
2. Username: `streamlit-user`
3. Attach policy: `AmazonS3FullAccess`
4. Save the **Access Key ID** and **Secret Access Key**

### 6.2 Configure AWS CLI

In your terminal, run:

```bash
aws configure
```

Enter the following when prompted:

```
AWS Access Key ID [None]: <Your Access Key ID>
AWS Secret Access Key [None]: <Your Secret Access Key>
Default region name [None]: us-west-2
Default output format [None]: json
```

### 6.3 Install Streamlit and Run App

```bash
pip install streamlit
streamlit run app.py
```

The app should open in your browser at `http://localhost:8501`.

## Step 7: CloudWatch Monitoring

**View Metrics:**

CloudWatch → Metrics → All metrics → Glue → Metrics with no dimensions
