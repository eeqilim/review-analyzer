import boto3
import ast
import time
import logging
import json
from collections import Counter

athena = boto3.client("athena")
s3 = boto3.client("s3")
comprehend = boto3.client("comprehend")

OUTPUT_BUCKET = "pj-cs6240"
OUTPUT_PREFIX = "lambda"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def run_athena_query(query, database="reviews_db"):
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": f"s3://{OUTPUT_BUCKET}/athena-results/"}
    )
    query_id = response["QueryExecutionId"]

    # Wait for query to finish
    while True:
        status = athena.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]["Status"]["State"]
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(2)

    if status != "SUCCEEDED":
        raise Exception(f"Athena query failed: {status}")

    results = athena.get_query_results(QueryExecutionId=query_id)["ResultSet"]["Rows"]
    headers = [h["VarCharValue"] for h in results[0]["Data"]]
    data = []

    for row in results[1:]:
        row_data = {}
        for i, col in enumerate(row["Data"]):
            row_data[headers[i]] = col.get("VarCharValue", None)
        data.append(row_data)
    return data


def batch_comprehend_call(func, texts, batch_size=25):
    results = []

    for i in range(0, len(texts), batch_size):
        batch = [t[:5000] for t in texts[i:i + batch_size]]  # truncate to 5k chars
        response = func(TextList=batch, LanguageCode="en")
        results.extend(response["ResultList"])
    return results


def classify_sentiment(texts):
    return batch_comprehend_call(comprehend.batch_detect_sentiment, texts)


def extract_key_phrases(texts):
    return batch_comprehend_call(comprehend.batch_detect_key_phrases, texts)


def parse_tfidf(tfidf_raw):
    if not tfidf_raw:
        return None
    try:
        tfidf_dict = json.loads(tfidf_raw)
        return tfidf_dict
    except:
        try:
            tfidf_json_like = tfidf_raw.replace('=', ':')

            for key in ['type', 'size', 'indices', 'values']:
                tfidf_json_like = tfidf_json_like.replace(f'{key}:', f'"{key}":')

            tfidf_dict = ast.literal_eval(tfidf_json_like)

            return tfidf_dict

        except Exception as e:
            print(f"Error parsing TF-IDF: {e}")
            return None


def generate_insights_summary(review_records):
    if not review_records:
        return {}

    positive_reviews = [r for r in review_records if r['sentiment'] == 'positive']
    negative_reviews = [r for r in review_records if r['sentiment'] == 'negative']
    neutral_reviews = [r for r in review_records if r['sentiment'] == 'neutral']
    mixed_reviews = [r for r in review_records if r['sentiment'] == 'mixed']

    # Key phrases
    positive_phrases = [p["Text"].lower() for r in positive_reviews for p in r.get("key_phrases", [])]
    negative_phrases = [p["Text"].lower() for r in negative_reviews for p in r.get("key_phrases", [])]

    # Top topics
    positive_topics = Counter(positive_phrases).most_common(10)
    negative_topics = Counter(negative_phrases).most_common(10)

    # Sample reviews
    positive_samples = [r['review'][:200] + "..." for r in positive_reviews[:3]]
    negative_samples = [r['review'][:200] + "..." for r in negative_reviews[:3]]

    # Build summary
    summary = {
        "total_reviews": len(review_records),
        "sentiment_counts": {
            "positive": len(positive_reviews),
            "negative": len(negative_reviews),
            "neutral": len(neutral_reviews),
            "mixed": len(mixed_reviews)
        },
        "common_positive_topics": [{"topic": t[0], "mentions": t[1]} for t in positive_topics],
        "common_negative_topics": [{"topic": t[0], "mentions": t[1]} for t in negative_topics],
        "positive_review_samples": positive_samples,
        "negative_review_samples": negative_samples
    }

    return summary


def summarize_reviews(summary, asin):
    return {
        "ASIN": asin,
        "Summary": summary,
        "Timestamp": int(time.time())
    }


def save_to_s3(data, key):
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json"
    )


def process_asin(asin, max_reviews=None):
    query = f"""
        SELECT reviewText, overall, tfidf_features
        FROM output
        WHERE asin = '{asin}'
        {f'LIMIT {max_reviews}' if max_reviews else ''}
    """
    rows = run_athena_query(query)
    if not rows:
        logger.info(f"No reviews found for ASIN {asin}")
        return

    texts = [r.get("reviewText") for r in rows if r.get("reviewText")]
    ratings = [float(r.get("overall", 0)) for r in rows]
    tfidf_vectors = [r.get("tfidf_features") for r in rows]

    if not texts:
        logger.info(f"No valid review text for ASIN {asin}")
        return

    sentiment_results = classify_sentiment(texts)
    key_phrase_results = extract_key_phrases(texts)
    parsed_tfidf = [parse_tfidf(vec) for vec in tfidf_vectors]

    review_records = []
    for i, text in enumerate(texts):
        review_records.append({
            "review": text,
            "rating": ratings[i],
            "sentiment": sentiment_results[i]["Sentiment"].lower(),
            "key_phrases": key_phrase_results[i].get("KeyPhrases", []),
            "tfidf_features": parsed_tfidf[i]
        })

    summary = generate_insights_summary(review_records)
    final_summary = summarize_reviews(summary, asin)

    save_to_s3(review_records, f"{OUTPUT_PREFIX}/{asin}/reviews.json")
    save_to_s3(final_summary, f"{OUTPUT_PREFIX}/{asin}/summary.json")


def lambda_handler(event, context):
    asin_rows = run_athena_query("SELECT DISTINCT asin FROM output")
    asins = [r["asin"] for r in asin_rows]

    asins_to_process = asins[:10]  # limit to first 10 ASINs

    for asin in asins_to_process:
        process_asin(asin, max_reviews=200)  # limit to 200 reviews per ASIN

    return {
        "status": "success",
        "asin_count": len(asins_to_process)
    }
