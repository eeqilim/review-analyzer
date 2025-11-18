import boto3
import json
import time
from collections import Counter
import ast

athena = boto3.client("athena")
s3 = boto3.client("s3")
comprehend = boto3.client("comprehend", region_name="us-west-2")

DATABASE = "reviews_db"
TABLE = "output"
OUTPUT_BUCKET = "pj-cs6240"
PREFIX = "lambda"
ATHENA_OUTPUT = f"s3://{OUTPUT_BUCKET}/{PREFIX}/athena_results/"


def run_athena_query(query):
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT}
    )
    qid = resp["QueryExecutionId"]

    while True:
        result = athena.get_query_execution(QueryExecutionId=qid)
        state = result["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        reason = result["QueryExecution"]["Status"].get("StateChangeReason", "Unknown error")
        raise Exception(f"Athena query failed: {state}. Reason: {reason}")

    results = athena.get_query_results(QueryExecutionId=qid)
    return results["ResultSet"]["Rows"][1:]


def classify_sentiment(texts):
    sentiments = []
    batch_size = 25

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            truncated_batch = [t[:5000] for t in batch]
            response = comprehend.batch_detect_sentiment(
                TextList=truncated_batch,
                LanguageCode='en'
            )

            for result in response['ResultList']:
                sentiments.append(result['Sentiment'].lower())

        except Exception:
            for text in batch:
                try:
                    res = comprehend.detect_sentiment(Text=text[:5000], LanguageCode='en')
                    sentiments.append(res['Sentiment'].lower())
                except:
                    sentiments.append('neutral')

    return sentiments


def extract_key_phrases(texts):
    phrases_all = []
    batch_size = 25

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            truncated_batch = [t[:5000] for t in batch]
            response = comprehend.batch_detect_key_phrases(TextList=truncated_batch, LanguageCode='en')

            for result in response['ResultList']:
                phrases_all.extend([kp['Text'].lower() for kp in result['KeyPhrases'] if kp['Score'] > 0.8])

        except Exception as e:
            print(f"Key phrase extraction error: {e}")
            continue

    return phrases_all


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


def generate_insights_summary(sentiment_records):
    if not sentiment_records:
        return None

    positive_reviews = [r for r in sentiment_records if r['sentiment'] == 'positive']
    negative_reviews = [r for r in sentiment_records if r['sentiment'] == 'negative']

    positive_texts = [r['review'] for r in positive_reviews[:50]]
    negative_texts = [r['review'] for r in negative_reviews[:50]]

    positive_phrases = extract_key_phrases(positive_texts) if positive_texts else []
    negative_phrases = extract_key_phrases(negative_texts) if negative_texts else []

    positive_topics = Counter(positive_phrases).most_common(10)
    negative_topics = Counter(negative_phrases).most_common(10)

    positive_samples = [r['review'][:200] + "..." for r in positive_reviews[:3]]
    negative_samples = [r['review'][:200] + "..." for r in negative_reviews[:3]]

    summary_parts = []

    if positive_topics:
        summary_parts.append(f"Customers frequently praise: {', '.join([t[0] for t in positive_topics[:5]])}")

    if negative_topics:
        summary_parts.append(f"Common complaints include: {', '.join([t[0] for t in negative_topics[:5]])}")

    return {
        "common_positive_topics": [{"topic": t[0], "mentions": t[1]} for t in positive_topics],
        "common_negative_topics": [{"topic": t[0], "mentions": t[1]} for t in negative_topics],
        "positive_review_samples": positive_samples,
        "negative_review_samples": negative_samples,
        "narrative_summary": ". ".join(
            summary_parts) + "." if summary_parts else "Reviews discuss various aspects of the product."
    }


def summarize_reviews(sentiment_records, asin):
    if not sentiment_records:
        return {"error": "No reviews found"}

    counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    total_rating = 0

    for record in sentiment_records:
        sentiment = record.get("sentiment", "neutral")
        counts[sentiment] = counts.get(sentiment, 0) + 1
        total_rating += record.get("rating", 0)

    total = len(sentiment_records)
    avg_rating = total_rating / total if total > 0 else 0
    insights = generate_insights_summary(sentiment_records)

    return {
        "asin": asin,
        "total_reviews": total,
        "average_rating": round(avg_rating, 2),
        "sentiment_distribution": {k: {"count": v, "percentage": round(v / total * 100, 1)} for k, v in counts.items()},
        "sentiment_summary": f"{counts['positive']} positive, {counts['negative']} negative, {counts['neutral']} neutral, {counts['mixed']} mixed. Avg rating: {avg_rating:.2f}",
        "content_insights": insights
    }


def save_to_s3(data, key):
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=f"{PREFIX}/{key}", Body=json.dumps(data, indent=2))


def lambda_handler(event, context):
    asin = event.get("asin", "B0002XL2V6")

    query = f'''
        SELECT asin, reviewtext, overall, tfidf_features
        FROM "{DATABASE}"."{TABLE}"
        WHERE asin = '{asin}'
        LIMIT 200
    '''

    try:
        rows = run_athena_query(query)
    except Exception as e:
        return {"status": "error", "message": str(e), "asin": asin}

    if not rows:
        return {"status": "no_reviews_found", "asin": asin}

    sentiment_records = []
    for r in rows:
        try:
            data = r["Data"]
            text = data[1].get("VarCharValue")
            rating = float(data[2].get("VarCharValue"))
            tfidf_vector = parse_tfidf(data[3].get("VarCharValue"))

            if text and rating is not None:
                sentiment_records.append({
                    "asin": asin,
                    "review": text,
                    "rating": rating,
                    "tfidf_features": tfidf_vector
                })

        except Exception as e:
            print(f"Error parsing row: {e}")
            continue

    if not sentiment_records:
        return {"status": "no_valid_reviews", "asin": asin}

    texts = [r["review"] for r in sentiment_records]
    sentiments = classify_sentiment(texts)
    for i, s in enumerate(sentiments):
        sentiment_records[i]["sentiment"] = s

    summary = summarize_reviews(sentiment_records, asin)
    timestamp = int(time.time())

    save_to_s3(sentiment_records, f"{asin}/{timestamp}_reviews.json")
    save_to_s3(summary, f"{asin}/{timestamp}_summary.json")

    return {
        "status": "success",
        "asin": asin,
        "reviews_processed": len(sentiment_records),
        "summary": summary,
        "reviews": sentiment_records,
        "s3_location": f"s3://{OUTPUT_BUCKET}/{PREFIX}/{asin}/"
    }
