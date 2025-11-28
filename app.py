import streamlit as st
import boto3
import json

s3 = boto3.client("s3")
OUTPUT_BUCKET = "pj-cs6240"
PREFIX = "lambda"


def list_files(asin):
    """
    List JSON files for a given ASIN.
    """
    prefix = f"{PREFIX}/{asin}/"
    response = s3.list_objects_v2(Bucket=OUTPUT_BUCKET, Prefix=prefix)
    files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith(".json")]
    return sorted(files)


def load_json_from_s3(key):
    """
    Load JSON content from S3.
    """
    obj = s3.get_object(Bucket=OUTPUT_BUCKET, Key=key)
    data = json.loads(obj['Body'].read().decode('utf-8'))
    return data


def filter_by_sentiment(reviews, sentiment_types):
    return [r for r in reviews if r['sentiment'] in sentiment_types]


st.set_page_config(page_title="Product Review Explorer", layout="wide")
st.title("Product Review Explorer")


def list_existing_asins():
    paginator = s3.get_paginator("list_objects_v2")
    asin_set = set()
    for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=PREFIX + "/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) > 1:
                asin_set.add(parts[1])
    return sorted(list(asin_set))


existing_asins = list_existing_asins()

asin_input = st.selectbox(
    "Select an ASIN",
    options=existing_asins,
    index=0 if existing_asins else -1
)

sentiment_filter = st.multiselect("Filter by Sentiment",
                                  options=["positive", "negative", "neutral", "mixed"],
                                  default=["positive", "negative", "neutral", "mixed"]
                                  )

if asin_input:
    st.info(f"Loading reviews for ASIN: {asin_input} ...")
    files = list_files(asin_input)

    if not files:
        st.warning("No review files found for this ASIN.")
    else:
        all_reviews = []
        for f in files:
            data = load_json_from_s3(f)
            if isinstance(data, list):
                all_reviews.extend(data)
            elif isinstance(data, dict) and "reviews" in data:
                all_reviews.extend(data["reviews"])

        # Filter by sentiment
        filtered_reviews = filter_by_sentiment(all_reviews, sentiment_filter)

        st.write(f"Total reviews loaded: {len(all_reviews)}")
        st.write(f"Reviews after sentiment filter: {len(filtered_reviews)}")

        # Show table of reviews
        if filtered_reviews:
            for r in filtered_reviews[:50]:
                st.subheader(f"Rating: {r['rating']} | Sentiment: {r['sentiment']}")
                st.write(r['review'])
                if 'tfidf_features' in r:
                    st.write("TF-IDF Features:")
                    st.json(r['tfidf_features'])

        # Show summary
        summary_files = [f for f in files if "summary" in f]
        if summary_files:
            summary_data = load_json_from_s3(summary_files[-1])
            st.subheader("Review Summary")
            st.json(summary_data.get("Summary", {}))
