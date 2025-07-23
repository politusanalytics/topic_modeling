import os
import requests
import random
import sys
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
import re
import emoji
from unicode_tr import unicode_tr
import json
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime
import time
import dateutil.parser
from collections import Counter
import openai
from tqdm import tqdm

random.seed(42)

# === Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "o3"
MIN_TEXT_LENGTH_QUANTILE = 0.15
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CUSTOM_STOPWORDS_FILE = "custom_removal_words.txt"
REDUCE_OUTLIERS_FLAG = True

# Read user-defined list of custom removal words
with open(CUSTOM_STOPWORDS_FILE, encoding="utf-8") as f:
    custom_removal_words = [line.strip() for line in f if line.strip()]

# Read data file from command-line argument
if len(sys.argv) < 5:
    print("Usage: python script.py <file_path> <topic_domain> <min_cluster_size> <num_groups>")
    sys.exit(1)

FILE_PATH = sys.argv[1]              # Path to the input data file (CSV, Excel, or JSON)
TOPIC_DOMAIN = sys.argv[2]           # Descriptive domain for the dataset (e.g., "Koç Üniversitesi")
MIN_CLUSTER_SIZE = int(sys.argv[3])  # Minimum number of documents in a topic cluster
NUM_GROUPS = int(sys.argv[4])        # Desired number of categories to group topic labels into

# Load general Turkish stopwords from GitHub
url = "https://raw.githubusercontent.com/ahmetax/trstop/master/dosyalar/turkce-stop-words"
response = requests.get(url)
tr_stops = response.text.strip().split("\n")

# Add Turkish alphabet letters
turkish_letters = list("abcçdefgğhıijklmnoöprsştuüvyz")
tr_stops.extend(turkish_letters)

# Remove duplicates and sort
tr_stops = sorted(set(tr_stops + custom_removal_words))


### Read and Prepare Data
def clean_tweet(text, tr_stops):
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs (any web links starting with http or www)
    text = emoji.replace_emoji(text, replace="")  # Remove emojis
    text = re.sub(r"(?<!\w)([@#])(?=\W|$)", "", text)  # Remove standalone @ or # that are not part of a word
    text = re.sub(r"[^\w\s@#]", "", text)  # Remove non-alphanumeric characters (except @ and #)
    tokens = text.split()  # Split into tokens based on whitespace
    processed_tokens = []
    for token in tokens:
        token = unicode_tr(token).lower()  # Lowercase using Turkish-specific rules (e.g., "I" → "ı")
        if token.startswith("#"):  # Keep hashtags
            processed_tokens.append(token)
        elif token.startswith("@"):  # Remove mentions
            continue
        elif token in tr_stops:  # Remove stopwords & custom removal words
            continue
        elif token.lower() == "rt":  # Remove "rt" (retweet indicator)
            continue
        elif len(token) == 1:  # Remove single-character tokens
            continue
        else:  # Keep everything else
            processed_tokens.append(token)
    return " ".join(processed_tokens).strip()


### Data Preparation
def load_data(file_path=FILE_PATH, id_col="id", text_col="text", date_col="date"):
    # Read based on file extension
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, dtype=str)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, dtype=str)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
        df[[id_col, text_col, date_col]] = df[[id_col, text_col, date_col]].astype(str)
    elif file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
        df[[id_col, text_col, date_col]] = df[[id_col, text_col, date_col]].astype(str)
    else:
        raise ValueError("Unsupported file type. Use .csv, .xlsx, .json, or .jsonl")

    # Check required columns
    for col in [id_col, text_col, date_col]:
        if col not in df.columns:
            raise ValueError(f"File must contain '{id_col}', '{text_col}', and '{date_col}' columns.")

    # Drop rows with missing values in critical columns
    df = df[[id_col, text_col, date_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).reset_index(drop=True)

    return df

# Load data
df = load_data(FILE_PATH)
print(f"[INFO] Loaded {len(df):,} tweets from {FILE_PATH}")

# Clean text
df["clean_text"] = df["text"].apply(lambda tweet: clean_tweet(tweet, tr_stops))
df = df[df["clean_text"] != ""].reset_index(drop=True)

# Create "date_month"
df["date_month"] = df["date"].dt.to_period("M").astype(str)

# Remove short texts
df["text_length"] = df["clean_text"].str.len()
df = df[df["text_length"] >= df["text_length"].quantile(MIN_TEXT_LENGTH_QUANTILE)].copy().reset_index(drop=True)
print(f"[INFO] Remaining tweets after cleaning and filtering: {len(df):,}")

# Get cleaned tweet texts and timestamps
tweets = list(df["clean_text"])
timestamps = list(df["date_month"])

### Create Topic Model
def generate_embeddings(texts, model_name=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_topic_model(tweets, embeddings, min_cluster_size=MIN_CLUSTER_SIZE, reduce_outliers_flag=REDUCE_OUTLIERS_FLAG, tr_stops=None, df=None):
    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', min_dist=0.0, random_state=42)
    vectorizer = CountVectorizer(
        tokenizer=TweetTokenizer(reduce_len=True).tokenize,
        token_pattern=None,
        stop_words=tr_stops,
        ngram_range=(1, 1),
        min_df=2
    )
    cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer,
        hdbscan_model=cluster_model,
        umap_model=umap_model,
        top_n_words=20,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=False
    )
    topics, probs = topic_model.fit_transform(tweets, embeddings)
    if reduce_outliers_flag:
        topics = topic_model.reduce_outliers(documents=tweets, topics=topics, strategy="distributions")
        topic_model.update_topics(tweets, topics=topics, top_n_words=20)
    topic_info_df = topic_model.get_topic_info()
    topic_info_df["Representation"] = topic_info_df["Representation"].apply(lambda x: ', '.join(x))
    topic_info_df["Representative_Docs"] = topic_info_df["Representative_Docs"].apply(lambda x: ' | '.join(x))
    representative_docs_original = []
    for i in range(len(topic_info_df)):
        representative_docs = topic_info_df.loc[i, "Representative_Docs"].split(" | ")
        originals = []
        for doc in representative_docs:
            match = df[df["clean_text"] == doc]
            if not match.empty:
                originals.append(match.iloc[0]["text"])
        representative_docs_original.append(" | ".join(originals))
    topic_info_df["Representative_Docs_Original"] = representative_docs_original
    return topic_model, topic_info_df

#### Get Topic Label
def get_topic_label(client, topic_number, count, representation, representative_docs, topic_domain=TOPIC_DOMAIN, max_retries=5):
    tweets_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(representative_docs)])
    system_prompt = {
        "role": "system",
        "content": (
            "Sen bir metin analizi yardımcısısın. Görevin, sosyal medya gönderilerinden çıkarılmış konulara "
            "kısa, açıklayıcı ve birbirleriyle tutarlı başlıklar vermektir. Benzer konulara benzer başlıklar vererek "
            "etiketlerin karşılaştırılabilir olmasını sağla. Etiket yalnızca birkaç kelimeden oluşmalıdır."
        )
    }
    user_prompt = {
    "role": "user",
    "content": (
        f"Aşağıda, BERTopic ile çıkarılmış bir konunun bilgileri yer almaktadır. "
        f"Tweetler {topic_domain} ile ilgilidir.\n"
        f"Lütfen bu bilgilere dayanarak bu konuya kısa, açıklayıcı ve tutarlı bir etiket ver. "
        f"Etiket yalnızca Türkçe ve mümkünse birkaç kelime ile sınırlı olmalıdır.\n\n"
        f"Bilgiler:\n"
        f"- Konu numarası: {topic_number}\n"
        f"- Tweet sayısı: {count}\n"
        f"- Temsili kelimeler: {representation}\n"
        f"- Örnek tweetler:\n"
        f"{tweets_text}\n\n"
        f"Lütfen sadece etiketi yaz."
    )
}
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[system_prompt, user_prompt]
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait_time = attempt * 60
            print(f"[RateLimit] Waiting {wait_time}s before retrying... (Attempt {attempt}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[Error] Unexpected error on attempt {attempt}: {e}")
            break
    return "API Error: Rate limit exceeded after retries"


def add_labels_with_openai(topic_df, client, df):
    labels = []
    random_sample_docs_all = []
    random_sample_docs_original_all = []
    for i in tqdm(range(len(topic_df)), desc="Labelling Topics"):
        row = topic_df.iloc[i]
        topic_number = row["Topic"]
        count = row["Count"]
        representation = row["Representation"]
        # Filter all tweets with this topic
        topic_tweets_df = df[df["Topic"] == topic_number]
        # Sample up to 20 preprocessed tweets (clean_text) and their original versions
        sampled_rows = topic_tweets_df.sample(n=min(20, len(topic_tweets_df)), random_state=42) if not topic_tweets_df.empty else pd.DataFrame()
        sampled_preprocessed = sampled_rows["clean_text"].tolist()
        sampled_original = sampled_rows["text"].tolist()
        # Add to the new columns
        random_sample_docs_all.append(" | ".join(sampled_preprocessed))
        random_sample_docs_original_all.append(" | ".join(sampled_original))
        # Get label
        try:
            label = get_topic_label(client, topic_number, count, representation, representative_docs=sampled_original, topic_domain=TOPIC_DOMAIN)
        except Exception as e:
            print(f"[Error] Failed to label topic {topic_number}: {e}")
            label = f"API Error: {e}"
        labels.append(label)
    topic_df["Label"] = labels
    topic_df["Random_Sample_Docs"] = random_sample_docs_all
    topic_df["Random_Sample_Docs_Original"] = random_sample_docs_original_all
    return topic_df


def group_labels_with_openai(topic_df, df_tweets, client, topic_domain=TOPIC_DOMAIN, num_groups=NUM_GROUPS, max_retries=5):
    unique_labels = sorted(topic_df["Label"].dropna().unique().tolist())

    system_prompt = {
        "role": "system",
        "content": (
            "Sen bir metin analizi yardımcısısın. Aşağıda verilen konu etiketlerini içeriklerine göre gruplamakla görevlisin. "
            "Etiketleri içerik benzerliğine göre mantıklı kategorilere ayırmalı ve her kategoriye açıklayıcı bir başlık vermelisin. "
            "Her etiketi yalnızca bir kategoriye ata. Toplamda belirtilen sayıda kategori oluştur."
        )
    }

    user_prompt = {
        "role": "user",
        "content": (
            f"Aşağıda, {topic_domain} ile ilgili sosyal medya konularına ait etiketler bulunmaktadır.\n"
            f"Lütfen bu etiketleri içeriklerine göre {num_groups} farklı kategoriye ayır. "
            f"Her kategori için açıklayıcı bir başlık bul ve her etiketi yalnızca bir kategoriye ata.\n\n"
            f"Çıktıyı şu formatta ver:\n"
            f"<Etiket>: <Kategori Başlığı>\n"
            f"<Etiket>: <Kategori Başlığı>\n"
            f"...\n\n"
            f"Etiket Listesi:\n"
            + "\n".join(f"- {label}" for label in unique_labels)
        )
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[system_prompt, user_prompt],
            )
            output_text = response.choices[0].message.content.strip()
            break
        except openai.RateLimitError:
            wait_time = attempt * 60
            print(f"[RateLimit] Waiting {wait_time}s before retrying... (Attempt {attempt}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[Error] Unexpected error on attempt {attempt}: {e}")
            return topic_df, df_tweets

    label_to_group = {}
    for line in output_text.splitlines():
        if ":" in line:
            label, group = line.split(":", 1)
            label_to_group[label.strip()] = group.strip()

    topic_df["Label_Grouped"] = topic_df["Label"].map(label_to_group)
    df_tweets["Label_Grouped"] = df_tweets["Label"].map(label_to_group)
    return topic_df, df_tweets


if __name__ == "__main__":
    # Run topic modeling & add labels
    embeddings = generate_embeddings(tweets)
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    version_tag = f"minclust{MIN_CLUSTER_SIZE}_{'outliers_merged' if REDUCE_OUTLIERS_FLAG else 'outliers_kept'}"
    print(f"\nRunning version: {version_tag}")

    # Step 1: Build topic model and extract topic_df
    topic_model, topic_df = build_topic_model(tweets, embeddings, MIN_CLUSTER_SIZE, REDUCE_OUTLIERS_FLAG, tr_stops, df)
    # Step 2: Assign topics to individual tweets
    df_tweets = pd.concat([df[["id", "text", "clean_text"]], pd.Series(topic_model.topics_, name="Topic")], axis=1)
    # Step 3: Add labels and random samples to topic_df
    topic_df = add_labels_with_openai(topic_df, client, df_tweets)
    # Step 4: Reorder topic_df columns to move "Label" to the end
    desired_order = [
        "Topic", "Count", "Name", "Representation",
        "Representative_Docs", "Representative_Docs_Original",
        "Random_Sample_Docs", "Random_Sample_Docs_Original", "Label"
    ]
    topic_df = topic_df[[col for col in desired_order if col in topic_df.columns]]
    # Step 5: Add topic labels to df_tweets
    topic_to_label = dict(zip(topic_df["Topic"], topic_df["Label"]))
    df_tweets["Label"] = df_tweets["Topic"].map(topic_to_label)
    # Step 6: Group labels into categories using OpenAI
    topic_df, df_tweets = group_labels_with_openai(topic_df, df_tweets, client, num_groups=NUM_GROUPS)
    # Step 7: Export outputs
    topic_df.to_excel(f"Topics_{version_tag}.xlsx", index=False)
    df_tweets.to_excel(f"Tweets_{version_tag}.xlsx", index=False)

