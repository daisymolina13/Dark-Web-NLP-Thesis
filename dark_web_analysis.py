# Honors Thesis - Daisy Molina
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("drug_listings.csv")

# --- Quick Check ---
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Step 2: Preprocessing

# Load spaCy model 
nlp = spacy.load("en_core_web_sm")

# Clean text function 
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text

# Lemmatize function 
def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and len(token) > 2])

# Apply cleaning 
print("Cleaning text... this may take a few minutes")
df['cleaned'] = df['product_description'].apply(clean_text)

#  Apply lemmatization (on a sample first to test) 
print("Lemmatizing sample...")
sample_df = df.head(500).copy()
sample_df['lemmatized'] = sample_df['cleaned'].apply(lemmatize)

# Check results 
print("\nSample of cleaned vs lemmatized:")
print(sample_df[['product_description', 'cleaned', 'lemmatized']].head(3))
print("\nPreprocessing complete!")

# Step 3: TF-IDF Feature Extraction

print("Running TF-IDF...")

# Initialize TF-IDF 
tfidf = TfidfVectorizer(
    max_features=1000,    # top 1000 most important words
    min_df=5,             # word must appear in at least 5 listings
    max_df=0.95,          # ignore words appearing in 95%+ of listings
    ngram_range=(1, 2)    # capture single words AND two-word phrases
)

# Fit on lemmatized sample 
tfidf_matrix = tfidf.fit_transform(sample_df['lemmatized'])

#  Check results 
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("\nTop 20 most important terms:")
feature_names = tfidf.get_feature_names_out()
print(list(feature_names[:20]))

print("\nTF-IDF complete!")

# Step 4: K-Means Clustering

print("Running clustering...")

# Find optimal number of clusters 
inertia = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(tfidf_matrix)
    inertia.append(km.inertia_)

#  Plot elbow curve to find best k 
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method - Finding Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('elbow_curve.png')
print("Elbow curve saved as elbow_curve.png")

# Apply KMeans with k=5 (we'll adjust after seeing elbow curve)
km_model = KMeans(n_clusters=5, random_state=42, n_init=10)
sample_df['cluster'] = km_model.fit_predict(tfidf_matrix)

# Check cluster distribution 
print("\nCluster distribution:")
print(sample_df['cluster'].value_counts().sort_index())

# Top terms per cluster 
print("\nTop 10 terms per cluster:")
order_centroids = km_model.cluster_centers_.argsort()[:, ::-1]
for i in range(5):
    terms = [feature_names[ind] for ind in order_centroids[i, :10]]
    print(f"Cluster {i}: {terms}")

print("\nClustering complete!")

# Step 5: Personality Profiling Big Five

print("Running personality profiling...")

# --- Word lists based on Big Five markers ---
# Simplified lexicon-based proxies for each trait
personality_lexicon = {
    'openness': ['unique', 'original', 'creative', 'special', 'exotic', 
                 'rare', 'experience', 'variety', 'different', 'new'],
    'conscientiousness': ['guarantee', 'discreet', 'reliable', 'secure', 
                          'safe', 'trusted', 'professional', 'quality', 
                          'careful', 'precise'],
    'extraversion': ['best', 'amazing', 'great', 'excellent', 'fantastic', 
                     'love', 'enjoy', 'happy', 'excited', 'awesome'],
    'agreeableness': ['free', 'bonus', 'help', 'support', 'friendly', 
                      'welcome', 'please', 'thank', 'kind', 'good'],
    'neuroticism': ['warning', 'risk', 'danger', 'problem', 'issue', 
                    'worry', 'concern', 'caution', 'fear', 'avoid']
}

# Score each listing 
def score_personality(text):
    if not isinstance(text, str):
        return pd.Series([0, 0, 0, 0, 0])
    words = text.lower().split()
    scores = {}
    for trait, lexicon in personality_lexicon.items():
        scores[trait] = sum(1 for w in words if w in lexicon)
    return pd.Series(scores)

# Apply to sample 
personality_scores = sample_df['lemmatized'].apply(score_personality)
sample_df = pd.concat([sample_df, personality_scores], axis=1)

# Average scores by cluster 
print("\nAverage personality scores by cluster:")
trait_cols = list(personality_lexicon.keys())
cluster_personality = sample_df.groupby('cluster')[trait_cols].mean().round(3)
print(cluster_personality)

# Visualize 
plt.figure(figsize=(10, 6))
cluster_personality.T.plot(kind='bar', figsize=(12, 6))
plt.title('Big Five Personality Traits by Cluster')
plt.xlabel('Personality Trait')
plt.ylabel('Average Score')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('personality_by_cluster.png')
print("\nPersonality chart saved as personality_by_cluster.png")
print("Personality profiling complete!")

# Step 6: Longitudinal Analysis

print("Running longitudinal analysis...")

# Use 'source' as platform proxy for time 
print("\nUnique sources (platforms):")
print(df['source'].value_counts())

# Apply cleaning to full dataset by source 
print("\nCleaning full dataset by source (this may take a few minutes)...")
df['cleaned'] = df['product_description'].apply(clean_text)

# Group by source and get top TF-IDF terms per platform
sources = df['source'].unique()

platform_terms = {}
for source in sources:
    source_df = df[df['source'] == source]['cleaned']
    if len(source_df) < 10:
        continue
    tfidf_temp = TfidfVectorizer(max_features=500, min_df=2, max_df=0.95)
    try:
        matrix = tfidf_temp.fit_transform(source_df)
        terms = tfidf_temp.get_feature_names_out()[:15].tolist()
        platform_terms[source] = terms
    except:
        pass

print("\nTop 15 terms per platform/source:")
for source, terms in platform_terms.items():
    print(f"\nSource {source}: {terms}")

# Plot listing counts per source 
plt.figure(figsize=(10, 5))
df['source'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Number of Listings per Platform Source')
plt.xlabel('Source')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.savefig('listings_per_source.png')
print("\nListings per source chart saved as listings_per_source.png")

# Personality scores across sources
print("\nCalculating personality drift across sources...")
samples = []
for source in df['source'].unique():
    source_data = df[df['source'] == source]
    sample = source_data.sample(min(100, len(source_data)), random_state=42)
    samples.append(sample)

df_sample_full = pd.concat(samples).reset_index(drop=True)

personality_full = df_sample_full['cleaned'].apply(score_personality)
df_sample_full = pd.concat([df_sample_full, personality_full], axis=1)

source_personality = df_sample_full.groupby('source')[trait_cols].mean().round(3)
print("\nPersonality scores by source:")
print(source_personality)

# Plot personality drift 
source_personality.plot(kind='bar', figsize=(12, 6))
plt.title('Personality Trait Drift Across Platform Sources')
plt.xlabel('Source')
plt.ylabel('Average Score')
plt.legend(title='Trait', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('personality_drift.png')
print("\nPersonality drift chart saved as personality_drift.png")
print("\nLongitudinal analysis complete!")
print("\n✓ All steps complete! Check your folder for all saved charts.")
