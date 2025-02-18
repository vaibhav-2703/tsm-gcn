import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('/mnt/data/1976-2020-president.csv')

# Sample preprocessing: assuming 'tweet' column exists (replace accordingly)
df = df[['year', 'party_simplified', 'candidate', 'candidatevotes', 'totalvotes']]
df = df.dropna()

# Compute sentiment scores (Placeholder: Replace with proper sentiment model)
def sentiment_analysis(text):
    return np.random.uniform(-1, 1)  # Dummy sentiment scores

df['sentiment'] = df['candidate'].apply(sentiment_analysis)

# Extract keywords using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
tfidf_matrix = vectorizer.fit_transform(df['candidate'].astype(str))
keywords = vectorizer.get_feature_names_out()

# Cluster keywords
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix.T)
keyword_clusters = {keywords[i]: kmeans.labels_[i] for i in range(len(keywords))}

# Create graph
G = nx.Graph()

# Add keyword nodes with cluster as attribute
for keyword, cluster in keyword_clusters.items():
    G.add_node(keyword, cluster=cluster)

# Add edges based on co-occurrence
for i in range(len(keywords)):
    for j in range(i + 1, len(keywords)):
        if kmeans.labels_[i] == kmeans.labels_[j]:
            G.add_edge(keywords[i], keywords[j])

# Plot knowledge graph
plt.figure(figsize=(12, 8))
colors = [G.nodes[n]['cluster'] for n in G.nodes()]
nx.draw(G, with_labels=True, node_color=colors, cmap=plt.cm.rainbow, node_size=3000, edge_color='gray')
plt.title("Knowledge Graph of Election Data (Keyword Clustering)")
plt.show()
