import pandas as pd
import networkx as nx
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

num_tweets = 10000
dataset_path = "sentiment140.csv"

df = pd.read_csv(dataset_path, encoding='latin-1', header=None, nrows=num_tweets)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = df[['text', 'target']].dropna()
df['sentiment'] = df['target'].map({0: 'negative', 4: 'positive'})
df = df[['text', 'sentiment']]

G = nx.Graph()
for i, row in df.iterrows():
    G.add_node(f"tweet_{i}", text=row['text'], sentiment=row['sentiment'])

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['text']).astype('float32')
tfidf_array = tfidf_matrix.toarray()

norms = np.linalg.norm(tfidf_array, axis=1, keepdims=True)
tfidf_norm = tfidf_array / (norms + 1e-10)

d = tfidf_norm.shape[1]
index = faiss.IndexFlatIP(d)
index.add(np.ascontiguousarray(tfidf_norm))

k = 3
similarities, indices = index.search(np.ascontiguousarray(tfidf_norm), k)

threshold = 0.80
for i in range(len(df)):
    if similarities[i][1] > threshold:
        neighbor = indices[i][1]
        sim_score = float(similarities[i][1])
        G.add_edge(f"tweet_{i}", f"tweet_{neighbor}", weight=sim_score)

nx.write_gml(G, "sentiment140_graph.gml")
print(f"Graph saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

sub_nodes = list(G.nodes)[:200]
sub_graph = G.subgraph(sub_nodes)
pos = nx.spring_layout(sub_graph)
nx.draw(sub_graph, pos, node_size=20, with_labels=False)
plt.show()
