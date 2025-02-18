import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import nltk
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('/mnt/data/1976-2020-president.csv')

df = df[['year', 'party_simplified', 'candidate', 'candidatevotes', 'totalvotes']].dropna()

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
for keyword, cluster in keyword_clusters.items():
    G.add_node(keyword, cluster=cluster)
for i in range(len(keywords)):
    for j in range(i + 1, len(keywords)):
        if kmeans.labels_[i] == kmeans.labels_[j]:
            G.add_edge(keywords[i], keywords[j])

# Convert NetworkX graph to PyTorch Geometric format
node_idx = {node: i for i, node in enumerate(G.nodes())}
edge_index = torch.tensor([[node_idx[u], node_idx[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
num_nodes = len(G.nodes())
node_features = torch.rand((num_nodes, 16))  # Random embeddings for now

data = Data(x=node_features, edge_index=edge_index)

# Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, torch.randint(0, 2, (num_nodes,)))
    loss.backward()
    optimizer.step()

# Visualize node embeddings
embeddings = model(data.x, data.edge_index).detach().numpy()
plt.figure(figsize=(10, 6))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c='blue', alpha=0.6)
plt.title("Graph Embeddings Visualization after GCN Training")
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")
plt.show()
