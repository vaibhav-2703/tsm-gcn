import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.decomposition import PCA
from transformers import pipeline
import nltk
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('/mnt/data/1976-2020-president.csv')
df = df[['year', 'party_simplified', 'candidate', 'candidatevotes', 'totalvotes']].dropna()

# Sentiment Analysis using a Pretrained Model
sentiment_pipeline = pipeline("sentiment-analysis")
df['sentiment'] = df['candidate'].apply(lambda x: sentiment_pipeline(x)[0]['label'])

# Extract keywords using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
tfidf_matrix = vectorizer.fit_transform(df['candidate'].astype(str))
keywords = vectorizer.get_feature_names_out()

# Cluster keywords
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix.T)
keyword_clusters = {keywords[i]: kmeans.labels_[i] for i in range(len(keywords))}

# Create Knowledge Graph with Tweets & Keywords
G = nx.Graph()
for keyword, cluster in keyword_clusters.items():
    G.add_node(keyword, cluster=cluster, node_type='keyword')
for i, row in df.iterrows():
    tweet_node = f"Tweet_{i}"
    G.add_node(tweet_node, sentiment=row['sentiment'], node_type='tweet')
    for keyword in keywords:
        if keyword in row['candidate'].lower():
            G.add_edge(tweet_node, keyword)

# Convert to PyTorch Geometric format
node_idx = {node: i for i, node in enumerate(G.nodes())}
edge_index = torch.tensor([[node_idx[u], node_idx[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
num_nodes = len(G.nodes())
node_features = torch.rand((num_nodes, 16))  # Placeholder node embeddings

data = Data(x=node_features, edge_index=edge_index)

# Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 3)

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
    labels = torch.randint(0, 3, (num_nodes,))  # Placeholder labels
    loss = F.nll_loss(out, labels)
    loss.backward()
    optimizer.step()

# Visualize Interactive Knowledge Graph
pos = nx.spring_layout(G)
node_x, node_y, node_text, node_color = [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node} ({G.nodes[node]['node_type']})")
    node_color.append(0 if G.nodes[node]['node_type'] == 'keyword' else 1)
node_trace = go.Scatter(
    x=node_x, y=node_y, mode='markers+text', text=node_text,
    marker=dict(showscale=True, colorscale='Rainbow', size=15, color=node_color, colorbar=dict(title='Node Type'))
)
fig = go.Figure(data=[node_trace])
fig.update_layout(title="Interactive Knowledge Graph (Tweets & Keywords)", showlegend=False, hovermode='closest')
fig.show()

# Visualize GCN embeddings
embeddings = model(data.x, data.edge_index).detach().numpy()
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6)
plt.title("Graph Embeddings Visualization after GCN Training")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
