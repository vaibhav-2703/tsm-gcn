import os
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
import nltk
nltk.download('punkt')

# Force GPU usage if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Warning: Running on CPU!")

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
node_features = torch.rand((num_nodes, 16)).to(device)  # Placeholder node embeddings

data = Data(x=node_features, edge_index=edge_index).to(device)

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

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    labels = torch.randint(0, 3, (num_nodes,)).to(device)  # Placeholder labels
    loss = F.nll_loss(out, labels)
    loss.backward()
    optimizer.step()

# Pre-Poll Prediction using SARIMAX
poll_results = df.groupby('year').agg({'candidatevotes': 'sum', 'totalvotes': 'sum'})
poll_results['vote_share'] = poll_results['candidatevotes'] / poll_results['totalvotes']

# Adjust SARIMAX parameters to avoid conflicts
model_sarimax = SARIMAX(poll_results['vote_share'], order=(2,1,2), seasonal_order=(1,1,1,4))
model_fit = model_sarimax.fit()
future_years = np.arange(df['year'].max() + 4, df['year'].max() + 20, 4)
predicted_vote_share = model_fit.forecast(steps=len(future_years))

# Visualizing Pre-Poll Predictions with Actual Results
plt.figure(figsize=(12,6))
sns.lineplot(x=poll_results.index, y=poll_results['vote_share'], label='Historical Vote Share', marker='o', linewidth=2)
sns.lineplot(x=future_years, y=predicted_vote_share, label='Predicted Vote Share', marker='o', linestyle='dashed', linewidth=2)
plt.axvline(x=df['year'].max(), color='red', linestyle='dotted', label='Last Known Election')
plt.title('Pre-Poll Prediction vs Actual Election Results')
plt.xlabel('Year')
plt.ylabel('Vote Share')
plt.legend()
plt.grid(True)
plt.show()
