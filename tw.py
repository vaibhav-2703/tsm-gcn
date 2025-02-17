import os
import re
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

###############################################
# 1. Data Loading and Preprocessing
###############################################

data_path = "twitter-2016train-A.txt"  # Ensure this file is in your working directory.
df = pd.read_csv(data_path, sep="\t", header=None, names=["id", "sentiment", "text"])
df = df.dropna()

# Map sentiment to integer labels: negative=0, neutral=1, positive=2
sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(sentiment_map)

# Preprocess tweet text: extract words, mentions (@) and hashtags (#)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s@#]", "", text)  # Remove punctuation but keep @ and #
    tokens = text.split()
    words = []
    mentions = []
    hashtags = []
    for token in tokens:
        if token.startswith("@"):
            mentions.append(token)
        elif token.startswith("#"):
            hashtags.append(token)
        else:
            words.append(token)
    return words, mentions, hashtags

all_words = set()
all_mentions = set()
all_hashtags = set()
processed_tweets = []  # list of tuples: (words, mentions, hashtags)
for text in df["text"]:
    words, mentions, hashtags = preprocess_text(text)
    processed_tweets.append((words, mentions, hashtags))
    all_words.update(words)
    all_mentions.update(mentions)
    all_hashtags.update(hashtags)

word_list = sorted(list(all_words))
mention_list = sorted(list(all_mentions))
hashtag_list = sorted(list(all_hashtags))

###############################################
# 2. Building the Knowledge Graph
###############################################

# We create a heterogeneous graph with four types of nodes:
# 1. Tweet nodes: indices [0, num_tweets - 1]
# 2. Word nodes: indices [num_tweets, num_tweets + num_words - 1]
# 3. Mention nodes: indices [num_tweets + num_words, num_tweets + num_words + num_mentions - 1]
# 4. Hashtag nodes: indices [num_tweets + num_words + num_mentions, num_nodes - 1]
num_tweets = df.shape[0]
num_words = len(word_list)
num_mentions = len(mention_list)
num_hashtags = len(hashtag_list)
num_nodes = num_tweets + num_words + num_mentions + num_hashtags

# Create mapping dictionaries
word2idx = {word: i + num_tweets for i, word in enumerate(word_list)}
mention2idx = {m: i + num_tweets + num_words for i, m in enumerate(mention_list)}
hashtag2idx = {h: i + num_tweets + num_words + num_mentions for i, h in enumerate(hashtag_list)}

edges = []

# For each tweet, add edges to its words, mentions, and hashtags
for tweet_idx, (words, mentions, hashtags) in enumerate(processed_tweets):
    for word in set(words):
        if word in word2idx:
            edges.append((tweet_idx, word2idx[word]))
    for m in set(mentions):
        if m in mention2idx:
            edges.append((tweet_idx, mention2idx[m]))
    for h in set(hashtags):
        if h in hashtag2idx:
            edges.append((tweet_idx, hashtag2idx[h]))

# Create undirected edges (bidirectional)
undirected_edges = edges + [(v, u) for u, v in edges]

# Build the graph using NetworkX
G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(undirected_edges)
# Add self-loops for stability in GCN and for network analysis
G.add_edges_from([(i, i) for i in range(num_nodes)])

###############################################
# 3. Network Analysis (MinorProject.pdf Methods)
###############################################

# Compute the normalized adjacency matrix (for GCN later)
def normalize_adj(adj):
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj).dot(D_inv_sqrt)

adj = nx.adjacency_matrix(G)
adj_norm = normalize_adj(adj)

# Convert scipy sparse matrix to torch sparse tensor using torch.sparse_coo_tensor (avoiding deprecation warnings)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

A_hat = sparse_mx_to_torch_sparse_tensor(adj_norm)

# Compute the largest eigenvalue of the (dense) adjacency matrix for epidemic threshold estimation
A_dense = nx.to_numpy_array(G)
eigvals = np.linalg.eigvals(A_dense)
lambda1 = max(np.abs(eigvals))
tau = 1.0 / lambda1 if lambda1 != 0 else None
print(f"Estimated largest eigenvalue (lambda1): {lambda1:.4f}")
print(f"Epidemic threshold (tau = 1/lambda1): {tau:.4f}")

# Implement a simple Independent Cascade Model (ICM) simulation
def independent_cascade(G, seeds, beta=0.05, max_steps=10):
    """
    Simulate Independent Cascade Model on graph G.
    :param G: NetworkX graph.
    :param seeds: Initial set of active nodes.
    :param beta: Activation probability.
    :param max_steps: Maximum number of simulation steps.
    :return: Set of all activated nodes.
    """
    active = set(seeds)
    newly_active = set(seeds)
    step = 0
    while newly_active and step < max_steps:
        next_active = set()
        for node in newly_active:
            for neighbor in G.neighbors(node):
                if neighbor not in active:
                    # Activate with probability beta
                    if random.random() < beta:
                        next_active.add(neighbor)
        newly_active = next_active
        active.update(newly_active)
        step += 1
    return active

# Run ICM simulation on the knowledge graph using a random set of seed tweet nodes
seed_count = 5
seed_nodes = random.sample(range(num_tweets), seed_count)
activated = independent_cascade(G, seed_nodes, beta=0.05, max_steps=10)
print(f"ICM Simulation: Seed nodes = {seed_nodes}")
print(f"Total activated nodes (cascade size): {len(activated)} out of {num_nodes}")

###############################################
# 4. Sentiment Analysis using GCN
###############################################

# Only tweet nodes (indices 0 to num_tweets-1) have sentiment labels.
all_indices = np.arange(num_tweets)
train_idx, test_idx, y_train, y_test = train_test_split(
    all_indices, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"].values
)
train_idx, val_idx, y_train, y_val = train_test_split(
    train_idx, y_train, test_size=0.1, random_state=42, stratify=y_train
)
train_idx = torch.LongTensor(train_idx)
val_idx = torch.LongTensor(val_idx)
test_idx = torch.LongTensor(test_idx)

# Define the GCN Model Components
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = dropout
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.spmm(adj, x)
        x = self.linear(x)
        return x

class GCN(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        # Learnable node embeddings (initial features)
        self.embedding = nn.Embedding(num_nodes, in_dim)
        self.gcn1 = GCNLayer(in_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, num_classes, dropout)
        
    def forward(self, adj):
        x = self.embedding.weight  # shape: (num_nodes, in_dim)
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = self.gcn2(x, adj)
        return x

# Hyperparameters for sentiment analysis
in_dim = 200         # Dimension of node embeddings
hidden_dim = 128     # Hidden layer dimension
num_classes = 3      # Sentiment classes: negative, neutral, positive
dropout = 0.5
learning_rate = 0.01
weight_decay = 5e-4
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(num_nodes, in_dim, hidden_dim, num_classes, dropout).to(device)
A_hat = A_hat.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
labels = torch.LongTensor(df["label"].values).to(device)

# Training Loop for Sentiment Analysis
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(A_hat)
    # Compute loss only on tweet nodes
    loss = criterion(output[:num_tweets][train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits = model(A_hat)
            val_loss = criterion(logits[:num_tweets][val_idx], labels[val_idx])
            preds = torch.argmax(logits[:num_tweets][val_idx], dim=1)
            correct = (preds == labels[val_idx]).sum().item()
            acc = correct / len(val_idx)
        model.train()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {acc:.4f}")

# Testing and Detailed Evaluation
model.eval()
with torch.no_grad():
    logits = model(A_hat)
    preds = torch.argmax(logits[:num_tweets][test_idx], dim=1)
    correct = (preds == labels[test_idx]).sum().item()
    test_acc = correct / len(test_idx)
print(f"Test Accuracy: {test_acc:.4f}")

report = classification_report(labels[test_idx].cpu(), preds.cpu(),
                               target_names=["negative", "neutral", "positive"])
print(report)

# Save the trained model
torch.save(model.state_dict(), "gcn_twitter_sentiment.pth")

# The project integrates a robust knowledge graph from the Twitter dataset by linking tweets,
# words, mentions, and hashtags. It then computes key network properties (e.g. epidemic threshold)
# and simulates information spread using the Independent Cascade Model—concepts inspired by minorproject.pdf :contentReference[oaicite:0]{index=0}.
# Finally, a Graph Convolutional Network is trained to perform sentiment analysis on tweets.
