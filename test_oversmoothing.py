import torch
import src.util as ut
from torch_geometric.data import Data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize_node_embeddings(data: Data):
    """
    Visualize node embeddings in 2D using PCA.
    """
    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(data.x.detach().cpu().numpy())
    plt.scatter(reduced_x[:, 0], reduced_x[:, 1], alpha=0.7)
    plt.title("Node Embeddings")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig('output/output.png')


def check_feature_similarity(data: Data):
    """
    Compute the average pairwise cosine similarity of node features.
    """
    x = data.x  # Node features
    norm_x = torch.nn.functional.normalize(x, p=2, dim=1)  # Normalize along feature dim
    similarity_matrix = torch.matmul(norm_x, norm_x.T)  # Cosine similarity matrix

    # Exclude diagonal (self-similarity)
    num_nodes = similarity_matrix.size(0)
    mean_similarity = (similarity_matrix.sum() - num_nodes) / (num_nodes * (num_nodes - 1))

    return mean_similarity.item()


data = Data(x=torch.randn(100, 64))  # Random graph with 100 nodes and 64 features
mean_similarity = check_feature_similarity(data)
visualize_node_embeddings(data)
print(f"Mean Pairwise Similarity: {mean_similarity:.4f}")
