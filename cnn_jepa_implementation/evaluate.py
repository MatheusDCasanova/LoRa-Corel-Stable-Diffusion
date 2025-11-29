import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from umap import UMAP
from torch.utils.data import DataLoader
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

from cnn_jepa import CNNJEPA

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32

class CorelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dirs, transform=None):
        self.image_paths = []
        for root_dir in root_dirs:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, "*.png")))
        self.transform = transform
        # Extract labels from filenames (assuming pattern like "0001_xxxx.png")
        self.labels = [os.path.basename(p).split('_')[0] for p in self.image_paths]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), "0000"

def extract_features(model, dataloader, device):
    """Extract features from the context encoder (without masking)"""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            # Use context encoder without mask to get full features
            features = model.context_encoder(images, mask=None)
            # Global average pooling to get feature vector
            features = features.mean(dim=[2, 3])  # (B, 2048)
            features_list.append(features.cpu().numpy())
            labels_list.extend(labels)
    
    features = np.concatenate(features_list, axis=0)
    return features, labels_list

def perform_clustering(features, n_clusters=6):
    """Perform multiple clustering algorithms"""
    print("\n" + "="*50)
    print("Performing clustering...")
    
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / (norms + 1e-10)
    
    results = {}
    
    # 1. KMeans
    print("Running KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    results['KMeans'] = kmeans.fit_predict(features_norm)
    
    # 2. Agglomerative
    print("Running Agglomerative Clustering...")
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    results['Agglomerative'] = agg.fit_predict(features_norm)
    
    # 3. Spectral
    print("Running Spectral Clustering...")
    # Use fewer components for spectral to be faster/stable if needed, or just raw
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
    results['Spectral'] = spectral.fit_predict(features_norm)
    
    # 4. GMM
    print("Running GMM...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    results['GMM'] = gmm.fit_predict(features_norm)
    
    # 5. DBSCAN (with PCA)
    print("Running DBSCAN...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    features_pca = pca.fit_transform(features_norm)
    norms_pca = np.linalg.norm(features_pca, axis=1, keepdims=True)
    features_pca = features_pca / (norms_pca + 1e-10)
    
    dbscan = DBSCAN(eps=0.70, min_samples=10)
    results['DBSCAN'] = dbscan.fit_predict(features_pca)
    
    return results

def compute_metrics(features, pred_labels, true_labels):
    """Compute clustering metrics"""
    # Convert string labels to numeric
    unique_labels = sorted(list(set(true_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    true_labels_numeric = np.array([label_map[label] for label in true_labels])
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(true_labels_numeric, pred_labels)
    
    # Silhouette Score (if valid clusters)
    if len(set(pred_labels)) > 1:
        silhouette = silhouette_score(features, pred_labels)
        davies_bouldin = davies_bouldin_score(features, pred_labels)
    else:
        silhouette = -1
        davies_bouldin = -1
    
    return {
        'ARI': ari,
        'Silhouette': silhouette,
        'Davies-Bouldin': davies_bouldin,
        'N_Clusters': len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    }

def visualize_latent_space(features, labels, pred_labels, method='tsne', save_path='latent_space.png'):
    """Visualize latent space using t-SNE or UMAP"""
    print(f"\n" + "="*50)
    print(f"Performing {method.upper()} dimensionality reduction...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:  # umap
        reducer = UMAP(n_components=2, random_state=42)
    
    embedded = reducer.fit_transform(features)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Convert string labels to numeric for coloring
    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_map[label] for label in labels])
    
    # Plot 1: True labels
    scatter1 = axes[0].scatter(embedded[:, 0], embedded[:, 1], 
                               c=numeric_labels, cmap='tab10', 
                               s=30, alpha=0.7, edgecolors='w', linewidth=0.5)
    axes[0].set_title(f'True Labels ({method.upper()})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Dimension 1', fontsize=12)
    axes[0].set_ylabel('Dimension 2', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Class', fontsize=10)
    
    # Plot 2: Predicted clusters
    scatter2 = axes[1].scatter(embedded[:, 0], embedded[:, 1], 
                               c=pred_labels, cmap='tab10', 
                               s=30, alpha=0.7, edgecolors='w', linewidth=0.5)
    axes[1].set_title(f'Predicted Clusters ({method.upper()})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Dimension 1', fontsize=12)
    axes[1].set_ylabel('Dimension 2', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Cluster', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.close()

def visualize_combined_clusters(features, labels, clustering_results, true_labels, save_path='clustering_comparison.png'):
    """
    Visualize Ground Truth and all clustering results in a grid.
    Layout: 2x3 grid
    - Ground Truth
    - KMeans
    - Agglomerative
    - Spectral
    - GMM
    - DBSCAN
    """
    print(f"\n" + "="*50)
    print(f"Generating combined visualization...")
    
    # Compute UMAP embedding once for all plots
    reducer = UMAP(n_components=2, random_state=42)
    embedded = reducer.fit_transform(features)
    
    # Prepare figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Helper to plot
    def plot_embedding(ax, labels, title, ari=None):
        unique_labels = sorted(list(set(labels)))
        # Handle noise in DBSCAN (-1)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            unique_labels.append(-1)
            
        # Create color map
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: c for l, c in zip(unique_labels, colors)}
        if -1 in label_to_color:
            label_to_color[-1] = (0.8, 0.8, 0.8, 1.0) # Grey for noise
            
        # Convert labels to colors
        c = [label_to_color[l] for l in labels]
        
        ax.scatter(embedded[:, 0], embedded[:, 1], c=c, s=10, alpha=0.7)
        
        title_text = title
        if ari is not None:
            title_text += f"\nARI: {ari:.3f}"
        
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 1. Ground Truth
    # Convert string labels to numeric for consistency if needed, but here we just need distinct values
    plot_embedding(axes[0], true_labels, "Ground Truth")
    
    # 2. KMeans
    ari = adjusted_rand_score(true_labels, clustering_results['KMeans'])
    plot_embedding(axes[1], clustering_results['KMeans'], "K-means", ari)
    
    # 3. Agglomerative
    ari = adjusted_rand_score(true_labels, clustering_results['Agglomerative'])
    plot_embedding(axes[2], clustering_results['Agglomerative'], "Agglomerative", ari)
    
    # 4. Spectral
    ari = adjusted_rand_score(true_labels, clustering_results['Spectral'])
    plot_embedding(axes[3], clustering_results['Spectral'], "Spectral", ari)
    
    # 5. GMM
    ari = adjusted_rand_score(true_labels, clustering_results['GMM'])
    plot_embedding(axes[4], clustering_results['GMM'], "GMM", ari)
    
    # 6. DBSCAN
    ari = adjusted_rand_score(true_labels, clustering_results['DBSCAN'])
    plot_embedding(axes[5], clustering_results['DBSCAN'], "DBSCAN", ari)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined visualization: {save_path}")
    plt.close()

def plot_clustering_metrics(kmeans_metrics, dbscan_metrics, save_path='clustering_metrics.png'):
    """Plot clustering metrics comparison"""
    metrics_names = ['ARI', 'Silhouette', 'Davies-Bouldin']
    kmeans_values = [kmeans_metrics[m] for m in metrics_names]
    dbscan_values = [dbscan_metrics[m] for m in metrics_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, kmeans_values, width, label='KMeans', alpha=0.8)
    bars2 = ax.bar(x + width/2, dbscan_values, width, label='DBSCAN', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Clustering Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics plot: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best_cnn_jepa.pth', help='Path to checkpoint')
    parser.add_argument('--n_clusters', type=int, default=6, help='Number of clusters for KMeans')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset_path = "/home/matheuscasanova/workspace/LoRa-Corel-Stable-Diffusion/corel"
    dataset = CorelDataset(root_dirs=[dataset_path], transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = CNNJEPA().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print("✓ Model loaded successfully")
    
    # Extract features
    features, labels = extract_features(model, dataloader, device)
    print(f"✓ Extracted features shape: {features.shape}")
    
    # Perform clustering
    clustering_results = perform_clustering(features, n_clusters=args.n_clusters)
    
    # Compute metrics for all methods
    print("\n" + "="*50)
    print("CLUSTERING METRICS:")
    
    metrics_summary = {}
    
    for method, pred_labels in clustering_results.items():
        print(f"\n{method.upper()}:")
        metrics = compute_metrics(features, pred_labels, labels)
        metrics_summary[method] = metrics
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    # Visualize combined clusters
    visualize_combined_clusters(
        features, labels, clustering_results, labels,
        save_path=os.path.join(args.output_dir, 'clustering_comparison.png')
    )
    
    print("\n" + "="*50)
    print("✓ EVALUATION COMPLETE!")
    print(f"✓ Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
