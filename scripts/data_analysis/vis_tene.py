
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_latent_codes_tsne(latent_codes_1, latent_codes_2, save_path="latent_codes_tsne.png"):

    import torch
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # Combine latent codes for t-SNE
    all_latent_codes = torch.cat([latent_codes_1, latent_codes_2], dim=0).cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(all_latent_codes)

    # Split t-SNE result back into two sets
    tsne_result_1 = tsne_result[:len(latent_codes_1)]
    tsne_result_2 = tsne_result[len(latent_codes_1):]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_result_1[:, 0], tsne_result_1[:, 1], label='Dataset 100STYLES', alpha=0.6)
    plt.scatter(tsne_result_2[:, 0], tsne_result_2[:, 1], label='Dataset LOCO', alpha=0.6)
    plt.legend()
    plt.title("t-SNE Visualization of Latent Codes")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(save_path)
    print(f"Latent codes visualization saved at {save_path}")



def visualize_tsne(latent_features, labels=None, perplexity=30, learning_rate=200, n_iter=1000, random_state=42):
    """
    Visualize latent features using t-SNE.
    
    Parameters:
        latent_features (ndarray): The latent features to be transformed.
        labels (ndarray, optional): The labels for coloring the scatter plot. Default is None.
        perplexity (int, optional): The perplexity for t-SNE. Default is 30.
        learning_rate (float, optional): The learning rate for t-SNE. Default is 200.
        n_iter (int, optional): The number of iterations for t-SNE. Default is 1000.
        random_state (int, optional): The random state for reproducibility. Default is 42.
    """
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state)
    latent_features_2D = tsne.fit_transform(latent_features)

    # Create the scatter plot
    plt.figure(figsize=(10, 7))
    if labels is not None:
        unique_labels = set(labels)
        for label in unique_labels:
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            plt.scatter(latent_features_2D[indices, 0], latent_features_2D[indices, 1], label=f'Class {label}', alpha=0.6, s=20)
    else:
        plt.scatter(latent_features_2D[:, 0], latent_features_2D[:, 1], alpha=0.6, s=20)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE Visualization of Latent Features')
    if labels is not None:
        plt.legend()
    plt.savefig("latent_codes_tsne.png")


# Example usage
# Assuming latent_features is a numpy array with the latent features and labels is an optional array of labels
visualize_tsne(latent_features, labels)

