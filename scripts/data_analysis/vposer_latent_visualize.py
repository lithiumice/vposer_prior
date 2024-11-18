

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from omegaconf import DictConfig, OmegaConf
import logging
from human_body_prior.model_hub import get_vposer_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LatentVisualizerConfig:
    """Configuration class for LatentVisualizer"""
    vposer_ckpt: str
    output_dir: str
    device: str
    point_size: int
    alpha: float
    overlap_threshold: float
    dimensions: List[int]

class LatentVisualizer:
    def __init__(self, config: LatentVisualizerConfig):
        """Initialize LatentVisualizer with configuration"""
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize VPoser model
        try:
            self.vposer = get_vposer_model(
                device=self.device, 
                vposer_ckpt=config.vposer_ckpt
            )
            logger.info("VPoser model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VPoser model: {e}")
            raise

    def extract_poses(self, npz_file: str) -> torch.Tensor:
        """Extract poses from NPZ file"""
        try:
            data = np.load(npz_file)
            poses = data['poses'][:, 1:22].reshape(-1, 63)
            return torch.from_numpy(poses).float().to(self.device)
        except Exception as e:
            logger.error(f"Failed to extract poses from {npz_file}: {e}")
            raise

    def encode_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Encode poses using VPoser"""
        with torch.no_grad():
            return self.vposer.encode(poses).mean

    def process_dataset(self, npz_files: List[str]) -> torch.Tensor:
        """Process a list of NPZ files and return encoded latent codes"""
        latent_codes = []
        for npz_file in npz_files:
            try:
                poses = self.extract_poses(npz_file)
                latent_code = self.encode_poses(poses)
                latent_codes.append(latent_code)
            except Exception as e:
                logger.warning(f"Skipping file {npz_file} due to error: {e}")
                continue
        
        return torch.cat(latent_codes, dim=0)

    def apply_pca(self, latent_codes_1: torch.Tensor, 
                 latent_codes_2: torch.Tensor, 
                 n_components: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply PCA to reduce dimensionality of latent codes"""
        # Ensure equal lengths
        min_len = min(len(latent_codes_1), len(latent_codes_2))
        latent_codes_1 = latent_codes_1[:min_len]
        latent_codes_2 = latent_codes_2[:min_len]

        # Convert to numpy and combine
        codes_1_np = latent_codes_1.cpu().numpy()
        codes_2_np = latent_codes_2.cpu().numpy()
        combined = np.vstack((codes_1_np, codes_2_np))

        # Apply PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(combined)

        # Split and convert back to torch tensors
        reduced_1 = torch.from_numpy(reduced[:len(codes_1_np)])
        reduced_2 = torch.from_numpy(reduced[len(codes_1_np):])

        logger.info(f"PCA reduced dimensions from {combined.shape[1]} to {n_components}")
        return reduced_1, reduced_2

    def visualize_latent_codes(self, 
                             latent_codes_red: torch.Tensor, 
                             latent_codes_blue: torch.Tensor,
                             save_path: str,
                             title_list: List[str] = ["D1", "D2"],
                             vis_title: bool = True):
        """Visualize latent codes with optional overlap highlighting"""
        plt.figure(figsize=(10, 8))
        
        # Convert tensors to numpy for plotting
        red_data = latent_codes_red[:, self.config.dimensions].cpu().numpy()
        blue_data = latent_codes_blue[:, self.config.dimensions].cpu().numpy()

        # Plot points
        plt.scatter(red_data[:, 0], red_data[:, 1], 
                   color='red', label=title_list[0], 
                   s=self.config.point_size, alpha=self.config.alpha)
        plt.scatter(blue_data[:, 0], blue_data[:, 1], 
                   color='blue', label=title_list[1], 
                   s=self.config.point_size, alpha=self.config.alpha)

        if vis_title:
            plt.xlabel(f'Latent Dimension {self.config.dimensions[0]}')
            plt.ylabel(f'Latent Dimension {self.config.dimensions[1]}')
            plt.title('Latent Space Visualization')
            plt.legend()
        else:
            plt.axis('off')

        # Save plot
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0 if not vis_title else None)
        plt.close()
        logger.info(f"Visualization saved to {save_path}")

def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file"""
    return OmegaConf.load(config_path)

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Initialize visualizer
    visualizer_config = LatentVisualizerConfig(
        vposer_ckpt=config.paths.vposer_ckpt,
        output_dir=config.paths.output_dir,
        device=config.device,
        point_size=config.visualization.point_size,
        alpha=config.visualization.alpha,
        overlap_threshold=config.visualization.overlap_threshold,
        dimensions=config.visualization.dimensions
    )
    visualizer = LatentVisualizer(visualizer_config)

    # Get file lists
    dataset1_files = glob(str(Path(config.paths.dataset1_path) / config.sampling.file_pattern))
    dataset2_files = glob(str(Path(config.paths.dataset2_path) / "*.npz"))
    
    # Sample files
    dataset1_sample = np.random.choice(dataset1_files, config.sampling.sample_size, replace=False)
    dataset2_sample = np.random.choice(dataset2_files, config.sampling.sample_size, replace=False)

    # Process datasets
    latent_codes_1 = visualizer.process_dataset(dataset1_sample)
    latent_codes_2 = visualizer.process_dataset(dataset2_sample)

    # Apply PCA
    reduced_codes_1, reduced_codes_2 = visualizer.apply_pca(
        latent_codes_1, 
        latent_codes_2, 
        config.pca.n_components
    )

    # Generate visualizations
    output_dir = Path(config.paths.output_dir)
    visualizer.visualize_latent_codes(
        reduced_codes_1,
        reduced_codes_2,
        save_path=str(output_dir / "latent_codes_dim_0_1.png"),
        vis_title=True
    )
    visualizer.visualize_latent_codes(
        reduced_codes_1,
        reduced_codes_2,
        save_path=str(output_dir / "latent_codes_dim_0_1_no_title.png"),
        vis_title=False
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Program failed with error: {e}", exc_info=True)