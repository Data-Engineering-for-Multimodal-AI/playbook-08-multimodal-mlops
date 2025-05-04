import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple, Any
import json

from src.utils import get_pipeline_logger, get_param, load_schema

logger = get_pipeline_logger("video_features")


def extract_color_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Extract color histogram features from an image.
    
    Args:
        image: RGB image as numpy array
        bins: Number of histogram bins per channel
        
    Returns:
        Flattened histogram features
    """
    hist_features = []
    
    # Extract histogram for each color channel
    for i in range(3):  # RGB channels
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        # Normalize histogram
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.append(hist)
        
    return np.hstack(hist_features)


class FeatureExtractor:
    """Class for extracting features from images using pretrained CNN models."""
    
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True, device: str = None):
        """
        Initialize the feature extractor with the specified model.
        
        Args:
            model_name: Name of the model to use ('resnet18', 'resnet50', etc.)
            pretrained: Whether to use pretrained weights
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using {self.device} for feature extraction")
        
        # Load model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Remove the final fully connected layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Feature vector as numpy array
        """
        # Preprocess the image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
            
        # Convert to numpy and flatten
        features = features.cpu().numpy().flatten()
        
        return features


def process_video_features(video_metadata_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Process video frames to extract features.
    
    Args:
        video_metadata_path: Path to video metadata CSV
        output_path: Path to save extracted features
        
    Returns:
        DataFrame with extracted features
    """
    # Get parameters
    if output_path is None:
        output_path = os.path.join(get_param('features.output_path'), 'video')
    
    histogram_bins = get_param('features.video.histogram_bins')
    use_embeddings = get_param('features.video.use_embeddings')
    embedding_model = get_param('features.video.embedding_model')
    pretrained = get_param('features.video.pretrained')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load video metadata
    logger.info(f"Loading video metadata from {video_metadata_path}")
    video_df = pd.read_csv(video_metadata_path)
    
    # Initialize feature extractor if needed
    if use_embeddings:
        feature_extractor = FeatureExtractor(model_name=embedding_model, pretrained=pretrained)
    
    # Process each frame
    features_list = []
    
    for idx, row in video_df.iterrows():
        if idx % 10 == 0:
            logger.info(f"Processing frame {idx}/{len(video_df)}")
            
        try:
            # Load frame
            frame_path = row['ingested_path']
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Could not read frame: {frame_path}")
                continue
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract histogram features
            hist_features = extract_color_histogram(frame, bins=histogram_bins)
            
            # Extract CNN embeddings if requested
            if use_embeddings:
                embedding_features = feature_extractor.extract_features(frame)
            else:
                embedding_features = np.array([])
                
            # Create feature record
            feature_record = {
                'frame_id': row['frame_id'],
                'timestamp': row['timestamp'],
                'hist_features': hist_features.tolist(),
            }
            
            if use_embeddings:
                feature_record['embedding_features'] = embedding_features.tolist()
                
            features_list.append(feature_record)
            
        except Exception as e:
            logger.error(f"Error processing frame {idx}: {str(e)}")
    
    # Create features DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Save features
    features_path = os.path.join(output_path, "video_features.parquet")
    features_df.to_parquet(features_path, index=False)
    logger.info(f"Saved video features to {features_path}")
    
    # Save summary
    summary = {
        'total_frames': len(features_df),
        'histogram_features_dim': histogram_bins * 3,
        'embedding_features_dim': len(embedding_features) if use_embeddings else 0,
        'embedding_model': embedding_model if use_embeddings else None
    }
    
    summary_path = os.path.join(output_path, "video_features_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Feature extraction complete. Processed {len(features_df)} frames")
    
    return features_df


if __name__ == "__main__":
    video_metadata_path = os.path.join(get_param('ingest.video.output_path'), "video_metadata.csv")
    process_video_features(video_metadata_path) 