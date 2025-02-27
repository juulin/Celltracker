import os
import glob
import tifffile as tiff
from skimage import io, filters
import numpy as np

class ImageProcessor:
    def __init__(self):
        # No file_path required in constructor
        self.images = None
        
    def load_tif_stack(self, file_path):
        """
        Load TIF stack from a file path. If path is a directory, 
        load the first TIF file found in the directory.
        """
        if os.path.isdir(file_path):
            # Path is a directory, find TIF files
            tif_files = glob.glob(os.path.join(file_path, '*.tif')) + \
                        glob.glob(os.path.join(file_path, '*.tiff'))
            
            if not tif_files:
                raise ValueError(f"No TIF files found in directory: {file_path}")
            
            print(f"Found {len(tif_files)} TIF files. Loading the first one: {tif_files[0]}")
            file_path = tif_files[0]
        
        # Now file_path should be a file, not a directory
        if not os.path.isfile(file_path):
            raise ValueError(f"File does not exist: {file_path}")
            
        print(f"Loading TIF stack from: {file_path}")
        try:
            # Try using tifffile first (better for multi-dimensional TIFFs)
            self.images = tiff.imread(file_path)
            print(f"Stack loaded with shape: {self.images.shape}")
            return self.images
        except Exception as e:
            print(f"Error with tifffile, trying skimage: {e}")
            try:
                # Fallback to skimage
                self.images = io.imread(file_path)
                print(f"Stack loaded with shape: {self.images.shape}")
                return self.images
            except Exception as e:
                raise IOError(f"Failed to load TIF file: {e}")

    def preprocess_images(self, images=None):
        """
        Preprocess images (e.g., normalization, filtering)
        """
        if images is not None:
            self.images = images
            
        if self.images is None:
            raise ValueError("No images loaded. Call load_tif_stack first.")
            
        # Simple Gaussian filtering as an example
        processed_images = []
        for i in range(len(self.images)):
            img = self.images[i]
            # Apply Gaussian filter
            filtered = filters.gaussian(img, sigma=1)
            processed_images.append(filtered)
            
        self.processed_images = np.array(processed_images)
        return self.processed_images

    def extract_features(self, images=None):
        """
        Extract relevant features from each frame
        """
        if images is not None:
            self.processed_images = images
            
        if not hasattr(self, 'processed_images') or self.processed_images is None:
            raise ValueError("No processed images. Call preprocess_images first.")
            
        features = []
        for image in self.processed_images:
            # Example feature extraction (mean intensity)
            mean_intensity = image.mean()
            features.append(mean_intensity)
        return features