import numpy as np
import pandas as pd

class Statistics:
    def __init__(self):
        """
        Initialize the Statistics class without requiring tracked_cells in constructor
        """
        pass
        
    def calculate_statistics(self, tracked_cells):
        """
        Calculate statistics for each tracked cell across frames
        
        Parameters:
        -----------
        tracked_cells : dict
            Dictionary of tracked cells, where keys are cell IDs and values are 
            dictionaries containing cell data across frames
            
        Returns:
        --------
        stats_data : dict
            Dictionary containing statistics for each cell
        """
        stats_data = {}
        
        for cell_id, cell_data in tracked_cells.items():
            # Extract intensity values for nucleus and cytoplasm
            nucleus_intensities = cell_data.get('nucleus_intensities', [])
            cytoplasm_intensities = cell_data.get('cytoplasm_intensities', [])
            
            if not nucleus_intensities and not cytoplasm_intensities:
                print(f"Warning: No intensity data for cell {cell_id}")
                continue
                
            # Convert to numpy arrays for calculations
            nucleus_intensities = np.array(nucleus_intensities)
            nucleus_intensities = nucleus_intensities[~np.isnan(nucleus_intensities)]
            
            cytoplasm_intensities = np.array(cytoplasm_intensities)
            cytoplasm_intensities = cytoplasm_intensities[~np.isnan(cytoplasm_intensities)]
            
            # Check if we have data to analyze
            if len(nucleus_intensities) == 0 and len(cytoplasm_intensities) == 0:
                print(f"Warning: Only NaN values for cell {cell_id}")
                continue
            
            # Initialize stats dictionary for this cell
            cell_stats = {
                'frames_tracked': len(cell_data.get('frames', [])),
            }
            
            # Calculate nucleus statistics
            if len(nucleus_intensities) > 0:
                cell_stats.update({
                    'nucleus_mean_intensity': np.mean(nucleus_intensities),
                    'nucleus_median_intensity': np.median(nucleus_intensities),
                    'nucleus_std_dev': np.std(nucleus_intensities),
                    'nucleus_sem': np.std(nucleus_intensities) / np.sqrt(len(nucleus_intensities)),
                    'nucleus_min_intensity': np.min(nucleus_intensities),
                    'nucleus_max_intensity': np.max(nucleus_intensities),
                    'nucleus_intensity_values': nucleus_intensities.tolist()
                })
            
            # Calculate cytoplasm statistics
            if len(cytoplasm_intensities) > 0:
                cell_stats.update({
                    'cytoplasm_mean_intensity': np.mean(cytoplasm_intensities),
                    'cytoplasm_median_intensity': np.median(cytoplasm_intensities),
                    'cytoplasm_std_dev': np.std(cytoplasm_intensities),
                    'cytoplasm_sem': np.std(cytoplasm_intensities) / np.sqrt(len(cytoplasm_intensities)),
                    'cytoplasm_min_intensity': np.min(cytoplasm_intensities),
                    'cytoplasm_max_intensity': np.max(cytoplasm_intensities),
                    'cytoplasm_intensity_values': cytoplasm_intensities.tolist()
                })
                
            # Calculate nucleus-to-cytoplasm ratio if both are available
            if len(nucleus_intensities) > 0 and len(cytoplasm_intensities) > 0:
                n_c_ratios = nucleus_intensities / cytoplasm_intensities
                cell_stats.update({
                    'nucleus_cytoplasm_ratio_mean': np.mean(n_c_ratios),
                    'nucleus_cytoplasm_ratio_median': np.median(n_c_ratios),
                    'nucleus_cytoplasm_ratio_std': np.std(n_c_ratios)
                })
            
            stats_data[cell_id] = cell_stats
            
        return stats_data