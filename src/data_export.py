import os
import numpy as np
import pandas as pd

def export_to_csv(data, filename):
    """
    Export data to a CSV file
    
    Parameters:
    -----------
    data : dict
        Dictionary containing cell tracking data and statistics
    filename : str
        Path to output CSV file
    """
    # Check if path is a directory
    if os.path.isdir(filename):
        # If directory, use a default filename
        filename = os.path.join(filename, "cell_tracking_results.csv")
        print(f"Output path is a directory. Using default filename: {filename}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # If no file extension, add .csv
    if not os.path.splitext(filename)[1]:
        filename = filename + '.csv'
    
    print(f"Exporting data to: {filename}")
    
    # Convert the nested dictionary structure to a format suitable for pandas
    rows = []
    for cell_id, stats in data.items():
        # Basic row with cell ID
        row = {'cell_id': cell_id}
        
        # Add all statistics except array values
        for stat_name, stat_value in stats.items():
            if not stat_name.endswith('_values'):  # Skip arrays of values
                row[stat_name] = stat_value
        
        rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Create a per-frame CSV with more detailed measurements
    per_frame_rows = []
    
    # For each cell and frame
    for cell_id, stats in data.items():
        if 'nucleus_intensity_values' in stats and 'cytoplasm_intensity_values' in stats:
            nucleus_values = stats['nucleus_intensity_values']
            cytoplasm_values = stats['cytoplasm_intensity_values']
            
            # For each frame, calculate running statistics
            for frame_idx in range(min(len(nucleus_values), len(cytoplasm_values))):
                # Get values up to current frame for calculating running statistics
                nucleus_values_so_far = nucleus_values[:frame_idx+1]
                cytoplasm_values_so_far = cytoplasm_values[:frame_idx+1]
                
                # Create basic frame data
                frame_data = {
                    'cell_id': cell_id,
                    'frame': frame_idx,
                    'nucleus_intensity': nucleus_values[frame_idx],
                    'cytoplasm_intensity': cytoplasm_values[frame_idx],
                }
                
                # Calculate nucleus-to-cytoplasm ratio if possible
                if cytoplasm_values[frame_idx] > 0:
                    frame_data['nc_ratio'] = nucleus_values[frame_idx] / cytoplasm_values[frame_idx]
                else:
                    frame_data['nc_ratio'] = None
                
                # Add running statistics for nucleus
                if len(nucleus_values_so_far) > 0:
                    clean_nucleus_values = np.array([v for v in nucleus_values_so_far if v is not None and not np.isnan(v)])
                    if len(clean_nucleus_values) > 0:
                        frame_data.update({
                            'nucleus_mean_so_far': np.mean(clean_nucleus_values),
                            'nucleus_median_so_far': np.median(clean_nucleus_values),
                            'nucleus_min_so_far': np.min(clean_nucleus_values),
                            'nucleus_max_so_far': np.max(clean_nucleus_values),
                        })
                        if len(clean_nucleus_values) > 1:  # Need at least 2 points for std
                            frame_data['nucleus_std_so_far'] = np.std(clean_nucleus_values)
                            frame_data['nucleus_sem_so_far'] = np.std(clean_nucleus_values) / np.sqrt(len(clean_nucleus_values))
                
                # Add running statistics for cytoplasm
                if len(cytoplasm_values_so_far) > 0:
                    clean_cytoplasm_values = np.array([v for v in cytoplasm_values_so_far if v is not None and not np.isnan(v)])
                    if len(clean_cytoplasm_values) > 0:
                        frame_data.update({
                            'cytoplasm_mean_so_far': np.mean(clean_cytoplasm_values),
                            'cytoplasm_median_so_far': np.median(clean_cytoplasm_values),
                            'cytoplasm_min_so_far': np.min(clean_cytoplasm_values),
                            'cytoplasm_max_so_far': np.max(clean_cytoplasm_values),
                        })
                        if len(clean_cytoplasm_values) > 1:  # Need at least 2 points for std
                            frame_data['cytoplasm_std_so_far'] = np.std(clean_cytoplasm_values)
                            frame_data['cytoplasm_sem_so_far'] = np.std(clean_cytoplasm_values) / np.sqrt(len(clean_cytoplasm_values))
                
                # Calculate running N/C ratio statistics if both are available
                if len(clean_nucleus_values) > 0 and len(clean_cytoplasm_values) > 0:
                    # Calculate N/C ratio for each frame up to current
                    nc_ratios = []
                    for n_idx, c_idx in zip(range(len(nucleus_values_so_far)), range(len(cytoplasm_values_so_far))):
                        n_val = nucleus_values_so_far[n_idx]
                        c_val = cytoplasm_values_so_far[c_idx]
                        if n_val is not None and c_val is not None and c_val > 0 and not np.isnan(n_val) and not np.isnan(c_val):
                            nc_ratios.append(n_val / c_val)
                    
                    if nc_ratios:
                        nc_ratios = np.array(nc_ratios)
                        frame_data.update({
                            'nc_ratio_mean_so_far': np.mean(nc_ratios),
                            'nc_ratio_median_so_far': np.median(nc_ratios),
                            'nc_ratio_min_so_far': np.min(nc_ratios),
                            'nc_ratio_max_so_far': np.max(nc_ratios),
                        })
                        if len(nc_ratios) > 1:
                            frame_data['nc_ratio_std_so_far'] = np.std(nc_ratios)
                
                # Add the frame data to rows
                per_frame_rows.append(frame_data)
    
    # Create per-frame DataFrame
    df_per_frame = pd.DataFrame(per_frame_rows)
    per_frame_filename = filename.replace('.csv', '_per_frame.csv')
    
    try:
        # Export the DataFrames to CSV files
        df.to_csv(filename, index=False)
        print(f"Summary data exported to {filename}")
        
        df_per_frame.to_csv(per_frame_filename, index=False)
        print(f"Per-frame data exported to {per_frame_filename}")
        
        return True
    except Exception as e:
        print(f"Error exporting data: {e}")
        return False