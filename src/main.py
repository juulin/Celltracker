import os
import tifffile as tiff
import pandas as pd
import numpy as np
from image_processor import ImageProcessor
from cell_tracker import CellTracker
from custom_statistics import Statistics
from data_export import export_to_csv

def main():
    # Get user input for the TIF stack file path
    tif_stack_path = input("Please enter the path to the TIF stack: ")
    
    if not os.path.exists(tif_stack_path):
        print("The specified file does not exist.")
        return

    # Initialize the image processor
    image_processor = ImageProcessor()
    
    # Load the TIF stack
    images = image_processor.load_tif_stack(tif_stack_path)
    
    # Check if the image has multiple channels
    segment_channel_idx = 0
    measure_channel_idx = 0
    
    if len(images.shape) >= 3:
        # Determine the channel dimension and number of channels
        if len(images.shape) == 3:
            # Could be either (frames, height, width) or (height, width, channels)
            if images.shape[2] <= 10:  # Likely (height, width, channels)
                num_possible_channels = images.shape[2]
                channel_dim = 2
            else:  # Likely (frames, height, width)
                num_possible_channels = 1
                channel_dim = None
        elif len(images.shape) == 4:
            # Could be (frames, channels, height, width) or (frames, height, width, channels)
            if images.shape[1] < images.shape[3]:
                num_possible_channels = images.shape[1]
                channel_dim = 1
            else:
                num_possible_channels = images.shape[3]
                channel_dim = 3
        else:
            # Complex case, assume last dimension might be channels
            num_possible_channels = images.shape[-1]
            channel_dim = len(images.shape) - 1
            
        if num_possible_channels > 1:
            print(f"Detected {num_possible_channels} possible channels in the image stack.")
            
            # Ask user which channel to use for nucleus segmentation
            while True:
                try:
                    segment_channel_idx = int(input(f"Which channel would you like to use for NUCLEUS SEGMENTATION? (0-{num_possible_channels-1}): "))
                    if 0 <= segment_channel_idx < num_possible_channels:
                        break
                    else:
                        print(f"Please enter a number between 0 and {num_possible_channels-1}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print(f"Using channel {segment_channel_idx} for nucleus segmentation")
            
            # Ask user which channel to use for intensity measurements
            while True:
                try:
                    measure_channel_idx = int(input(f"Which channel would you like to use for INTENSITY MEASUREMENTS? (0-{num_possible_channels-1}): "))
                    if 0 <= measure_channel_idx < num_possible_channels:
                        break
                    else:
                        print(f"Please enter a number between 0 and {num_possible_channels-1}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print(f"Using channel {measure_channel_idx} for intensity measurements")
            
            # Extract the selected channels
            if channel_dim == 1:
                # (frames, channels, height, width)
                segment_channel_images = images[:, segment_channel_idx, :, :]
                measure_channel_images = images[:, measure_channel_idx, :, :]
            elif channel_dim == 2:
                # (height, width, channels) - single frame
                segment_channel_images = images[:, :, segment_channel_idx]
                measure_channel_images = images[:, :, measure_channel_idx]
                # Add a frame dimension if needed
                if len(segment_channel_images.shape) == 2:
                    segment_channel_images = np.expand_dims(segment_channel_images, axis=0)
                    measure_channel_images = np.expand_dims(measure_channel_images, axis=0)
            elif channel_dim == 3:
                # (frames, height, width, channels)
                segment_channel_images = images[:, :, :, segment_channel_idx]
                measure_channel_images = images[:, :, :, measure_channel_idx]
            else:
                # Default case
                segment_channel_images = images.take(indices=segment_channel_idx, axis=channel_dim)
                measure_channel_images = images.take(indices=measure_channel_idx, axis=channel_dim)
                
            # Ensure the selected channel data is still properly shaped
            if len(segment_channel_images.shape) == 2:
                # Single frame, reshape to add frame dimension
                segment_channel_images = np.expand_dims(segment_channel_images, axis=0)
                measure_channel_images = np.expand_dims(measure_channel_images, axis=0)
                
            print(f"Segmentation channel data shape: {segment_channel_images.shape}")
            print(f"Measurement channel data shape: {measure_channel_images.shape}")
        else:
            print("Single channel detected, using same channel for segmentation and measurement.")
            segment_channel_images = images
            measure_channel_images = images
    else:
        print("Image format doesn't appear to have multiple channels, using same data for segmentation and measurement.")
        segment_channel_images = images
        measure_channel_images = images
    
    # Process images for segmentation
    processed_segment_images = image_processor.preprocess_images(segment_channel_images)
    
    # Initialize the cell tracker with channel information
    cell_tracker = CellTracker()
    
    # Track cells across frames using segmentation channel
    tracked_cells = cell_tracker.track_cells(processed_segment_images, measure_channel_images)
    
    # Ask if user wants to visualize the tracking
    visualize = input("Do you want to visualize the cell tracking? (y/n): ").lower().strip() == 'y'
    if visualize:
        output_format = input("Enter output format (mp4, gif, or directory for image sequence): ").strip()
        if output_format.lower() in ['mp4', 'gif']:
            output_path = input(f"Enter output path for the {output_format} file: ").strip()
            if not output_path.lower().endswith(f'.{output_format}'):
                output_path = f"{output_path}.{output_format}"
        else:
            output_path = input("Enter directory path for saving image sequence: ").strip()
        
        # Call visualization function
        cell_tracker.visualize_tracking(processed_segment_images, tracked_cells, output_path)
    
    # Initialize statistics calculator
    statistics = Statistics()
    
    # Calculate statistics for each tracked cell
    stats_data = statistics.calculate_statistics(tracked_cells)
    
    # Export the results to a CSV file
    output_csv_path = input("Please enter the output CSV file path (including filename.csv): ")
    
    # Ensure it has a .csv extension
    if not output_csv_path.lower().endswith('.csv'):
        output_csv_path = output_csv_path + '.csv'
    
    if export_to_csv(stats_data, output_csv_path):
        print(f"Data exported successfully to {output_csv_path}")
    else:
        print("Failed to export data.")

if __name__ == "__main__":
    main()