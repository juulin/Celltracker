import numpy as np
import cv2
from skimage import measure, segmentation, filters
from scipy import ndimage
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment

class CellTracker:
    def __init__(self):
        self.next_id = 1
        self.cells = {}
        self.max_distance = 50  # Maximum distance (in pixels) for tracking the same cell between frames
    
    def segment_cells(self, segment_image, measure_image=None, frame_idx=None):
        """
        Segment individual cells in an image, separating nucleus and cytoplasm
        
        Parameters:
        -----------
        segment_image : numpy.ndarray
            Image to use for nucleus segmentation
        measure_image : numpy.ndarray, optional
            Image to use for cytoplasm segmentation and intensity measurements
        frame_idx : int, optional
            Index of the current frame for storing masks
            
        Returns:
        --------
        cells : list
            List of dictionaries containing cell properties
        labeled_nuclei : numpy.ndarray
            Labeled image of nuclei
        """
        # If measurement image not provided, use segmentation image
        if measure_image is None:
            measure_image = segment_image
        
        # Check image dimensions and convert if necessary for segmentation
        print(f"Segmentation image shape: {segment_image.shape}")
        
        # If image has more than 2 dimensions, convert to 2D
        if len(segment_image.shape) > 2:
            print(f"Converting {len(segment_image.shape)}D segmentation image to 2D")
            if len(segment_image.shape) == 3:
                segment_image = segment_image[:, :, 0] if segment_image.shape[2] <= 3 else np.mean(segment_image, axis=2)
            else:
                segment_image = np.mean(segment_image, axis=tuple(range(len(segment_image.shape)-2)))
                
        # Do the same for measurement image
        print(f"Measurement image shape: {measure_image.shape}")
        if len(measure_image.shape) > 2:
            print(f"Converting {len(measure_image.shape)}D measurement image to 2D")
            if len(measure_image.shape) == 3:
                measure_image = measure_image[:, :, 0] if measure_image.shape[2] <= 3 else np.mean(measure_image, axis=2)
            else:
                measure_image = np.mean(measure_image, axis=tuple(range(len(measure_image.shape)-2)))
                
        print(f"Processed segmentation image shape: {segment_image.shape}")
        print(f"Processed measurement image shape: {measure_image.shape}")
        
        # Step 1: Segment nuclei using segmentation image
        nucleus_threshold = filters.threshold_otsu(segment_image)
        nucleus_mask = segment_image > nucleus_threshold
        
        # Clean up nuclei mask
        nucleus_mask = ndimage.binary_opening(nucleus_mask, structure=np.ones((3, 3)))
        nucleus_mask = ndimage.binary_closing(nucleus_mask, structure=np.ones((3, 3)))
        
        # Step 2: Create cytoplasm mask from measurement channel
        # First create a mask for potential cytoplasm in measurement image
        cytoplasm_threshold = filters.threshold_otsu(measure_image) * 0.7  # Lower threshold for cytoplasm
        cytoplasm_potential_mask = measure_image > cytoplasm_threshold
        
        # Clean up cytoplasm mask
        cytoplasm_potential_mask = ndimage.binary_opening(cytoplasm_potential_mask, structure=np.ones((3, 3)))
        cytoplasm_potential_mask = ndimage.binary_closing(cytoplasm_potential_mask, structure=np.ones((5, 5)))
        
        # Step 3: Label nuclei
        labeled_nuclei, num_nuclei = ndimage.label(nucleus_mask)
        
        # Store masks for visualization if frame_idx is provided
        if frame_idx is not None:
            if not hasattr(self, 'frame_masks'):
                self.frame_masks = {}
            self.frame_masks[frame_idx] = {}
        
        # Extract properties for each nucleus using the segmentation image for shape and measurement image for intensity
        region_props = measure.regionprops(labeled_nuclei, segment_image)
        measure_props = measure.regionprops(labeled_nuclei, measure_image)
        
        cells = []
        for i, (prop, measure_prop) in enumerate(zip(region_props, measure_props)):
            # Filter out very small objects that might be noise
            if prop.area < 10:
                continue
                
            # Create a mask for this nucleus
            nucleus_id = prop.label
            nucleus_region = (labeled_nuclei == nucleus_id)
            
            # As requested: for cytoplasm, use both:
            # 1. A dilated nucleus mask from the segmentation channel
            # 2. The cytoplasm mask from the measurement channel
            dilated_nucleus = ndimage.binary_dilation(nucleus_region, structure=np.ones((7, 7)))
            
            # Create cell-specific cytoplasm by:
            # 1. Taking the intersection of dilated nucleus and cytoplasm potential mask
            # 2. Excluding any pixels that are part of the nucleus
            cell_cytoplasm = dilated_nucleus & cytoplasm_potential_mask & ~nucleus_mask
            
            # Calculate nucleus properties using the measurement image
            nucleus_area = prop.area
            nucleus_intensity = measure_prop.mean_intensity  # Use measurement image for intensity
            
            # Calculate cytoplasm properties
            cytoplasm_area = np.sum(cell_cytoplasm)
            cytoplasm_intensity = np.mean(measure_image[cell_cytoplasm]) if cytoplasm_area > 0 else 0
            
            # Store masks for this cell
            if frame_idx is not None:
                self.frame_masks[frame_idx][nucleus_id] = {
                    'nucleus_mask': nucleus_region,
                    'cytoplasm_mask': cell_cytoplasm
                }
            
            cells.append({
                'centroid': prop.centroid,
                'area': nucleus_area,
                'nucleus_intensity': nucleus_intensity,
                'cytoplasm_intensity': cytoplasm_intensity,
                'cytoplasm_area': cytoplasm_area,
                'bbox': prop.bbox,
                'label': nucleus_id,
                'nucleus_mask': nucleus_region,
                'cytoplasm_mask': cell_cytoplasm
            })
        
        print(f"Segmented {len(cells)} cells")
        return cells, labeled_nuclei
    
    def track_cells(self, segment_images, measure_images=None):
        """
        Track cells across multiple frames
        
        Parameters:
        -----------
        segment_images : numpy.ndarray
            Stack of images to use for segmentation and tracking
        measure_images : numpy.ndarray, optional
            Stack of images to use for intensity measurements (if None, uses segment_images)
            
        Returns:
        --------
        cells : dict
            Dictionary of tracked cells
        """
        if segment_images is None or len(segment_images) == 0:
            print("Error: No images provided for cell tracking")
            return {}
            
        # If measurement images not provided, use segmentation images
        if measure_images is None:
            measure_images = segment_images
        
        # Ensure both image stacks have the same number of frames
        if len(segment_images) != len(measure_images):
            print(f"Warning: Segment images ({len(segment_images)} frames) and measure images ({len(measure_images)} frames) have different frame counts")
            min_frames = min(len(segment_images), len(measure_images))
            segment_images = segment_images[:min_frames]
            measure_images = measure_images[:min_frames]
        
        print(f"Starting cell tracking on {len(segment_images)} frames")
        
        # Reset tracking data and masks
        self.next_id = 1
        self.cells = {}
        self.frame_masks = {}
        
        # Process the first frame
        first_frame_cells, first_frame_labeled = self.segment_cells(segment_images[0], measure_images[0], frame_idx=0)
        
        # Assign initial IDs to cells
        for cell in first_frame_cells:
            cell_id = self.next_id
            self.next_id += 1
            
            self.cells[cell_id] = {
                'frames': [0],
                'centroids': [cell['centroid']],
                'areas': [cell['area']],
                'nucleus_intensities': [cell['nucleus_intensity']],
                'cytoplasm_intensities': [cell['cytoplasm_intensity']],
                'cytoplasm_areas': [cell['cytoplasm_area']],
                'bboxes': [cell['bbox']],
                'labels': [cell['label']]
            }
        
        # Process subsequent frames
        for frame_idx in range(1, len(segment_images)):
            print(f"Processing frame {frame_idx}/{len(segment_images)-1}")
            current_frame_cells, current_frame_labeled = self.segment_cells(segment_images[frame_idx], 
                                                                           measure_images[frame_idx], 
                                                                           frame_idx=frame_idx)
            
            # Skip if no cells found in current frame
            if not current_frame_cells:
                print(f"No cells found in frame {frame_idx}")
                continue
                
            # Create cost matrix for assignment
            cost_matrix = np.zeros((len(self.cells), len(current_frame_cells)))
            
            # Calculate distances between cells in previous and current frames
            for i, (cell_id, cell_data) in enumerate(self.cells.items()):
                # Get the most recent centroid for the cell
                last_known_centroid = cell_data['centroids'][-1]
                
                for j, current_cell in enumerate(current_frame_cells):
                    # Calculate Euclidean distance between centroids
                    distance = np.sqrt(
                        (last_known_centroid[0] - current_cell['centroid'][0])**2 +
                        (last_known_centroid[1] - current_cell['centroid'][1])**2
                    )
                    
                    # Assign infinite cost if distance is too large
                    if distance > self.max_distance:
                        cost_matrix[i, j] = np.inf
                    else:
                        cost_matrix[i, j] = distance
            
            # Use Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Update tracked cells and assign new cells
            assigned_current_cells = set()
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] != np.inf:
                    # Get cell ID for this assignment
                    cell_id = list(self.cells.keys())[i]
                    
                    # Update cell data with new information
                    self.cells[cell_id]['frames'].append(frame_idx)
                    self.cells[cell_id]['centroids'].append(current_frame_cells[j]['centroid'])
                    self.cells[cell_id]['areas'].append(current_frame_cells[j]['area'])
                    self.cells[cell_id]['nucleus_intensities'].append(current_frame_cells[j]['nucleus_intensity'])
                    self.cells[cell_id]['cytoplasm_intensities'].append(current_frame_cells[j]['cytoplasm_intensity'])
                    self.cells[cell_id]['cytoplasm_areas'].append(current_frame_cells[j]['cytoplasm_area'])
                    self.cells[cell_id]['bboxes'].append(current_frame_cells[j]['bbox'])
                    self.cells[cell_id]['labels'].append(current_frame_cells[j]['label'])
                    
                    assigned_current_cells.add(j)
            
            # Create new entries for unassigned cells
            for j, cell in enumerate(current_frame_cells):
                if j not in assigned_current_cells:
                    cell_id = self.next_id
                    self.next_id += 1
                    
                    # Create new entry with placeholder None values for previous frames
                    self.cells[cell_id] = {
                        'frames': [frame_idx],
                        'centroids': [cell['centroid']],
                        'areas': [cell['area']],
                        'nucleus_intensities': [cell['nucleus_intensity']],
                        'cytoplasm_intensities': [cell['cytoplasm_intensity']],
                        'cytoplasm_areas': [cell['cytoplasm_area']],
                        'bboxes': [cell['bbox']],
                        'labels': [cell['label']]
                    }
        
        print(f"Finished tracking. Found {len(self.cells)} cells across {len(segment_images)} frames")
        return self.cells

    def get_tracked_cells(self):
        return self.cells

    def calculate_statistics(self):
        # Implement statistics calculation for tracked cells
        pass

    def visualize_tracking(self, images, tracked_cells, output_path=None):
        """
        Visualize cell tracking across frames with original images and segmentation masks side by side.
        
        Parameters:
        -----------
        images : numpy.ndarray
            Original image stack
        tracked_cells : dict
            Dictionary of tracked cells from track_cells method
        output_path : str, optional
            Path to save the visualization (video or directory for images)
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Circle
        from matplotlib.colors import hsv_to_rgb
        import numpy as np
        import os
        
        # Generate unique colors for each cell
        num_cells = len(tracked_cells)
        colors = {}
        for i, cell_id in enumerate(tracked_cells.keys()):
            # Create distinct colors using HSV (hue, saturation, value)
            hue = i / max(num_cells, 1)
            colors[cell_id] = hsv_to_rgb(np.array([hue, 1.0, 1.0]))
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Number of frames
        num_frames = len(images)
        
        # Get the shape for the mask image
        if len(images[0].shape) == 2:
            mask_shape = images[0].shape
        else:
            # Handle multi-dimensional first frame
            mask_shape = images[0].shape[:2] if len(images[0].shape) >= 2 else (100, 100)
        
        # Create empty mask image for visualization
        composite_mask = np.zeros((*mask_shape, 3))
        
        # Function to update the plot for animation
        def update_frame(frame_idx):
            nonlocal composite_mask  # Add this line to access the outer variable
            
            # Clear both axes
            ax1.clear()
            ax2.clear()
            
            # Panel 1: Original image with tracking overlays
            ax1.imshow(images[frame_idx], cmap='gray')
            ax1.set_title(f'Original Image - Frame {frame_idx+1}/{num_frames}')
            ax1.axis('off')
            
            # Reset the composite mask
            composite_mask.fill(0)
            
            # Add cell markers, labels, and create masks for visualization
            for cell_id, cell_data in tracked_cells.items():
                frames = cell_data.get('frames', [])
                if frame_idx in frames:
                    idx = frames.index(frame_idx)
                    centroid = cell_data['centroids'][idx]
                    
                    if centroid is not None:
                        y, x = centroid
                        
                        # Panel 1: Draw circles and labels on original image
                        nucleus_circle = Circle((x, y), radius=5, color=colors[cell_id], 
                                              fill=False, linewidth=2)
                        ax1.add_patch(nucleus_circle)
                        
                        cytoplasm_circle = Circle((x, y), radius=10, color=colors[cell_id], 
                                               fill=False, linewidth=1, linestyle='--')
                        ax1.add_patch(cytoplasm_circle)
                        
                        # Add cell ID text
                        ax1.text(x+12, y+12, str(cell_id), color=colors[cell_id], 
                                fontsize=12, fontweight='bold')
                        
                        # Panel 1: Draw trajectory
                        trajectory_points = []
                        for f_idx, f in enumerate(frames):
                            if f <= frame_idx and f_idx < len(cell_data['centroids']):
                                traj_point = cell_data['centroids'][f_idx]
                                if traj_point is not None:
                                    trajectory_points.append(traj_point)
                        
                        if trajectory_points:
                            traj_y, traj_x = zip(*trajectory_points)
                            ax1.plot(traj_x, traj_y, '-', color=colors[cell_id], linewidth=1, alpha=0.7)
                        
                        # Panel 2: Add to the mask visualization
                        label_value = cell_data['labels'][idx]
                        
                        # Check if we have stored masks
                        if hasattr(self, 'frame_masks') and frame_idx in self.frame_masks:
                            if label_value in self.frame_masks[frame_idx]:
                                nucleus_mask = self.frame_masks[frame_idx][label_value].get('nucleus_mask')
                                cytoplasm_mask = self.frame_masks[frame_idx][label_value].get('cytoplasm_mask')
                                
                                if nucleus_mask is not None:
                                    # Add nucleus mask in cell color (more intense)
                                    for c in range(3):
                                        composite_mask[:,:,c] += nucleus_mask * colors[cell_id][c]
                                
                                if cytoplasm_mask is not None:
                                    # Add cytoplasm mask in cell color (less intense)
                                    for c in range(3):
                                        composite_mask[:,:,c] += cytoplasm_mask * colors[cell_id][c] * 0.5
                            else:
                                print(f"Warning: No mask found for label {label_value} in frame {frame_idx}")
                        else:
                            # If no stored masks, create circular approximations
                            y_idx, x_idx = np.ogrid[:mask_shape[0], :mask_shape[1]]
                            # Create nucleus mask (circle)
                            nucleus_dist = np.sqrt((y_idx - y)**2 + (x_idx - x)**2)
                            nucleus_mask = nucleus_dist <= 5
                            
                            # Create cytoplasm mask (ring)
                            cytoplasm_mask = (nucleus_dist > 5) & (nucleus_dist <= 10)
                            
                            # Add to composite mask
                            for c in range(3):
                                composite_mask[:,:,c] += nucleus_mask * colors[cell_id][c]
                                composite_mask[:,:,c] += cytoplasm_mask * colors[cell_id][c] * 0.5
            
            # Clip mask values to valid range [0, 1]
            composite_mask = np.clip(composite_mask, 0, 1)
            
            # Panel 2: Display the composite mask
            ax2.imshow(composite_mask)
            ax2.set_title(f'Segmentation Masks - Frame {frame_idx+1}/{num_frames}')
            ax2.axis('off')
            
            return ax1, ax2
        
        # Create animation
        ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, 
                                      interval=200, blit=False)
        
        # Save animation or display it
        if output_path:
            if output_path.endswith('.mp4'):
                # Save as MP4 video
                writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Cell Tracker'), 
                                               bitrate=1800)
                ani.save(output_path, writer=writer)
                print(f"Animation saved as {output_path}")
            elif output_path.endswith('.gif'):
                # Save as GIF
                ani.save(output_path, writer='pillow', fps=5)
                print(f"Animation saved as {output_path}")
            else:
                # Save as image sequence
                os.makedirs(output_path, exist_ok=True)
                for frame_idx in range(num_frames):
                    update_frame(frame_idx)
                    plt.savefig(os.path.join(output_path, f'frame_{frame_idx:03d}.png'))
                print(f"Image sequence saved in {output_path}")
        else:
            # Display the animation
            plt.tight_layout()
            plt.show()
        
        plt.close()