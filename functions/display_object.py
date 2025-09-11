#!/usr/bin/env python3
"""
Fast PLY visualization and image capture for Tiled upload
Optimized for speed with minimal quality loss
"""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
from pathlib import Path

def fast_capture_ply_views(ply_file, output_dir="fast_ply_views", num_views=8):
    """
    Fast capture of PLY file from multiple angles
    """
    
    # Clean up any existing images first
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"Cleaned up existing directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading PLY file: {ply_file}")
    mesh = trimesh.load(ply_file)
    
    # Center the mesh
    mesh.vertices -= mesh.centroid
    
    # Scale to reasonable size
    scale = 2.0 / mesh.extents.max()
    mesh.vertices *= scale
    
    # Apply transformations like in testing_trimesh.py
    # Move object down and flip it upside down
    move_down = trimesh.transformations.translation_matrix([0.2, 0.8, 0.4])
    
    # Rotate to face upwards (rotate 90 degrees around X-axis)
    face_upwards = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, -1])
    
    # Apply both transformations
    combined_transform = trimesh.transformations.concatenate_matrices(move_down, face_upwards)
    mesh.apply_transform(combined_transform)
    
    # Re-center the mesh after transformations
    mesh.vertices -= mesh.centroid
    
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Set up fast matplotlib rendering
    plt.ioff()  # Turn off interactive mode for speed
    
    # Pre-calculate rotation angles
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    
    print(f"Capturing {num_views} views...")
    
    for i, angle in enumerate(angles):
        print(f"   View {i+1}/{num_views} (angle: {np.degrees(angle):.1f}°)")
        
        # Create rotated mesh (rotate around X-axis)
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(rotation_matrix)
        
        # Create figure with better size for quality
        fig = plt.figure(figsize=(10, 10), dpi=150, facecolor='black')  # Larger, higher DPI
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')  # Black background for the plot
        
        # Fast rendering settings
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        
        # Disable axes for speed
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.axis('off')
        
        # Set camera position (looking down at 90 degrees - straight down)
        ax.view_init(elev=90, azim=0)
        
        # Plot mesh with completely solid rendering
        ax.plot_trisurf(
            rotated_mesh.vertices[:, 0],
            rotated_mesh.vertices[:, 1], 
            rotated_mesh.vertices[:, 2],
            triangles=rotated_mesh.faces,
            color='white',  # Use white for maximum opacity
            alpha=1.0,  # Completely solid (no transparency)
            edgecolor='none',  # No edges for speed
            shade=True,
            antialiased=False,  # Disable antialiasing for solid appearance
            linewidth=0.0  # Ensure no edge lines
        )
        
        # Save with minimal quality for speed
        output_path = os.path.join(output_dir, f"view_{i:02d}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150, facecolor='black')
        plt.close(fig)  # Close immediately to free memory
        
        # Convert RGBA to RGB and then to grayscale for Tiled compatibility
        from PIL import Image
        img = Image.open(output_path)
        if img.mode == 'RGBA':
            # Convert RGBA to RGB by compositing on black background
            rgb_img = Image.new('RGB', img.size, (0, 0, 0))  # Black background
            rgb_img.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            rgb_img.save(output_path, format='PNG')
            print(f"   Converted RGBA to RGB: view_{i:02d}.png")
        
        # Convert to grayscale
        img = Image.open(output_path)
        grayscale_img = img.convert('L')
        grayscale_img.save(output_path, format='PNG')
        print(f"   Converted to grayscale: view_{i:02d}.png")
    
    print(f"Captured {num_views} views in {output_dir}/")
    return output_dir

def ultra_fast_capture(ply_file, output_dir="ultra_fast_views", num_views=4):
    """
    Ultra-fast capture with minimal views and quality
    """
    
    # Clean up any existing images first
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f" Cleaned up existing directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Ultra-fast loading: {ply_file}")
    mesh = trimesh.load(ply_file)
    
    # Quick centering and scaling
    mesh.vertices -= mesh.centroid
    mesh.vertices *= 2.0 / mesh.extents.max()
    
    # Apply transformations like in testing_trimesh.py
    # Move object down and flip it upside down
    move_down = trimesh.transformations.translation_matrix([0, 0.8, 0])
    
    # Rotate to face upwards (rotate 90 degrees around X-axis)
    face_upwards = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 0])
    
    # Apply both transformations
    combined_transform = trimesh.transformations.concatenate_matrices(move_down, face_upwards)
    mesh.apply_transform(combined_transform)
    
    # Re-center the mesh after transformations
    mesh.vertices -= mesh.centroid
    
    # Even fewer views for speed
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]  # 4 cardinal directions
    
    print(f"Capturing {len(angles)} views (ultra-fast mode)...")
    
    for i, angle in enumerate(angles):
        print(f"   View {i+1}/{len(angles)}")
        
        # Rotate mesh (rotate around X-axis)
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(rotation_matrix)
        
        # Better figure setup for quality
        fig = plt.figure(figsize=(8, 8), dpi=120, facecolor='black')  # Larger, higher DPI
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')  # Black background for the plot
        
        # Set bounds
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.axis('off')
        
        # Fixed camera angle (looking down at 90 degrees - straight down)
        ax.view_init(elev=90, azim=0)
        
        # Plot with completely solid rendering
        ax.plot_trisurf(
            rotated_mesh.vertices[:, 0],
            rotated_mesh.vertices[:, 1], 
            rotated_mesh.vertices[:, 2],
            triangles=rotated_mesh.faces,
            color='white',  # Use white for maximum opacity
            alpha=1.0,  # Completely solid (no transparency)
            edgecolor='none',
            antialiased=False,  # Disable antialiasing for solid appearance
            linewidth=0.0  # Ensure no edge lines
        )
        
        # Save with low quality
        output_path = os.path.join(output_dir, f"view_{i:02d}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=120, facecolor='black')
        plt.close(fig)
        
        # Convert RGBA to RGB and then to grayscale for Tiled compatibility
        from PIL import Image
        img = Image.open(output_path)
        if img.mode == 'RGBA':
            # Convert RGBA to RGB by compositing on black background
            rgb_img = Image.new('RGB', img.size, (0, 0, 0))  # Black background
            rgb_img.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            rgb_img.save(output_path, format='PNG')
            print(f"Converted RGBA to RGB: view_{i:02d}.png")
        
        # Convert to grayscale
        img = Image.open(output_path)
        grayscale_img = img.convert('L')
        grayscale_img.save(output_path, format='PNG')
        print(f"Converted to grayscale: view_{i:02d}.png")
    
    print(f"Ultra-fast capture complete: {output_dir}/")
    return output_dir

def batch_upload_to_tiled(image_dir, tiled_uri="http://localhost:8000", api_key="API_KEy"):
    """
    Upload images as a 3D image stack to Tiled
    """
    try:
        from tiled.client import from_uri
        from PIL import Image
        import numpy as np
        
        print(f"Uploading images from {image_dir} to Tiled as image stack...")
        
        # Connect to Tiled
        client = from_uri(tiled_uri, api_key=api_key)
        
        # Get all PNG files and sort them
        image_files = sorted(list(Path(image_dir).glob("*.png")))
        print(f"Found {len(image_files)} images to stack")
        
        if not image_files:
            print("No images found to upload")
            return None
        
        # Load all images and stack them
        image_arrays = []
        for i, image_file in enumerate(image_files):
            print(f"Loading {image_file.name}...")
            
            # Load image using PIL
            pil_img = Image.open(str(image_file))
            
            # Keep grayscale images as grayscale (mode 'L')
            if pil_img.mode not in ['L', 'RGB']:
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                else:
                    pil_img = pil_img.convert('L')
            
            # Convert to numpy array
            img_array = np.array(pil_img)
            image_arrays.append(img_array)
        
        # Stack all images into a 3D array
        print("Creating 3D image stack...")
        image_stack = np.stack(image_arrays, axis=0)
        print(f"Stack shape: {image_stack.shape}")
        print(f"Stack dtype: {image_stack.dtype}")
        
        # Upload the 3D stack to Tiled
        print("Uploading image stack to Tiled...")
        uploaded_array = client.write_array(
            image_stack,
            metadata={
                "description": "PLY file visualization views as image stack",
                "num_views": len(image_files),
                "shape": image_stack.shape,
                "dtype": str(image_stack.dtype),
                "source": "fast_ply_capture",
                "filenames": [f.name for f in image_files]
            }
        )
        
        print(f"Successfully uploaded image stack to Tiled!")
        print(f"Stack name: {uploaded_array}")
        print(f"Dimensions: {image_stack.shape[0]} views × {image_stack.shape[1]} × {image_stack.shape[2]}")
        
        # Extract the UUID from the uploaded array
        try:
            if hasattr(uploaded_array, 'uri'):
                array_uuid = uploaded_array.uri.split('/')[-1]
            else:
                # Fallback to string representation
                array_uuid = str(uploaded_array)
        except Exception as e:
            print(f"Error extracting UUID: {e}")
            array_uuid = str(uploaded_array)
        
        print(f"Tiled UUID: {array_uuid}")
        
        return uploaded_array, array_uuid
        
    except ImportError:
        print("Tiled not available - skipping upload")
        return None, None
    except Exception as e:
        print(f"Upload failed: {e}")
        return None, None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 fast_ply_capture.py <ply_file> [mode] [num_views]")
        print("Modes: fast (default), ultra_fast")
        print("Example: python3 fast_ply_capture.py scene_texture.ply ultra_fast 4")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "fast"
    num_views = int(sys.argv[3]) if len(sys.argv) > 3 else (4 if mode == "ultra_fast" else 8)
    
    if not os.path.exists(ply_file):
        print(f"PLY file not found: {ply_file}")
        sys.exit(1)
    
    print(f"Mode: {mode}, Views: {num_views}")
    
    if mode == "ultra_fast":
        output_dir = ultra_fast_capture(ply_file, num_views=num_views)
    else:
        output_dir = fast_capture_ply_views(ply_file, num_views=num_views)
    
    # Try to upload to Tiled
    uploaded_array, array_uuid = batch_upload_to_tiled(output_dir)
    
    if array_uuid:
        print(f"Complete! Images saved in: {output_dir}/")
        print(f"Tiled UUID for metadata: {array_uuid}")
    else:
        print(f"Complete! Images saved in: {output_dir}/ (Tiled upload failed)")

	
