from ophyd import EpicsMotor, EpicsSignal
from ophyd.areadetector.plugins import PluginBase
from ophyd.areadetector import AreaDetector, ADComponent, ImagePlugin, TIFFPlugin
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.cam import CamBase
from bluesky import RunEngine
from databroker import Broker, temp
from bluesky.plans import scan, count
import bluesky.plan_stubs as bps
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import time, sys, os, subprocess
from collections import defaultdict

# Manual Tiled writing imports (similar to store.txt approach)
try:
    from tiled.client import from_uri
    TILED_AVAILABLE = True
except ImportError:
    TILED_AVAILABLE = False
    print("Warning: Tiled client not available, running without Tiled integration")
class PvaPlugin(PluginBase):
    _suffix = 'Pva1:'
    _plugin_type = 'NDPluginPva'
    _default_read_attrs = ['enable']
    _default_configuration_attrs = ['enable']

    array_callbacks = ADComponent(EpicsSignal, 'ArrayCallbacks')

# Create RunEngine
RE = RunEngine({})

# Set up manual Tiled connection (similar to store.txt approach)
tiled_client = None
if TILED_AVAILABLE:
    try:
        tiled_uri = "http://localhost:8000"
        tiled_api_key = "API_KEY"
        
        tiled_client = from_uri(tiled_uri, api_key=tiled_api_key)
        print(f"✓ Tiled client connected to {tiled_uri}")
    except Exception as e:
        print(f" Failed to connect to Tiled: {e}")
        TILED_AVAILABLE = False
        tiled_client = None

# Define the motor
motor = EpicsMotor('DMC01:A', name='motor')

# Define the camera deviceca
class MyCamera(AreaDetector):
    cam = ADComponent(AreaDetectorCam, 'cam1:') #Fixed the single camera issue?
    image = ADComponent(ImagePlugin, 'image1:')
    tiff = ADComponent(TIFFPlugin, 'TIFF1:')
    pva = ADComponent(PvaPlugin, 'Pva1:')

# Instantiate the camera
camera = MyCamera('13ARV1:', name='camera')
camera.wait_for_connection()

#CAM OPTIONS
camera.stage_sigs[camera.cam.acquire] = 0 
camera.stage_sigs[camera.cam.image_mode] = 0 # single multiple continuous
camera.stage_sigs[camera.cam.trigger_mode] = 0 # internal external

#IMAGE OPTIONS
camera.stage_sigs[camera.image.enable] = 1 # pva plugin
camera.stage_sigs[camera.image.queue_size] = 2000

#TIFF OPTIONS
camera.stage_sigs[camera.tiff.enable] = 1
camera.stage_sigs[camera.tiff.auto_save] = 1
camera.stage_sigs[camera.tiff.file_write_mode] = 0  # Or 'Single' works too
camera.stage_sigs[camera.tiff.nd_array_port] = 'SP1'  
camera.stage_sigs[camera.tiff.auto_increment] = 1       #Doesn't work, must be ignored

#PVA OPTIONS
camera.stage_sigs[camera.pva.enable] = 1
camera.stage_sigs[camera.pva.blocking_callbacks] = 'No'
camera.stage_sigs[camera.pva.queue_size] = 2000  # or higher
camera.stage_sigs[camera.pva.nd_array_port] = 'SP1' 
camera.stage_sigs[camera.pva.array_callbacks] = 0  # disable during scan

def wait_for_file(filepath, timeout=5.0, poll_interval=0.1):
    """Wait until a file appears on disk, or timeout."""
    start = time.time()
    while not os.path.exists(filepath):
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for file: {filepath}")
        time.sleep(poll_interval)

def scan_with_saves(start_pos, end_pos, num_points):
    #Requirements for image capturing
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    callbacks_signal = EpicsSignal('13ARV1:image1:EnableCallbacks', name='callbacks_signal')
    acquire_signal = EpicsSignal('13ARV1:cam1:Acquire', name='acquire_signal')

    yield from bps.mv(callbacks_signal, 0)
    max_retries = 50
    positions = np.linspace(start_pos, end_pos, num_points)
    yield from bps.open_run()
    camera.cam.array_callbacks.put(0, wait=True)

    print("\n--- Staging camera ---")
    yield from bps.stage(camera)

    current_number = camera.tiff.file_number.get()

    for i, pos in enumerate(positions):
        print(f"\nMoving to pos={pos}")
        yield from bps.mv(motor, pos / 2.8125)
        yield from bps.sleep(2.0) 
        yield from bps.mv(acquire_signal, 0)  # Triggers a single image

        #for img_idx in range(NUM_IMAGES_PER_POS):
        filename = f'scan_{timestamp}_pos_{i}_shot_angle_{pos:.2f}'           
        current_number += 1
        filepath = os.path.join(save_dir, f"{filename}_{current_number}.tiff")

        yield from bps.mv(camera.tiff.file_name, filename)
        yield from bps.mv(camera.tiff.file_number, current_number)

        for attempt in range(1, max_retries + 1):

            try:
                print(f"[Attempt {attempt}] Capturing → {filepath}")
                yield from bps.mv(acquire_signal, 1)  # Triggers a single image
                yield from bps.sleep(1)

                # Wait for file to appear
                wait_for_file(filepath, timeout=5.0)

                print(f"✓ Image saved at {filepath}")
                break  # Exit retry loop if successful

            except TimeoutError:
                print(f"--Timeout waiting for image at {filepath}")
                if attempt == max_retries:
                    print(f"--Failed after {max_retries} attempts, skipping position {pos}")
                else:
                    print("↻ Retrying acquisition...")
                    yield from bps.mv(acquire_signal, 0)  # Triggers a single image
                    yield from bps.sleep(0.5)
    
    print("\n--- Unstaging camera ---")
    yield from bps.unstage(camera)

    yield from bps.mv(motor, 0.0)
    yield from bps.close_run()

def cropImages(inputDir):
    crop_box = (800, 800, 1600, 1500)
    output_dir = inputDir.replace('raw_images/', 'images/')

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(inputDir):
        if filename.endswith('.tiff'):
            image_path = os.path.join(inputDir, filename)
            img = Image.open(image_path)
            cropped = img.crop(crop_box)
            cropped.save(os.path.join(output_dir, filename))

def convert_image_format(image_dir: str, output_image_dir: str):
    os.makedirs(output_image_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(".tiff") or filename.lower().endswith(".tif"):
            tiff_path = os.path.join(image_dir, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_image_dir, png_filename)

            with Image.open(tiff_path) as im:
                im.save(png_path, format="PNG")

def resize_image_for_tiled(img_array, target_max_size=800):
    """
    CONFIGURATION: You can change target_max_size to make images larger/smaller in Tiled
    - 800 = good balance of quality and performance (default)
    - 600 = smaller, faster loading
    - 1200 = larger, higher quality but slower loading
    """
    """
    Resize image array to a reasonable size for Tiled display
    
    Args:
        img_array: numpy array of the image
        target_max_size: maximum dimension (width or height) for the resized image
    
    Returns:
        resized numpy array
    """
    original_height, original_width = img_array.shape[:2]
    
    # Calculate scaling factor to keep aspect ratio
    if original_width > original_height:
        if original_width > target_max_size:
            scale_factor = target_max_size / original_width
        else:
            scale_factor = 1.0
    else:
        if original_height > target_max_size:
            scale_factor = target_max_size / original_height
        else:
            scale_factor = 1.0
    
    # Only resize if we need to make it smaller
    if scale_factor < 1.0:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Convert to PIL Image for resizing
        if len(img_array.shape) == 2:  # Grayscale
            img_pil = Image.fromarray(img_array)
        else:  # RGB or RGBA
            img_pil = Image.fromarray(img_array)
        
        # Resize with high quality resampling
        img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return np.array(img_resized)
    else:
        return img_array

def save_photogrammetry_scan_metadata_to_tiled(scan_params, scan_timestamp):
    """Save photogrammetry scan metadata to Tiled"""
    if not TILED_AVAILABLE or tiled_client is None:
        print(" Tiled not available for scan metadata upload")
        return False
        
    try:
        print(f" Uploading photogrammetry scan metadata to Tiled...")
        
        # Create container for photogrammetry scan metadata
        container_path = "photogrammetry_scan_metadata"
        if container_path not in tiled_client:
            tiled_client.create_container(container_path)
        container = tiled_client[container_path]
        
        # Create metadata array
        start_pos, end_pos, num_points, scan_type = scan_params
        metadata_array = np.array([start_pos, end_pos, num_points])
        
        # Use scan_type as the metadata key
        unique_key = f"metadata_{scan_type}"
        
        # Check if key exists and delete if needed
        if unique_key in container:
            print(f" Key {unique_key} exists, deleting...")
            del container[unique_key]
        
        # Store metadata array with fallback for ZARR issues
        try:
            container.write_array(
                metadata_array,
                metadata={
                    "description": f"Photogrammetry scan from {start_pos}° to {end_pos}° with {num_points} points",
                    "timestamp": scan_timestamp,
                    "start_position": start_pos,
                    "end_position": end_pos,
                    "num_points": num_points,
                    "scan_type": scan_type,
                    "motor_name": str(motor),
                    "datetime": datetime.now().isoformat(),
                    "scan_range_degrees": (end_pos - start_pos),
                    "step_size": (end_pos - start_pos) / (num_points - 1) if num_points > 1 else 0
                },
                key=unique_key
            )
            print(f"✓ Photogrammetry scan metadata uploaded to Tiled with key: {unique_key}")
            return True
            
        except Exception as zarr_error:
            if "zarr" in str(zarr_error).lower():
                print(f" ZARR storage failed for scan metadata, trying alternative approach...")
                
                # Alternative: Store as simple scalar
                container.write_array(
                    np.array([1.0]),  # Simple scalar to avoid ZARR
                    metadata={
                        "description": f"Photogrammetry scan from {start_pos}° to {end_pos}° with {num_points} points",
                        "timestamp": scan_timestamp,
                        "start_position": start_pos,
                        "end_position": end_pos,
                        "num_points": num_points,
                        "scan_type": scan_type,
                        "motor_name": str(motor),
                        "datetime": datetime.now().isoformat(),
                        "scan_range_degrees": (end_pos - start_pos),
                        "step_size": (end_pos - start_pos) / (num_points - 1) if num_points > 1 else 0,
                        "fallback_mode": "simplified"
                    },
                    key=unique_key
                )
                print(f"✓ Photogrammetry scan metadata uploaded to Tiled with key: {unique_key} (fallback mode)")
                return True
            else:
                raise zarr_error
        
    except Exception as e:
        print(f" Failed to upload photogrammetry scan metadata to Tiled: {e}")
        return False

def upload_photogrammetry_images_to_tiled(image_dirs, scan_timestamp, scan_params):
    """Upload all processed photogrammetry images to Tiled"""
    if not TILED_AVAILABLE or tiled_client is None:
        print(" Tiled not available for image upload")
        return False
    
    try:
        print(f" Uploading photogrammetry images to Tiled...")
        
        # Extract scan type from scan_params for container organization
        start_pos, end_pos, num_points, scan_type = scan_params
        
        # Use scan_type as the container name directly
        container_path = scan_type
        if container_path not in tiled_client:
            tiled_client.create_container(container_path)
        container = tiled_client[container_path]
        
        uploaded_count = 0
        total_images = 0
        
        # Process each image directory
        for dir_type, image_dir in image_dirs.items():
            if not os.path.exists(image_dir):
                print(f" Directory not found: {image_dir}")
                continue
                
            print(f" Processing {dir_type} images from: {image_dir}")
            
            # Get image files
            if dir_type == "png":
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
            else:
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.tiff', '.tif'))]
            
            # Sort files by position number for proper ordering (1, 2, 3, 4, 5... not 1, 10, 11, 2, 3...)
            def extract_position_number(filename):
                """Extract position number from filename for proper numeric sorting"""
                try:
                    if '_pos_' in filename:
                        parts = filename.split('_')
                        pos_idx = next((i for i, part in enumerate(parts) if part == 'pos'), -1)
                        if pos_idx >= 0 and pos_idx + 1 < len(parts):
                            return int(parts[pos_idx + 1])
                    return 0  # Default if no position found
                except (ValueError, IndexError):
                    return 0
            
            image_files.sort(key=extract_position_number)  # Sort by position number
            total_images += len(image_files)
            
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                
                try:
                    # Load image as numpy array
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    # Resize image for better Tiled display (max 800px on longest side)
                    original_shape = img_array.shape
                    img_array = resize_image_for_tiled(img_array, target_max_size=800)
                    if img_array.shape != original_shape:
                        print(f"  Resized {img_file} from {original_shape[:2]} to {img_array.shape[:2]} for Tiled")
                    
                    # Extract position info from filename if possible
                    position_index = 0
                    angle_value = 0.0
                    if '_pos_' in img_file:
                        try:
                            parts = img_file.split('_')
                            pos_idx = next((i for i, part in enumerate(parts) if part == 'pos'), -1)
                            if pos_idx >= 0 and pos_idx + 1 < len(parts):
                                position_index = int(parts[pos_idx + 1])
                            
                            # Try to extract angle if available
                            if 'angle' in img_file.lower():
                                angle_idx = next((i for i, part in enumerate(parts) if 'angle' in part.lower()), -1)
                                if angle_idx >= 0 and angle_idx + 1 < len(parts):
                                    angle_value = float(parts[angle_idx + 1])
                        except (ValueError, IndexError):
                            pass
                    
                    # Use original filename as the key (remove extension for cleaner key)
                    filename_without_ext = os.path.splitext(img_file)[0]
                    unique_key = filename_without_ext
                    
                    # Check if key exists and delete if needed
                    if unique_key in container:
                        print(f" Key {unique_key} exists, deleting...")
                        del container[unique_key]
                    
                    # Create metadata for image
                    img_metadata = {
                        "description": f"Photogrammetry {dir_type} image at position {position_index}",
                        "timestamp": scan_timestamp,
                        "position_index": position_index,
                        "angle_degrees": angle_value,
                        "filename": img_file,
                        "image_type": f"photogrammetry_{dir_type}",
                        "scan_params": scan_params,
                        "scan_type": scan_type,
                        "motor_name": str(motor),
                        "datetime": datetime.now().isoformat(),
                        "scroll_wheel_enabled": True,
                        "directory_type": dir_type,
                        "container_path": container_path,
                        "original_shape": original_shape,
                        "resized_shape": img_array.shape,
                        "resized_for_tiled": img_array.shape != original_shape
                    }
                    
                    # Upload to Tiled with fallback handling
                    try:
                        container.write_array(
                            img_array.astype(np.float32),
                            metadata=img_metadata,
                            key=unique_key
                        )
                        uploaded_count += 1
                        if uploaded_count % 10 == 0:  # Progress update every 10 images
                            print(f"   ✓ Uploaded {uploaded_count}/{total_images} images...")
                            
                    except Exception as zarr_error:
                        if "zarr" in str(zarr_error).lower():
                            # Fallback: Store metadata with image info
                            container.write_array(
                                np.array([position_index, angle_value, len(img_array.flatten())]),
                                metadata={**img_metadata, "fallback_mode": "metadata_only", "image_path": img_path, "image_shape": img_array.shape},
                                key=unique_key
                            )
                            uploaded_count += 1
                        else:
                            raise zarr_error
                    
                except Exception as e:
                    print(f" Failed to upload {img_file}: {e}")
        
        print(f"✓ Successfully uploaded {uploaded_count}/{total_images} photogrammetry images to Tiled")
        print(f" Images stored in container: {container_path}")
        return uploaded_count > 0
        
    except Exception as e:
        print(f" Failed to upload photogrammetry images to Tiled: {e}")
        return False

def upload_photogrammetry_image_stack_to_tiled(image_dir, scan_timestamp, scan_params):
    """Upload all PNG images as a 3D array stack for scroll wheel navigation"""
    if not TILED_AVAILABLE or tiled_client is None:
        print(" Tiled not available for image stack upload")
        return False
    
    try:
        print(f"Creating image stack for scroll wheel...")
        
        # Extract scan type from scan_params
        start_pos, end_pos, num_points, scan_type = scan_params
        
        # Use scan_type as the container name directly (same as individual images)
        stack_container_path = scan_type
        if stack_container_path not in tiled_client:
            tiled_client.create_container(stack_container_path)
        stack_container = tiled_client[stack_container_path]
        
        if not os.path.exists(image_dir):
            print(f" PNG directory not found: {image_dir}")
            return False
        
        # Get all PNG files and sort them by position number
        png_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
        
        # Sort files by position number for proper ordering (1, 2, 3, 4, 5... not 1, 10, 11, 2, 3...)
        def extract_position_number(filename):
            """Extract position number from filename for proper numeric sorting"""
            try:
                if '_pos_' in filename:
                    parts = filename.split('_')
                    pos_idx = next((i for i, part in enumerate(parts) if part == 'pos'), -1)
                    if pos_idx >= 0 and pos_idx + 1 < len(parts):
                        return int(parts[pos_idx + 1])
                return 0  # Default if no position found
            except (ValueError, IndexError):
                return 0
        
        png_files.sort(key=extract_position_number)  # Sort by position number
        
        if not png_files:
            print(f" No PNG files found in {image_dir}")
            return False
        
        print(f" Loading {len(png_files)} PNG images for stack...")
        
        # Load first image to get dimensions (resize for Tiled)
        first_img_path = os.path.join(image_dir, png_files[0])
        first_img = Image.open(first_img_path)
        first_img_array = np.array(first_img)
        first_img_resized = resize_image_for_tiled(first_img_array, target_max_size=800)
        img_height, img_width = first_img_resized.shape[:2]
        
        # Create 3D array: [num_images, height, width]
        image_stack = np.zeros((len(png_files), img_height, img_width), dtype=np.float32)
        
        # Load all images into the stack
        for i, png_file in enumerate(png_files):
            img_path = os.path.join(image_dir, png_file)
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Resize image for Tiled display
            img_array_resized = resize_image_for_tiled(img_array, target_max_size=800)
            
            # Ensure consistent dimensions
            if img_array_resized.shape[:2] == (img_height, img_width):
                image_stack[i] = img_array_resized
            else:
                # Additional resize if dimensions still don't match
                img_pil = Image.fromarray(img_array_resized.astype(np.uint8))
                img_resized = img_pil.resize((img_width, img_height), Image.Resampling.LANCZOS)
                image_stack[i] = np.array(img_resized)
            
            if (i + 1) % 5 == 0:  # Progress update
                print(f"  Loaded {i + 1}/{len(png_files)} images...")
        
        print(f" 3D image stack created: {image_stack.shape}")
        
        # Create stack key using simple format
        stack_key = "image_stack"
        
        # Check if key exists and delete if needed
        if stack_key in stack_container:
            print(f" Key {stack_key} exists, deleting...")
            del stack_container[stack_key]
        
        # Create rich metadata for the stack
        stack_metadata = {
            "description": f"3D photogrammetry image stack with {len(png_files)} images",
            "timestamp": scan_timestamp,
            "scan_type": scan_type,
            "num_images": len(png_files),
            "image_dimensions": f"{img_width}x{img_height}",
            "stack_shape": image_stack.shape,
            "start_position": start_pos,
            "end_position": end_pos,
            "num_points": num_points,
            "motor_name": str(motor),
            "datetime": datetime.now().isoformat(),
            "scroll_wheel_enabled": True,
            "data_type": "3D_image_stack",
            "png_files": png_files,
            "container_path": stack_container_path
        }
        
        # Upload the 3D stack to Tiled
        try:
            stack_container.write_array(
                image_stack,
                metadata=stack_metadata,
                key=stack_key
            )
            print(f" ✓ Stack uploaded to Tiled with key: {stack_key}")
            print(f" ✓ Stack stored in container: {stack_container_path}")
            print(f" ✓ Scroll wheel navigation enabled! Use mouse wheel to browse through {len(png_files)} images")
            return True
            
        except Exception as zarr_error:
            if "zarr" in str(zarr_error).lower():
                print(f"ZARR storage failed for image stack, trying alternative approach...")
                
                # Fallback: Store as metadata with stack info
                stack_container.write_array(
                    np.array([len(png_files), img_width, img_height]),
                    metadata={**stack_metadata, "fallback_mode": "metadata_only", "image_dir": image_dir},
                    key=stack_key
                )
                print(f" Image stack metadata uploaded to Tiled with key: {stack_key} (fallback mode)")
                return True
            else:
                raise zarr_error
        
    except Exception as e:
        print(f" Failed to upload image stack to Tiled: {e}")
        return False

if __name__ == "__main__":
    # Run scan
    try:
        print("Starting script")
        
        # Check Tiled availability
        if TILED_AVAILABLE and tiled_client:
            print(" Tiled integration enabled")
        else:
            print(" Running without Tiled integration")
        
        # File configuration
        base_path =  '/home/user/tmpData/AI_scan/' + sys.argv[4]
        save_dir = base_path + '/raw_images/'

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Then set the path in EPICS

        start_pos = float(sys.argv[1])
        end_pos = float(sys.argv[2])
        num_points = int(sys.argv[3])
        scan_type = sys.argv[4]

        # Generate timestamp for this scan
        scan_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        camera.tiff.file_path.put(save_dir)
        camera.tiff.file_template.put('%s%s_%d.tiff')

        print(f" Starting photogrammetry scan: {start_pos}° to {end_pos}° ({num_points} points)")
        RE(scan_with_saves(start_pos, end_pos, num_points))
        print(f" Photogrammetry scan completed")

        print(f" Cropping images...")
        cropImages(save_dir)
        print(f" Image cropping completed")

        image_dir_preprocess = os.path.join(base_path, "images")
        image_dir = os.path.join(base_path, "images_png")

        if ((os.path.exists(image_dir)) == 0):
            print(f" Converting images to PNG format...")
            convert_image_format(image_dir_preprocess, image_dir)
            print(f" ✓ Image format conversion completed")

        # Upload everything to Tiled after all processing is complete
        if TILED_AVAILABLE and tiled_client:
            print(f"\n Starting Tiled uploads for photogrammetry scan...")
            
            # Upload scan metadata
            scan_params = (start_pos, end_pos, num_points, scan_type)
            save_photogrammetry_scan_metadata_to_tiled(scan_params, scan_timestamp)
            
            # Only upload the final PNG images from images_png folder
            if os.path.exists(image_dir):
                print(f" Uploading only PNG images from: {image_dir}")
                image_dirs = {"png": image_dir}
                upload_photogrammetry_images_to_tiled(image_dirs, scan_timestamp, scan_params)
                
                # Also upload as a 3D image stack for scroll wheel navigation
                print(f"\n Creating scrollable 3D image stack...")
                upload_photogrammetry_image_stack_to_tiled(image_dir, scan_timestamp, scan_params)
            else:
                print(" PNG images directory not found for upload")
            
            print(f"✓ Tiled uploads completed!")
        else:
            print(f" Skipping Tiled uploads (Tiled not available)")

        #cropped_dir = save_dir.replace('images_uncropped/', 'images/')
        #average_output_dir = os.path.join(cropped_dir, 'averaged')
        #average_images_per_position(cropped_dir, average_output_dir)

    except KeyboardInterrupt:
        print("\nScan interrupted by user")
        RE.stop()
    except Exception as e:
        print(f"\nError during scan: {e}")
        #RE.stop()