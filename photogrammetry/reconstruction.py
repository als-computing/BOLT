import shutil, sys, subprocess, os, time 
from PIL import Image
import numpy as np
from datetime import datetime

# Manual Tiled writing imports
try:
    from tiled.client import from_uri
    TILED_AVAILABLE = True
except ImportError:
    TILED_AVAILABLE = False
    print("Warning: Tiled client not available, running without Tiled integration")

# Set up manual Tiled connection
tiled_client = None
if TILED_AVAILABLE:
    try:
        tiled_uri = "http://localhost:8000"
        tiled_api_key = "API_KEY"
        
        tiled_client = from_uri(tiled_uri, api_key=tiled_api_key)
        print(f"✓ Tiled client connected to {tiled_uri}")
    except Exception as e:
        print(f"Failed to connect to Tiled: {e}")
        TILED_AVAILABLE = False
        tiled_client = None

def ensure_directories(*paths: str) -> None:
    """ Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def feature_extraction(imageDir: str, databasePath: str, colmapPath: str):
    """ Run feature extraction. """
    subprocess.run([
        colmapPath, "feature_extractor",
        "--database_path", databasePath,
        "--image_path", imageDir,
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", "6"  # or 1 to be safe
    ], check=True)

def feature_matching(databasePath: str, colmapPath: str):
    """ 
    Run feature matching. 
    """
    subprocess.run([
        colmapPath, "exhaustive_matcher",
        "--database_path", databasePath,
        "--SiftMatching.use_gpu", "0",
    ], check=True)

def sparse_reconstruction(imageDir: str, databasePath: str, colmapPath: str, sparseDir: str):
    """ 
    Run sparse reconstruction. 
    """
    subprocess.run([
        colmapPath, "mapper",
        "--database_path", databasePath,
        "--image_path", imageDir,
        "--output_path", sparseDir
    ], check=True)

def image_undistorter(imageDir: str, denseDir: str, colmapPath: str, sparseDir: str):
    """ 
    Run image undistorter, which outputs acccording 
    to quality of point detection.
    """
    subprocess.run([
        colmapPath, "image_undistorter",
        "--image_path", imageDir,
        "--input_path", sparseDir, "0",
        "--output_path", denseDir,
        "--output_type", "COLMAP"
    ], check=True)

def interface_colmap(workspace_dir: str, imageDir: str, basePath: str, sceneMVS: str, denseDir: str, mvs_bin_dir: str):
    """ 
    Run interfaceCOLMAP, which converts gathered data from 
    colmap and output scene_mvs, containing necessary scene 
    for next reconstruction steps. 
    """
    subprocess.run([
        os.path.join(mvs_bin_dir, "InterfaceCOLMAP"),
        "-i", os.path.join(workspace_dir, "dense"),       #Needed for manual colmap
        #"-i", os.path.join(workspace_dir, "dense", "0"),  Needed for automaticReconstruction combo     
        "-o", sceneMVS,
        "--image-folder", imageDir
    ],cwd=denseDir)

def densify_point_cloud(sceneMVS: str, denseDir: str, mvs_bin_dir: str):
    """ 
    Densify point cloud connects points by creating more points
    in between already created points. """
    subprocess.run([
        os.path.join(mvs_bin_dir, "DensifyPointCloud"),
        sceneMVS
    ], cwd=denseDir)

def reconstruct_mesh(denseMVS: str, denseDir: str, mvs_bin_dir: str):
    """ 
    Creates mesh of object, lacking color but connecting points.
    """
    subprocess.run([
        os.path.join(mvs_bin_dir, "ReconstructMesh"),
        denseMVS
    ], cwd=denseDir)

def texture_mesh(denseDir: str, sceneMVS: str, mvs_bin_dir: str):
    """ 
    Texture mesh mixes data from both sceneMVS and the mesh 
    
    Note: This is time consuming and should be skipped if mesh quality is subpar
    """

    subprocess.run([
        os.path.join(mvs_bin_dir, "TextureMesh"), 
        sceneMVS,
        "-m", "scene_dense_mesh.ply"
    ],cwd=denseDir)

def run_colmap_pipeline(image_dir: str, workspace_dir: str, colmap_path: str = "colmap") -> None:
    """
    Full COLMAP pipeline: feature extraction → matching → sparse reconstruction → undistortion.
    """
    print(f"Running COLMAP pipeline on {image_dir} -> {workspace_dir}")
    
    # Validate input directory exists
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    database_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    dense_dir = os.path.join(workspace_dir, "dense")

    ensure_directories(workspace_dir, sparse_dir, dense_dir)

    feature_extraction(image_dir, database_path, colmap_path)
    feature_matching(database_path, colmap_path)
    sparse_reconstruction(image_dir, database_path, colmap_path, sparse_dir)
    image_undistorter(image_dir, dense_dir, colmap_path, os.path.join(sparse_dir, "0"))

    print("COLMAP pipeline completed.")

def run_openmvs_pipeline(base_path: str, image_dir: str, workspace_dir: str, mvs_bin_dir: str, image_file_name: str) -> None:
    """
    Run OpenMVS conversion and mesh reconstruction from COLMAP output.
    """

    dense_dir = os.path.join(workspace_dir, "dense")
    scene_mvs = os.path.join(dense_dir, "scene.mvs")
    dense_mvs = os.path.join(dense_dir, "scene_dense.mvs")
    ensure_directories(dense_dir)

    interface_colmap(workspace_dir, image_dir, base_path, scene_mvs, dense_dir, mvs_bin_dir)
    densify_point_cloud(scene_mvs, dense_dir, mvs_bin_dir)
    reconstruct_mesh(dense_mvs, dense_dir, mvs_bin_dir)
    texture_mesh(dense_dir, scene_mvs, mvs_bin_dir)

    print("OpenMVS pipeline completed.")

def automatic_reconstruction(image_dir: str, workspace_dir: str, colmap_path: str = "colmap"):
    """
    Run COLMAP's automatic reconstruction pipeline.
    """
    database_path = os.path.join(workspace_dir, "database_dir")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    dense_dir = os.path.join(workspace_dir, "dense")

    ensure_directories(workspace_dir, sparse_dir, dense_dir)

    subprocess.run([
        colmap_path, "automatic_reconstructor",
        "--workspace_path", workspace_dir,
        "--image_path", image_dir,
        "--data_type", "individual",  # or 'video' depending on your input
        "--quality", "medium",        # can be 'low', 'medium', or 'high'
        "--sparse", "true",
        "--dense", "true",
    ], check=True)

    print("Automatic reconstruction completed.")

def check_colmap_availability():
    """Check if COLMAP is available and working."""
    try:
        result = subprocess.run(['colmap', 'help'], capture_output=True, text=True, check=True)
        print(" COLMAP is available and working")
        return True
    except subprocess.CalledProcessError as e:
        print(f" COLMAP command failed: {e}")
        return False
    except FileNotFoundError:
        print(" COLMAP not found in PATH")
        return False

def check_openmvs_availability():
    """Check if OpenMVS tools are available and working."""
    mvs_bin_path = "/usr/local/bin/OpenMVS/"
    required_tools = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh"]
    
    if not os.path.exists(mvs_bin_path):
        print(f" OpenMVS directory not found: {mvs_bin_path}")
        return False
    
    missing_tools = []
    for tool in required_tools:
        tool_path = os.path.join(mvs_bin_path, tool)
        if not os.path.exists(tool_path):
            missing_tools.append(tool)
    
    if missing_tools:
        print(f" Missing OpenMVS tools: {', '.join(missing_tools)}")
        return False

    print("OpenMVS tools are available and working")
    return True

def validate_reconstruction_paths(base_path: str, image_folder: str):
    """Validate that all required paths exist for reconstruction."""
    image_dir = os.path.join(base_path, image_folder, "images_png")
    workspace_dir = os.path.join(base_path, image_folder, "workspace")
    
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return False
    
    print(f"Image directory found: {image_dir}")
    print(f"Workspace will be created at: {workspace_dir}")
    return True

def read_ply_file_basic(ply_path: str):
    """
    Read basic information from PLY file (vertex count, face count, etc.)
    Returns metadata about the 3D model without loading the full geometry.
    """
    if not os.path.exists(ply_path):
        return None
    
    vertex_count = 0
    face_count = 0
    file_size = os.path.getsize(ply_path)
    
    try:
        with open(ply_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    face_count = int(line.split()[-1])
                elif line == 'end_header':
                    break
        
        return {
            "vertex_count": vertex_count,
            "face_count": face_count,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        print(f"Error reading PLY file {ply_path}: {e}")
        return None

def upload_reconstruction_metadata_to_tiled(image_folder: str, reconstruction_stats: dict, pipeline_type: str, rotation_views_uuid: str = None):
    """Upload reconstruction metadata to Tiled"""
    if not TILED_AVAILABLE or tiled_client is None:
        print("Tiled not available for reconstruction metadata upload")
        return False, None
    
    try:
        print(f"Uploading reconstruction metadata to Tiled...")
        
        # Create reconstruction metadata array
        stats_array = np.array([
            reconstruction_stats.get('total_time', 0),
            reconstruction_stats.get('colmap_time', 0),
            reconstruction_stats.get('openmvs_time', 0)
        ])
        
        metadata = {
            "description": f"3D reconstruction metadata for {image_folder}",
            "image_folder": image_folder,
            "pipeline_type": pipeline_type,
            "reconstruction_stats": reconstruction_stats,
            "datetime": datetime.now().isoformat(),
            "data_type": "reconstruction_metadata",
            "rotation_views_uuid": rotation_views_uuid
        }
        
        # Upload metadata directly to Tiled
        uploaded_array = tiled_client.write_array(
            stats_array,
            metadata=metadata
        )
        
        # Extract UUID from the uploaded array
        try:
            if hasattr(uploaded_array, 'uri'):
                array_uuid = uploaded_array.uri.split('/')[-1]
            else:
                array_uuid = str(uploaded_array)
        except:
            array_uuid = str(uploaded_array)
        
        print(f"Reconstruction metadata uploaded to Tiled: {array_uuid}")
        if rotation_views_uuid:
            print(f" Rotation views UUID included: {rotation_views_uuid}")
        return True, array_uuid
        
    except Exception as e:
        print(f"Failed to upload reconstruction metadata to Tiled: {e}")
        return False, None

def generate_and_upload_rotation_views(image_folder: str, ply_path: str):
    """Generate rotation views of the PLY file and upload them to Tiled"""
    if not TILED_AVAILABLE or tiled_client is None:
        print("Tiled not available for rotation views upload")
        return False, None
    
    try:
        print(f" Generating rotation views for PLY file...")
        
        # Use display_object.py to generate views
        import subprocess
        
        # Create output folder for rotation views
        views_folder = os.path.join(os.path.dirname(ply_path), "rotation_views")
        
        # Empty the folder if it exists
        if os.path.exists(views_folder):
            print(f"Emptying existing rotation views folder: {views_folder}")
            import shutil
            shutil.rmtree(views_folder)
        
        # Generate views using display_object.py
        print(f"Generating 12 views using display_object.py...")
        script_path = "/home/user/Repos/BOLT/functions/display_object.py"
        cmd = ["python", script_path, ply_path, "fast", "12"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/user/Repos/bluesky-web")
            if result.returncode == 0:
                print("Successfully generated and uploaded views with display_object.py")
                print(result.stdout)
                
                # Extract UUID from the output if available
                import re
                uuid_pattern = r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
                uuid_matches = re.findall(uuid_pattern, result.stdout)
                if uuid_matches:
                    tiled_uuid = uuid_matches[-1]  # Get the last UUID found
                    print(f" Tiled UUID for rotation views: {tiled_uuid}")
                    return True, tiled_uuid
                else:
                    print(" No UUID found in output")
                    return True, None
            else:
                print(f"Error running display_object.py: {result.stderr}")
                return False, None
        except Exception as e:
            print(f"Error executing display_object.py: {e}")
            return False, None
        
    except Exception as e:
        print(f" Failed to generate and upload rotation views: {e}")
        return False, None

def upload_ply_metadata_to_tiled(image_folder: str, ply_files: list):
    """Upload PLY file metadata to Tiled (without the full 3D data)"""
    if not TILED_AVAILABLE or tiled_client is None:
        print("Tiled not available for PLY metadata upload")
        return False
    
    try:
        print(f"Uploading PLY metadata to Tiled...")
        
        for ply_file in ply_files:
            if not os.path.exists(ply_file):
                continue
                
            # Read PLY file information
            ply_info = read_ply_file_basic(ply_file)
            if ply_info is None:
                continue
            
            # Create metadata array with basic stats
            stats_array = np.array([
                ply_info["vertex_count"],
                ply_info["face_count"], 
                ply_info["file_size_bytes"]
            ])
            
            metadata = {
                "description": f"3D mesh model metadata: {os.path.basename(ply_file)}",
                "filename": os.path.basename(ply_file),
                "file_path": ply_file,
                "vertex_count": ply_info["vertex_count"],
                "face_count": ply_info["face_count"],
                "file_size_bytes": ply_info["file_size_bytes"],
                "file_size_mb": ply_info["file_size_mb"],
                "data_type": "3d_model_metadata",
                "image_folder": image_folder,
                "datetime": datetime.now().isoformat(),
                "download_url": f"file://{ply_file}",
                "viewer_hint": "Use MeshLab, CloudCompare, or Blender to view this 3D model"
            }
            
            # Upload PLY metadata directly to Tiled
            uploaded_array = tiled_client.write_array(
                stats_array,
                metadata=metadata
            )
            
            # Extract UUID from the uploaded array
            try:
                if hasattr(uploaded_array, 'uri'):
                    array_uuid = uploaded_array.uri.split('/')[-1]
                else:
                    array_uuid = str(uploaded_array)
            except:
                array_uuid = str(uploaded_array)
            
            print(f" PLY metadata uploaded: {array_uuid} ({ply_info['vertex_count']} vertices, {ply_info['face_count']} faces)")
        
        return True
        
    except Exception as e:
        print(f"Failed to upload PLY metadata to Tiled: {e}")
        return False

def find_reconstruction_outputs(workspace_dir: str):
    """Find the best quality reconstruction files (prioritize textured mesh)"""
    output_files = {
        "best_ply_file": None,
        "all_ply_files": [],
        "mvs_files": [],
        "colmap_files": []
    }
    
    dense_dir = os.path.join(workspace_dir, "dense")
    if os.path.exists(dense_dir):
        # Look for PLY files with quality priority
        ply_priority = [
            "scene_texture.ply",      # Best: Textured mesh (final result)
            "scene_dense_mesh.ply",   # Good: Dense mesh with geometry
            "scene_dense.ply"         # Basic: Point cloud
        ]
        
        all_ply_files = [f for f in os.listdir(dense_dir) if f.endswith('.ply')]
        output_files["all_ply_files"] = [os.path.join(dense_dir, f) for f in all_ply_files]
        
        # Find the best available PLY file based on priority
        for priority_file in ply_priority:
            if priority_file in all_ply_files:
                output_files["best_ply_file"] = os.path.join(dense_dir, priority_file)
                print(f" Best PLY file found: {priority_file}")
                break
        
        # If no priority files found, use the first available PLY
        if not output_files["best_ply_file"] and all_ply_files:
            output_files["best_ply_file"] = os.path.join(dense_dir, all_ply_files[0])
            print(f" Using available PLY file: {all_ply_files[0]}")
        
        # Look for MVS files
        for file in os.listdir(dense_dir):
            if file.endswith('.mvs'):
                output_files["mvs_files"].append(os.path.join(dense_dir, file))
    
    # Look for COLMAP sparse reconstruction
    sparse_dir = os.path.join(workspace_dir, "sparse")
    if os.path.exists(sparse_dir):
        colmap_files = ["cameras.txt", "images.txt", "points3D.txt"]
        for subdir in os.listdir(sparse_dir):
            subdir_path = os.path.join(sparse_dir, subdir)
            if os.path.isdir(subdir_path):
                for colmap_file in colmap_files:
                    file_path = os.path.join(subdir_path, colmap_file)
                    if os.path.exists(file_path):
                        output_files["colmap_files"].append(file_path)
    
    return output_files

if __name__ == "__main__":
    #File path names
    if len(sys.argv) < 2:
        print("Usage: python reconstruction.py <image_folder_name> [--automatic]")
        print("  --automatic: Use COLMAP automatic reconstruction instead of manual pipeline")
        sys.exit(1)
        
    image_file_name = sys.argv[1]
    use_automatic = "--automatic" in sys.argv

    #Where reconstructions and image data is stored
    base_path = "/home/user/tmpData/AI_scan/"

    #Depending on where openMVS is ran
    mvs_bin_path = "/usr/local/bin/OpenMVS/"

    # First check COLMAP availability
    if not check_colmap_availability():
        print("Cannot proceed without COLMAP")
        sys.exit(1)

    # Check OpenMVS availability
    if not check_openmvs_availability():
        print("OpenMVS not available - will only run COLMAP pipeline")
        run_openmvs = False
    else:
        run_openmvs = True

    # Validate paths
    if not validate_reconstruction_paths(base_path, image_file_name):
        print("Path validation failed")
        sys.exit(1)

    image_dir = os.path.join(base_path, image_file_name, "images_png")
    workspace_dir = os.path.join(base_path, image_file_name, "workspace")

    print("Starting reconstruction pipeline...")
    t0 = time.time()
    
    reconstruction_stats = {
        "start_time": datetime.now().isoformat(),
        "colmap_time": 0,
        "openmvs_time": 0,
        "total_time": 0
    }
    
    try:
        if use_automatic:
            # Run COLMAP automatic reconstruction
            print("Using COLMAP automatic reconstruction...")
            automatic_reconstruction(image_dir, workspace_dir)
            t1 = time.time()
            reconstruction_stats["colmap_time"] = t1 - t0
            print(f"COLMAP automatic reconstruction completed in {t1 - t0:.2f}s")
            pipeline_type = "automatic"
        else:
            # Run manual COLMAP pipeline
            run_colmap_pipeline(image_dir, workspace_dir)
            t1 = time.time()
            reconstruction_stats["colmap_time"] = t1 - t0
            print(f"COLMAP pipeline completed in {t1 - t0:.2f}s")
            pipeline_type = "manual"

        # Run OpenMVS pipeline if available
        if run_openmvs:
            print("Starting OpenMVS pipeline...")
            t2 = time.time()
            run_openmvs_pipeline(base_path, image_dir, workspace_dir, mvs_bin_path, image_file_name)
            t3 = time.time()
            reconstruction_stats["openmvs_time"] = t3 - t2
            reconstruction_stats["total_time"] = t3 - t0
            print(f"OpenMVS pipeline completed in {t3 - t2:.2f}s")
            print(f"Total reconstruction time: {t3 - t0:.2f}s")
        else:
            reconstruction_stats["total_time"] = t1 - t0
            print("OpenMVS pipeline skipped - COLMAP results available in workspace")
        
        # Upload reconstruction results to Tiled
        if TILED_AVAILABLE and tiled_client:
            print(f"\n Starting Tiled uploads for reconstruction results...")
            
            # Find all output files
            output_files = find_reconstruction_outputs(workspace_dir)
            
            # Upload only the best PLY file metadata
            rotation_views_uuid = None
            if output_files["best_ply_file"]:
                best_file_name = os.path.basename(output_files["best_ply_file"])
                print(f"Uploading best quality 3D model: {best_file_name}")
                upload_ply_metadata_to_tiled(image_file_name, [output_files["best_ply_file"]])
                
                # Generate and upload rotation views
                print(f" Generating rotation views for 3D model...")
                success, rotation_views_uuid = generate_and_upload_rotation_views(image_file_name, output_files["best_ply_file"])
                if not success:
                    print("Warning: Rotation views generation failed")
            else:
                print("No PLY files found to upload")
            
            # Upload reconstruction metadata with rotation views UUID
            success, metadata_uuid = upload_reconstruction_metadata_to_tiled(image_file_name, reconstruction_stats, pipeline_type, rotation_views_uuid)
            
            if success and metadata_uuid:
                print(f" Reconstruction metadata UUID: {metadata_uuid}")
                print(f" This UUID contains the rotation views UUID: {rotation_views_uuid}")
            
            print(f"Tiled uploads completed!")
        else:
            print(f"Skipping Tiled uploads (Tiled not available)")
        
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        sys.exit(1)
