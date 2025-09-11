# flake8: noqa
print(f"Loading file {__file__!r}")

from tkinter import S
from typing import Any, Dict, List, Optional
from ophyd import EpicsSignal
from bluesky_queueserver.manager.annotation_decorator import parameter_annotation_decorator
from datetime import datetime

from bluesky.plans import (
    adaptive_scan as _adaptive_scan,
    count,
    fly as _fly,
    grid_scan as _grid_scan,
    inner_product_scan as _inner_product_scan,
    list_grid_scan as _list_grid_scan,
    list_scan as _list_scan,
    log_scan as _log_scan,
    ramp_plan as _ramp_plan,
    rel_adaptive_scan as _rel_adaptive_scan,
    rel_grid_scan as _rel_grid_scan,
    rel_list_grid_scan as _rel_list_grid_scan,
    rel_list_scan as _rel_list_scan,
    rel_log_scan as _rel_log_scan,
    rel_scan as _rel_scan,
    rel_spiral as _rel_spiral,
    rel_spiral_fermat as _rel_spiral_fermat,
    rel_spiral_square as _rel_spiral_square,
    relative_inner_product_scan as _relative_inner_product_scan,
    spiral as _spiral,
    spiral_fermat as _spiral_fermat,
    spiral_square as _spiral_square,
    tune_centroid as _tune_centroid,
    tweak as _tweak,
    x2x_scan as _x2x_scan,
)

# Import mv and rd inside functions to hide them from queue server detection
# 1D scan for endstation x, z or filters
@parameter_annotation_decorator({
    "description": "Move motor to specified position",
    "parameters": {
        "motor": {
            "description": "Required. Inidividual motor that is moved to desired position",
            "default": "rotation_motor",
            "annotation": "typing.Any",
            "convert_device_names": True,

      
        },
        "position": {
            "description": "Required. The position for the motor, uses the default units of the motor",
            "default": 0.0,
            "min": 0,
            "max": 360,
            "step": 20,
         
        }
    }
})
# Move motor certian amount
def move_motor(motor="rotation_motor", position=0.0, *, md=None):
    from bluesky import plan_stubs as bps
    import math
    # Convert position to float to handle string inputs
    position_move = float(int(position) / 2.8125)

    yield from bps.mv(motor, position_move) 

    reading = yield from bps.read(motor)    

    position_match = math.isclose(reading[motor.name]["value"], position_move, rel_tol=1e-3)
    md = {
        "run_result": "success" if position_match else "failure"
    }
    yield from bps.open_run(md=md)
    yield from bps.create()                
    yield from bps.save()                  
    yield from bps.close_run()     # Close the run


@parameter_annotation_decorator({
    "description": "Read motor angle and store in Tiled using TiledWriter",
    "parameters": {
        "motor": {
            "description": "Required. Motor device to read",
            "annotation": "typing.Any",
            "convert_device_names": True,
        }
    }
})
# Measure current motor position
def get_angle(motor, *, md=None):
    from bluesky import plan_stubs as bps
    reading = yield from bps.read(motor)    
    md = {
        "motor_angle": reading[motor.name]["value"],
        "motor_name": motor.name,
        "angle_degrees": reading[motor.name]["value"] * 2.8125,
        "timestamp": datetime.now().isoformat()
    }

    yield from bps.open_run(md=md)
    yield from bps.create()                
    yield from bps.save()                  
    yield from bps.close_run()     # Close the run
    
    # Return the reading data
    return reading
    
# Simple image acquisition using TiledWriter (recommended approach)
@parameter_annotation_decorator({
    "description": "Capture single image and store directly in Tiled using TiledWriter",
    "parameters": {
        "camera": {
            "description": "Required. Area Detector camera device",
            "annotation": "typing.Any",
            "convert_device_names": True,
        },
        "motor": {
            "description": "Required. Motor device to read",
            "annotation": "typing.Any",
            "convert_device_names": True,
        }
    }
})
#Acquire image using Area Detector and save to Tiled
def camera_acquire(camera="camera", motor="rotation_motor", *, md=None):
    import subprocess
    from bluesky import plan_stubs as bps

    reading = yield from get_angle(motor)
    angle_value = float(reading[motor.name]["value"])

    angle = f"{(angle_value*2.8125):.2f}"
    run_id = yield from bps.open_run(md=md)
    run_id = run_id + "_"
    cmd = ["python", "/home/user/Repos/BOLT/functions/take_measurement.py", angle, run_id]
    #cmd = ["pwd"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print(result.returncode)

    yield from bps.close_run()

    return "image_captured"
    
@parameter_annotation_decorator({
    "description": "Perform rotation scan with camera acquisition, for testing",
    "parameters": {
        "start_angle": {
            "description": "Starting angle in degrees",
            "annotation": "str"
        },
        "end_angle": {
            "description": "Ending angle in degrees",
            "annotation": "str"
        },
        "num_points": {
            "description": "Number of measurement points",
            "annotation": "str"
        },
        "save_dir": {
            "description": "Directory in which to save images",
            "annotation": "str"
        }
    }
})
# Perform rotation scan with camera acquisition
def rotation_scan(start_angle="0", end_angle="90", num_points="10", save_dir="default", *, md=None):
    from bluesky import plan_stubs as bps
    import subprocess
    run_id = yield from bps.open_run(md=md)
    cmd = ["python", "/home/user/Repos/BOLT/functions/run_photogrammetry_scan.py", str(start_angle), str(end_angle), str(num_points), save_dir, run_id]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print(result.returncode)

    yield from bps.close_run()

    return "scan_completed"
    
@parameter_annotation_decorator({
    "description": "Perform reconstruction algorithm, on a given set of images",
    "parameters": {
        "image_dir": {
            "description": "Directory to save the images",
            "annotation": "str"
        }
    }
})
# Perform reconstruction algorithm, on a given set of images
def reconstruct_object(image_dir="default", *, md=None):
    import subprocess
    import re
    from bluesky import plan_stubs as bps
    
    yield from bps.open_run(md=md)
    cmd = ["python", "/home/user/Repos/BOLT/functions/reconstruction.py", image_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print(result.returncode)

    # Extract reconstruction metadata UUID from output
    reconstruction_metadata_uuid = None
    rotation_views_uuid = None
    
    if result.returncode == 0:
        # Extract reconstruction metadata UUID
        metadata_uuid_pattern = r'Reconstruction metadata UUID: ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
        metadata_matches = re.findall(metadata_uuid_pattern, result.stdout)
        if metadata_matches:
            reconstruction_metadata_uuid = metadata_matches[-1]
            print(f" Extracted reconstruction metadata UUID: {reconstruction_metadata_uuid}")
        
        # Extract rotation views UUID
        rotation_uuid_pattern = r'Tiled UUID for rotation views: ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
        rotation_matches = re.findall(rotation_uuid_pattern, result.stdout)
        if rotation_matches:
            rotation_views_uuid = rotation_matches[-1]
            print(f" Extracted rotation views UUID: {rotation_views_uuid}")

    # Store UUIDs in the current run's metadata
    reconstruction_metadata = {
        "reconstruction_metadata_uuid": reconstruction_metadata_uuid,
        "rotation_views_uuid": rotation_views_uuid,
        "reconstruction_completed": result.returncode == 0
    }
    
    # Update the current run's metadata with the UUIDs
    yield from bps.create()
    yield from bps.save()
    
    yield from bps.close_run()

    yield from bps.open_run(md=reconstruction_metadata)
    yield from bps.create()
    yield from bps.save()
    
    yield from bps.close_run()
    

    return f"Reconstruction completed - Metadata UUID: {reconstruction_metadata_uuid}"

@parameter_annotation_decorator({
    "description": "Analyze 3D reconstruction quality and store results in Bluesky metadata",
    "parameters": {
        "image_dir": {
            "description": "Directory name containing the reconstruction results",
            "annotation": "str"
        }
    }
})
# Analyze 3D reconstruction quality and store results in Bluesky metadata
def analyze_ply_quality(image_dir="default", *, md=None):
    import subprocess
    import sys
    import os
    from datetime import datetime
    
    # Start Bluesky run
    run_id = yield from bps.open_run(md=md)

    try:
        print(f" Starting PLY quality analysis for directory: {image_dir}")
        
        # Import and call the analyzer function directly
        sys.path.append(os.getcwd())
        print(" Importing PLY quality analyzer...")
        from ply_quality_analyzer import run_quality_analysis_with_metadata
        print(" PLY quality analyzer imported successfully")
        
        # Run the complete quality analysis
        print(" Running quality analysis...")
        analysis_results = run_quality_analysis_with_metadata(image_dir)
        print(" Quality analysis completed")
        
        # Store analysis results in Bluesky metadata
        quality_metadata = {
            "ply_quality_analysis": analysis_results,
            "analyzed_directory": image_dir,
            "analysis_completed": datetime.now().isoformat()
        }

        yield from bps.close_run()

        run_id = yield from bps.open_run(md=quality_metadata)

        # Update the current run with quality analysis metadata
        yield from bps.create()
        yield from bps.save()
        
        yield from bps.close_run()

        # Return the summary from the analyzer
        return analysis_results.get("summary", "Quality analysis completed")
            
    except Exception as e:
        error_metadata = {
            "ply_quality_analysis": {
                "analysis_success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "analysis_timestamp": datetime.now().isoformat(),
                "summary": f"Quality analysis failed with exception: {e}"
            },
            "analysis_run_id": run_id,
            "analyzed_directory": image_dir
        }
        
        # Update the current run with error metadata
        yield from bps.create()
        yield from bps.save()
        yield from bps.close_run()
        
        error_msg = f"Quality analysis failed with exception: {e}"
        print(f" {error_msg}")
        return error_msg


@parameter_annotation_decorator({
    "description": "testing if adding plans works",
    "parameters": {
        "image_dir": {
            "description": "Test dir to see if everything is working",
            "annotation": "str"
        }
    }
})
# Display the best available PLY file from reconstruction results
def display_object_from_file(image_dir="default", *, md=None):
    """
    Display the best available PLY file from reconstruction results.
    Priority order: scene_texture > scene_dense_mesh > scene_dense
    """
    import os
    import subprocess
    import sys
    from bluesky import plan_stubs as bps
    
    # Where reconstructions and image data is stored
    base_path = "/home/user/tmpData/AI_scan/"
    path = os.path.join(base_path, image_dir, "workspace", "dense")
    
    # Priority order for PLY files (best to worst)
    ply_priorities = [
        "scene_texture.ply",
        "scene_dense_mesh.ply", 
        "scene_dense.ply"
    ]
    
    # Find the best available PLY file
    best_ply = None
    for ply_file in ply_priorities:
        ply_path = os.path.join(path, ply_file)
        if os.path.exists(ply_path):
            best_ply = ply_path
            print(f"Found best available PLY: {ply_file}")
            break
    
    if best_ply is None:
        print(f" No PLY files found in {path}")
        print("   Expected files: scene_texture.ply, scene_dense_mesh.ply, or scene_dense.ply")
        yield from bps.open_run(md=md)
        yield from bps.create()
        yield from bps.save()
        yield from bps.close_run()

        return "No PLY files found"
    
    # Use the display_object.py script to generate views and upload to Tiled
    print(f" Displaying PLY file: {os.path.basename(best_ply)}")
    
    # Run the display_object.py script
    script_path = "/home/user/Repos/BOLT/functions/display_object.py"
    cmd = ["python", script_path, best_ply, "fast", "12"]
    
    # Extract Tiled UUID from subprocess output
    tiled_uuid = None
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/user/Repos/bluesky-web")
        if result.returncode == 0:
            print(" Successfully generated and uploaded PLY views to Tiled")
            print(result.stdout)
            
            # Extract UUID from the output
            import re
            uuid_pattern = r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
            uuid_matches = re.findall(uuid_pattern, result.stdout)
            if uuid_matches:
                tiled_uuid = uuid_matches[-1]  # Get the last UUID found (should be the array UUID)
                print(f" Extracted Tiled UUID: {tiled_uuid}")
            else:
                print(" No UUID found in output")
        else:
            print(f" Error running display script: {result.stderr}")
    except Exception as e:
        print(f" Error executing display: {e}")
    
    # Store results in Bluesky metadata
    display_metadata = {
        "display_plan": "completed",
        "ply_file_used": os.path.basename(best_ply),
        "ply_file_path": best_ply,
        "script_used": "display_object.py",
        "views_generated": 12,
        "tiled_array_uuid": tiled_uuid
    }
    
    run_id_0 = yield from bps.open_run(md={**md, **display_metadata} if md else display_metadata)
    yield from bps.create()
    yield from bps.save()
    run_id_1 = yield from bps.close_run()

    if tiled_uuid:
        print(f" Tiled Array UUID: {tiled_uuid}")
    
    return f"Display plan completed - used {os.path.basename(best_ply)} - Tiled UUID: {tiled_uuid}"