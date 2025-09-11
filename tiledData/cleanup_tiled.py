#!/usr/bin/env python3
"""
Script to clean up Tiled data using the API
"""

from tiled.client import from_uri
import os
import sys

def get_deepest_items(container, path=""):
    """Get all the deepest nested items in a container (bottom-up approach)"""
    deepest_items = []
    
    if hasattr(container, 'keys') and container.keys():
        # This container has contents, so it's not the deepest
        for key in container.keys():
            current_path = f"{path}/{key}" if path else key
            # Recursively get deeper items
            deepest_items.extend(get_deepest_items(container[key], current_path))
    else:
        # This is a leaf item (no contents), so it's one of the deepest
        deepest_items.append(path)
    
    return deepest_items

def delete_container_recursively(tiled_client, container_name):
    """Recursively delete a container and all its nested contents using bottom-up approach"""
    try:
        # Get the container
        container = tiled_client[container_name]
        
        # First, get all the deepest nested items
        deepest_items = get_deepest_items(container, container_name)
        
        if deepest_items:
            print(f"  Found {len(deepest_items)} deepest items, deleting bottom-up...")
            
            # Delete from deepest to shallowest
            for item_path in sorted(deepest_items, key=lambda x: x.count('/'), reverse=True):
                print(f"    Deleting: {item_path}")
                try:
                    # Navigate to the item and delete it
                    path_parts = item_path.split('/')
                    
                    # Navigate to the parent of the item to be deleted
                    current = tiled_client
                    for part in path_parts[:-1]:
                        current = current[part]
                    
                    # Delete the item
                    current[path_parts[-1]].delete()
                    print(f"      ✓ Deleted {item_path}")
                except Exception as e:
                    print(f"      ✗ Failed to delete {item_path}: {e}")
        
        # Now delete the container itself
        container.delete()
        return True
        
    except Exception as e:
        print(f"      ✗ Failed to delete container: {e}")
        return False

def cleanup_specific_container(container_name):
    """Clean up a specific container in Tiled"""
    
    # Connect to Tiled
    tiled_client = from_uri(
        "http://localhost:8000",
        api_key=API_KEY
    )
    
    print(f"Connected to Tiled server")
    
    if container_name in tiled_client:
        print(f"Deleting container: {container_name}")
        try:
            # Recursively delete everything using bottom-up approach
            if delete_container_recursively(tiled_client, container_name):
                print(f"  ✓ Deleted container {container_name}")
            else:
                print(f"  ✗ Failed to delete {container_name}")
            
        except Exception as e:
            print(f"  ✗ Failed to delete {container_name}: {e}")
    else:
        print(f"Container '{container_name}' not found")

def cleanup_tiled():
    """Clean up all data in Tiled"""
    
    # Connect to Tiled
    tiled_client = from_uri(
        "http://localhost:8000",
        api_key=API_KEY
    )
    
    print("Connected to Tiled server")
    
    # List all top-level items
    for key in list(tiled_client.keys()):
        print(f"Deleting: {key}")
        try:
            # Recursively delete everything using bottom-up approach
            if delete_container_recursively(tiled_client, key):
                print(f"  ✓ Deleted {key}")
            else:
                print(f"  ✗ Failed to delete {key}")
            
        except Exception as e:
            print(f"  ✗ Failed to delete {key}: {e}")
    
    print("Cleanup complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Delete specific container
        container_name = sys.argv[1]
        cleanup_specific_container(container_name)
    else:
        # Delete all containers
        cleanup_tiled()
