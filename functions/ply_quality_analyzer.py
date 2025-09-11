
"""
PLY Quality Analyzer for BOLT Beamline System.

This module provides comprehensive quality assessment for 3D reconstructions
from photogrammetry scans using trimesh and advanced geometric analysis.
"""

import numpy as np
import trimesh
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for 3D reconstructions."""
    
    # Basic mesh properties
    vertex_count: int = 0
    face_count: int = 0
    is_watertight: bool = False
    is_winding_consistent: bool = False
    
    # Geometric quality metrics
    surface_area: float = 0.0
    volume: float = 0.0
    bounding_box_volume: float = 0.0
    
    # Mesh quality indicators
    aspect_ratio_stats: Dict[str, float] = None
    edge_length_stats: Dict[str, float] = None
    
    # Reconstruction-specific metrics
    hole_count: int = 0  # Surface holes (excluding bottom hole)
    total_holes: int = 0  # Total holes including bottom hole
    bottom_hole_size: int = 0  # Size of the bottom hole
    noise_level: float = 0.0
    completeness_score: float = 0.0
    
    # Overall quality score
    overall_quality_score: float = 0.0
    
    # Defect detection
    detected_defects: List[str] = None
    critical_issues: List[str] = None

class PLYQualityAnalyzer:
    """Advanced quality analyzer for PLY files from photogrammetry reconstructions."""
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.75,
            'fair': 0.6,
            'poor': 0.4
        }
        
    def analyze_ply_file(self, ply_path: str) -> Tuple[QualityMetrics, List[Dict]]:
        """
        Comprehensive analysis of PLY file quality.
        
        Args:
            ply_path: Path to the PLY file
            
        Returns:
            Tuple of quality metrics and improvement recommendations
        """
        
        if not Path(ply_path).exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
            
        # Load mesh with trimesh
        mesh = self._load_with_trimesh(ply_path)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_quality_metrics(mesh)
        
        # Generate improvement recommendations
        recommendations = self._generate_recommendations(metrics, mesh)
        
        return metrics, recommendations
    
    def _load_with_trimesh(self, ply_path: str) -> trimesh.Trimesh:
        """Load PLY file using trimesh."""
        try:
            print(f"📦 Loading PLY file with trimesh: {ply_path}")
            print(f"📊 File size: {os.path.getsize(ply_path) / (1024*1024):.1f} MB")
            
            mesh = trimesh.load(ply_path)
            print(f"✅ Trimesh loaded successfully")
            
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"🔧 Converting mesh type: {type(mesh)}")
                # Handle point clouds or scenes
                if hasattr(mesh, 'geometry'):
                    geometries = list(mesh.geometry.values())
                    if geometries:
                        mesh = geometries[0]
                        print(f"✅ Extracted geometry from scene")
                else:
                    raise ValueError("No valid mesh geometry found")
            return mesh
        except Exception as e:
            print(f"❌ Warning: Failed to load with trimesh: {e}")
            return None
    
    def _calculate_quality_metrics(self, mesh: trimesh.Trimesh) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        
        print("🔧 Initializing quality metrics...")
        metrics = QualityMetrics()
        
        if mesh is None:
            print("❌ Mesh is None, returning empty metrics")
            return metrics
        
        print(f"📊 Mesh info: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Basic mesh properties
        print("🔍 Calculating basic mesh properties...")
        metrics.vertex_count = len(mesh.vertices)
        metrics.face_count = len(mesh.faces)
        print("🔍 Checking watertight status...")
        metrics.is_watertight = mesh.is_watertight
        print("🔍 Checking winding consistency...")
        metrics.is_winding_consistent = mesh.is_winding_consistent
        print("✅ Basic mesh properties calculated")
        
        # Geometric properties
        print("🔍 Calculating geometric properties...")
        try:
            print("🔍 Calculating surface area...")
            metrics.surface_area = mesh.area
            print(f"✅ Surface area: {metrics.surface_area}")
            
            if mesh.is_watertight:
                print("🔍 Calculating volume (watertight mesh)...")
                metrics.volume = abs(mesh.volume)
                print(f"✅ Volume: {metrics.volume}")
            else:
                print("⚠️ Mesh not watertight, skipping volume calculation")
                
            print("🔍 Calculating bounding box volume...")
            metrics.bounding_box_volume = mesh.bounding_box.volume
            print(f"✅ Bounding box volume: {metrics.bounding_box_volume}")
            
        except Exception as e:
            print(f"❌ Warning: Error calculating geometric properties: {e}")
        
        # Mesh quality analysis
        print("🔍 Starting mesh quality analysis...")
        metrics = self._analyze_mesh_quality(mesh, metrics)
        print("✅ Mesh quality analysis completed")
        
        # Calculate overall quality scores
        print("🔍 Calculating quality scores...")
        metrics = self._calculate_quality_scores(metrics)
        print("✅ Quality scores calculated")
        
        return metrics
    
    def _analyze_mesh_quality(self, mesh: trimesh.Trimesh, 
                            metrics: QualityMetrics) -> QualityMetrics:
        """Analyze mesh quality using trimesh."""
        
        try:
            print(f"🔍 Analyzing mesh quality for {len(mesh.faces)} faces...")
            
            # Triangle aspect ratio analysis
            if len(mesh.faces) > 0:
                print("🔍 Calculating face angles...")
                face_angles = mesh.face_angles
                print(f"✅ Face angles calculated: {len(face_angles)} faces")
                
                aspect_ratios = []
                print("🔍 Processing aspect ratios...")
                
                for i, face_angle_set in enumerate(face_angles):
                    if i % 10000 == 0 and i > 0:  # Progress update every 10k faces
                        print(f"   Processed {i}/{len(face_angles)} faces...")
                    
                    if len(face_angle_set) == 3:
                        max_angle = np.max(face_angle_set)
                        min_angle = np.min(face_angle_set)
                        aspect_ratio = max_angle / max(min_angle, 0.01)
                        aspect_ratios.append(aspect_ratio)
                
                if aspect_ratios:
                    print("🔍 Calculating aspect ratio statistics...")
                    metrics.aspect_ratio_stats = {
                        'mean': float(np.mean(aspect_ratios)),
                        'std': float(np.std(aspect_ratios)),
                        'min': float(np.min(aspect_ratios)),
                        'max': float(np.max(aspect_ratios))
                    }
                    print("✅ Aspect ratio analysis completed")
            
            # Edge length analysis
            print("🔍 Analyzing edge lengths...")
            if len(mesh.edges) > 0:
                edge_lengths = mesh.edges_unique_length
                if len(edge_lengths) > 0:
                    print("🔍 Calculating edge length statistics...")
                    metrics.edge_length_stats = {
                        'mean': float(np.mean(edge_lengths)),
                        'std': float(np.std(edge_lengths)),
                        'min': float(np.min(edge_lengths)),
                        'max': float(np.max(edge_lengths))
                    }
                    print("✅ Edge length analysis completed")
            
            # Hole detection with bottom hole analysis
            print("🔍 Detecting holes...")
            if not mesh.is_watertight:
                # Count boundary edges to estimate holes
                boundary_edges = mesh.edges[mesh.edges_unique_inverse]
                total_holes = len(boundary_edges) // 2  # Rough estimate
                
                # Detect bottom hole separately
                print("🔍 Analyzing bottom hole...")
                bottom_hole_size = self._detect_bottom_hole(mesh)
                metrics.bottom_hole_size = bottom_hole_size
                
                # Calculate actual surface holes (excluding bottom hole)
                surface_holes = max(0, total_holes - bottom_hole_size)
                metrics.hole_count = surface_holes
                metrics.total_holes = total_holes
                
                print(f"Hole detection completed:")
                print(f"Total holes: {total_holes}")
                print(f"Bottom hole size: {bottom_hole_size}")
                print(f"Surface holes: {surface_holes}")
            else:
                print("Mesh is watertight, no holes detected")
                metrics.hole_count = 0
                metrics.total_holes = 0
                metrics.bottom_hole_size = 0
            
            # Skip noise level estimation for now (it's very slow)
            print("Skipping noise level calculation (too slow for large meshes)")
            metrics.noise_level = 0.0
            
        except Exception as e:
            print(f"Warning: Error in mesh quality analysis: {e}")
        
        return metrics
    
    def _detect_bottom_hole(self, mesh) -> int:
        """Detect the size of the bottom hole in the mesh using a more accurate method."""
        try:
            # Get mesh bounds
            bounds = mesh.bounds
            min_z = bounds[0][2]  # Minimum Z coordinate
            max_z = bounds[1][2]  # Maximum Z coordinate
            
            # Define bottom region (lowest 5% of the mesh for more precision)
            bottom_threshold = min_z + (max_z - min_z) * 0.05
            
            # Find vertices in the bottom region
            bottom_vertices = mesh.vertices[:, 2] <= bottom_threshold
            
            if not np.any(bottom_vertices):
                return 0
            
            # Find faces that have at least one vertex in the bottom region
            bottom_vertex_indices = np.where(bottom_vertices)[0]
            bottom_face_mask = np.any(np.isin(mesh.faces, bottom_vertex_indices), axis=1)
            bottom_faces = mesh.faces[bottom_face_mask]
            
            if len(bottom_faces) == 0:
                return 0
            
            # Count boundary edges more accurately
            edge_counts = {}
            for face in bottom_faces:
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i+1)%3]]))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            
            # Boundary edges appear only once
            boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
            
            # Estimate hole size based on boundary edge count
            # A typical hole has roughly 3-6 edges per "hole unit"
            bottom_hole_size = max(1, len(boundary_edges) // 4)
            
            return bottom_hole_size
            
        except Exception as e:
            print(f"Warning: Could not detect bottom hole: {e}")
            return 0
    
    def _repair_mesh_holes(self, mesh, max_hole_size=1000):
        """Attempt to repair holes in the mesh using trimesh."""
        try:
            print("🔧 Attempting to repair mesh holes...")
            
            # Fill holes using trimesh
            filled_mesh = mesh.fill_holes(max_hole_size=max_hole_size)
            
            if filled_mesh is not None and len(filled_mesh.faces) > len(mesh.faces):
                print(f"Mesh repair successful: {len(filled_mesh.faces) - len(mesh.faces)} faces added")
                return filled_mesh
            else:
                print("Mesh repair did not add any faces")
                return mesh
                
        except Exception as e:
            print(f"Warning: Mesh repair failed: {e}")
            return mesh
    
    def _calculate_quality_scores(self, metrics: QualityMetrics) -> QualityMetrics:
        """Calculate overall quality scores."""
        
        scores = []
        
        # Mesh topology score (using surface holes, not total holes)
        topology_score = 1.0
        if not metrics.is_watertight:
            topology_score *= 0.7
        if not metrics.is_winding_consistent:
            topology_score *= 0.8
        if metrics.hole_count > 0:  # Only count surface holes, not bottom hole
            topology_score *= max(0.3, 1.0 - metrics.hole_count * 0.1)
        
        scores.append(topology_score)
        
        # Geometric accuracy score
        geometric_score = 1.0
        
        # Penalize high noise
        if metrics.noise_level > 0:
            geometric_score *= max(0.2, 1.0 - metrics.noise_level)
        
        # Penalize poor aspect ratios
        if metrics.aspect_ratio_stats:
            mean_aspect_ratio = metrics.aspect_ratio_stats.get('mean', 1.0)
            if mean_aspect_ratio > 3.0:  # Poor triangles
                geometric_score *= max(0.3, 1.0 - (mean_aspect_ratio - 3.0) * 0.1)
        
        scores.append(geometric_score)
        
        # Overall quality score
        metrics.overall_quality_score = np.mean(scores)
        
        # Detect critical issues
        metrics.critical_issues = []
        metrics.detected_defects = []
        
        if not metrics.is_watertight:
            metrics.detected_defects.append("Non-watertight mesh")
        if metrics.hole_count > 5:
            metrics.critical_issues.append("Excessive holes in reconstruction")
        if metrics.noise_level > 0.5:
            metrics.critical_issues.append("High noise level")
        if metrics.vertex_count < 1000:
            metrics.detected_defects.append("Low vertex density")
        if metrics.overall_quality_score < 0.4:
            metrics.critical_issues.append("Poor overall quality")
        
        return metrics
    
    def _generate_recommendations(self, metrics: QualityMetrics, 
                                mesh: trimesh.Trimesh) -> List[Dict]:
        """Generate specific improvement recommendations."""
        
        recommendations = []
        
        # Critical quality issues
        if metrics.overall_quality_score < self.quality_thresholds['poor']:
            recommendations.append({
                'category': 'scan_parameters',
                'priority': 'critical',
                'issue': 'Very poor reconstruction quality',
                'recommendation': 'Increase number of projections and reduce angular step size',
                'specific_parameters': {
                    'num_projections': 'increase by 50-100%',
                    'angular_step': 'reduce to 1-2 degrees',
                    'exposure_time': 'increase by 20-30%'
                }
            })
        
        # Hole detection recommendations (only for surface holes, not bottom hole)
        if metrics.hole_count > 3:
            recommendations.append({
                'category': 'scan_parameters',
                'priority': 'high',
                'issue': f'Multiple surface holes detected ({metrics.hole_count})',
                'recommendation': 'Increase angular coverage and projection density',
                'specific_parameters': {
                    'start_angle': 0,
                    'end_angle': 360,
                    'num_projections': max(180, metrics.hole_count * 30),
                    'overlap_ratio': 0.8
                }
            })
        
        # Bottom hole information
        if metrics.bottom_hole_size > 0:
            recommendations.append({
                'category': 'scan_parameters',
                'priority': 'info',
                'issue': f'Bottom hole detected ({metrics.bottom_hole_size} edges)',
                'recommendation': 'This is expected for objects that cannot be photographed from below',
                'specific_parameters': {
                    'note': 'Bottom hole is excluded from quality assessment',
                    'total_holes': metrics.total_holes,
                    'surface_holes': metrics.hole_count
                }
            })
        
        # Non-watertight mesh
        if not metrics.is_watertight:
            recommendations.append({
                'category': 'reconstruction_settings',
                'priority': 'high',
                'issue': 'Non-watertight mesh reconstruction',
                'recommendation': 'Adjust reconstruction algorithm parameters',
                'specific_parameters': {
                    'surface_reconstruction': 'Poisson with higher depth',
                    'hole_filling': 'enable',
                    'smoothing_iterations': 3
                }
            })
        
        return recommendations
    
    def get_quality_summary(self, metrics: QualityMetrics) -> str:
        """Generate a human-readable quality summary."""
        
        quality_level = 'poor'
        for level, threshold in self.quality_thresholds.items():
            if metrics.overall_quality_score >= threshold:
                quality_level = level
                break
        
        summary = f"""
=== 3D Reconstruction Quality Analysis ===

Overall Quality: {quality_level.upper()} ({metrics.overall_quality_score:.2f}/1.00)

Mesh Properties:
- Vertices: {metrics.vertex_count:,}
- Faces: {metrics.face_count:,}
- Watertight: {'Yes' if metrics.is_watertight else 'No'}
- Total Holes: {metrics.total_holes}
- Surface Holes: {metrics.hole_count}
- Bottom Hole: {metrics.bottom_hole_size} edges (expected)

Quality Scores:
- Overall Quality: {metrics.overall_quality_score:.2f}/1.00
- Noise Level: {metrics.noise_level:.3f}

"""
        
        if metrics.critical_issues:
            summary += "Critical Issues:\n"
            for issue in metrics.critical_issues:
                summary += f"{issue}\n"
            summary += "\n"
        
        if metrics.detected_defects:
            summary += "Detected Defects:\n"
            for defect in metrics.detected_defects:
                summary += f"🔍 {defect}\n"
        
        return summary

def find_best_ply_file(base_path: str, image_folder: str) -> str:
    """Find scene_texture.ply file in the reconstruction workspace."""
    print(f"🔍 Searching for scene_texture.ply in: {base_path}/{image_folder}")
    workspace_dir = os.path.join(base_path, image_folder, "workspace")
    dense_dir = os.path.join(workspace_dir, "dense")
    
    print(f"Workspace dir: {workspace_dir}")
    print(f"Dense dir: {dense_dir}")
    
    if not os.path.exists(dense_dir):
        raise FileNotFoundError(f"Reconstruction workspace not found: {dense_dir}")
    
    print(f"Dense directory exists")
    
    # Only look for scene_texture.ply
    ply_path = os.path.join(dense_dir, "scene_texture.ply")
    print(f"Looking for: scene_texture.ply")
    
    if os.path.exists(ply_path):
        print(f"Found scene_texture.ply")
        return ply_path
    else:
        # List files for debugging
        try:
            files_in_dense = os.listdir(dense_dir)
            print(f"Files in dense directory: {files_in_dense}")
        except Exception as e:
            print(f"Error listing dense directory: {e}")
        
        raise FileNotFoundError(f"scene_texture.ply not found in {dense_dir}. Only scene_texture.ply is supported.")

def analyze_reconstruction_quality(image_folder: str, base_path: str = "/home/user/tmpData/AI_scan/") -> dict:
    """
    Complete quality analysis of reconstruction results.
    
    Args:
        image_folder: Name of the image folder (scan name)
        base_path: Base path where reconstructions are stored
        
    Returns:
        Dictionary with complete analysis results
    """
    try:
        print(f"🔍 Finding best PLY file for: {image_folder}")
        # Find the best PLY file
        ply_path = find_best_ply_file(base_path, image_folder)
        ply_filename = os.path.basename(ply_path)
        
        print(f"Found PLY file: {ply_filename}")
        print(f"Full path: {ply_path}")
        
        # Initialize analyzer and run analysis
        print("Initializing PLY quality analyzer...")
        analyzer = PLYQualityAnalyzer()
        print("Starting PLY file analysis...")
        metrics, recommendations = analyzer.analyze_ply_file(ply_path)
        print("PLY file analysis completed")
        
        # Generate quality summary
        quality_summary = analyzer.get_quality_summary(metrics)
        
        # Create comprehensive results dictionary
        results = {
            "analysis_success": True,
            "ply_file_path": ply_path,
            "ply_filename": ply_filename,
            "quality_metrics": {
                "vertex_count": metrics.vertex_count,
                "face_count": metrics.face_count,
                "is_watertight": metrics.is_watertight,
                "is_winding_consistent": metrics.is_winding_consistent,
                "surface_area": metrics.surface_area,
                "volume": metrics.volume,
                "bounding_box_volume": metrics.bounding_box_volume,
                "hole_count": metrics.hole_count,
                "noise_level": metrics.noise_level,
                "overall_quality_score": metrics.overall_quality_score,
                "aspect_ratio_stats": metrics.aspect_ratio_stats,
                "edge_length_stats": metrics.edge_length_stats
            },
            "quality_assessment": {
                "overall_quality_score": metrics.overall_quality_score,
                "quality_level": "poor",
                "critical_issues": metrics.critical_issues or [],
                "detected_defects": metrics.detected_defects or []
            },
            "recommendations": recommendations,
            "quality_summary_text": quality_summary,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Determine quality level
        for level, threshold in analyzer.quality_thresholds.items():
            if metrics.overall_quality_score >= threshold:
                results["quality_assessment"]["quality_level"] = level
                break
        
        print(f"Quality analysis completed. Overall score: {metrics.overall_quality_score:.2f}")
        print(f"Quality level: {results['quality_assessment']['quality_level']}")
        
        return results
        
    except Exception as e:
        error_results = {
            "analysis_success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "analysis_timestamp": datetime.now().isoformat()
        }
        print(f"Quality analysis failed: {e}")
        return error_results

def run_quality_analysis_with_metadata(image_folder: str, base_path: str = "/home/user/tmpData/AI_scan/", repair_holes: bool = False) -> dict:
    """
    Complete quality analysis with structured metadata output for Bluesky integration.
    
    Args:
        image_folder: Name of the image folder (scan name)
        base_path: Base path where reconstructions are stored
        
    Returns:
        Dictionary with complete analysis results and metadata
    """
    try:
        print(f"Starting quality analysis for: {image_folder}")
        print(f"Base path: {base_path}")
        
        # Run the quality analysis
        print("Calling analyze_reconstruction_quality...")
        analysis_results = analyze_reconstruction_quality(image_folder, base_path)
        print("analyze_reconstruction_quality completed")
        
        # Optionally repair holes and re-analyze
        if repair_holes and analysis_results.get("analysis_success", False):
            print("epairing holes and re-analyzing...")
            try:
                # Get the PLY file path
                ply_path = analysis_results.get("ply_file_path")
                if ply_path and os.path.exists(ply_path):
                    # Load mesh and repair
                    mesh = trimesh.load(ply_path)
                    if not mesh.is_watertight:
                        print("Attempting to repair mesh holes...")
                        repaired_mesh = mesh.fill_holes(max_hole_size=1000)
                        if repaired_mesh is not None and len(repaired_mesh.faces) > len(mesh.faces):
                            print(f"Mesh repair successful: {len(repaired_mesh.faces) - len(mesh.faces)} faces added")
                            # Save repaired mesh
                            repaired_path = ply_path.replace('.ply', '_repaired.ply')
                            repaired_mesh.export(repaired_path)
                            print(f"Repaired mesh saved to: {repaired_path}")
                            
                            # Re-analyze the repaired mesh
                            print("🔍 Re-analyzing repaired mesh...")
                            repaired_results = analyze_reconstruction_quality(image_folder, base_path, repaired_path)
                            if repaired_results.get("analysis_success", False):
                                analysis_results["repaired_analysis"] = repaired_results
                                print("Repaired mesh analysis completed")
                        else:
                            print("Mesh repair did not improve the mesh")
            except Exception as e:
                print(f"Warning: Mesh repair failed: {e}")
        
        # Add additional metadata for Bluesky integration
        analysis_results.update({
            "bluesky_integration": True,
            "analyzed_directory": image_folder,
            "base_path": base_path,
            "analysis_completed": datetime.now().isoformat()
        })
        
        # Print summary for console output
        if analysis_results["analysis_success"]:
            quality_score = analysis_results.get("quality_assessment", {}).get("overall_quality_score", 0)
            quality_level = analysis_results.get("quality_assessment", {}).get("quality_level", "unknown")
            vertex_count = analysis_results.get("quality_metrics", {}).get("vertex_count", 0)
            
            print(f"\n{'='*60}")
            print(f"3D RECONSTRUCTION QUALITY ANALYSIS")
            print(f"{'='*60}")
            print(f"Directory: {image_folder}")
            print(f"PLY File: {analysis_results.get('ply_filename', 'Unknown')}")
            print(f"Quality Level: {quality_level.upper()}")
            print(f"Quality Score: {quality_score:.2f}/1.00")
            print(f"Vertices: {vertex_count:,}")
            print(f"Faces: {analysis_results.get('quality_metrics', {}).get('face_count', 0):,}")
            print(f"Watertight: {'Yes' if analysis_results.get('quality_metrics', {}).get('is_watertight', False) else 'No'}")
            print(f"Total Holes: {analysis_results.get('quality_metrics', {}).get('total_holes', 0)}")
            print(f"Surface Holes: {analysis_results.get('quality_metrics', {}).get('hole_count', 0)}")
            bottom_hole = analysis_results.get('quality_metrics', {}).get('bottom_hole_size', 0)
            if bottom_hole > 0:
                print(f"Bottom Hole: {bottom_hole} edges (expected)")
            
            # Show critical issues if any
            critical_issues = analysis_results.get("quality_assessment", {}).get("critical_issues", [])
            if critical_issues:
                print(f"\nCRITICAL ISSUES:")
                for issue in critical_issues:
                    print(f"   • {issue}")
            
            # Show recommendations
            recommendations = analysis_results.get("recommendations", [])
            if recommendations:
                print(f"\nRECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. [{rec['priority'].upper()}] {rec['issue']}")
                    print(f"      → {rec['recommendation']}")
            
            print(f"{'='*60}")
            
            # Return success summary
            summary = f"Quality analysis completed: {quality_level} quality (score: {quality_score:.2f}, {vertex_count:,} vertices)"
            analysis_results["summary"] = summary
            print(f"{summary}")
            
        else:
            error_msg = f"Quality analysis failed: {analysis_results.get('error', 'Unknown error')}"
            analysis_results["summary"] = error_msg
            print(f"{error_msg}")
        
        return analysis_results
        
    except Exception as e:
        error_results = {
            "analysis_success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "bluesky_integration": True,
            "analyzed_directory": image_folder,
            "base_path": base_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": f"Quality analysis failed with exception: {e}"
        }
        print(f"Script failed: {e}")
        return error_results

if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime
    
    if len(sys.argv) < 2:
        print("Usage: python ply_quality_analyzer.py <image_folder_name> [--repair-holes]")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    repair_holes = "--repair-holes" in sys.argv
    
    try:
        results = run_quality_analysis_with_metadata(image_folder, repair_holes=repair_holes)
        
        if not results["analysis_success"]:
            sys.exit(1)
            
    except Exception as e:
        print(f"Script failed: {e}")
        sys.exit(1)
