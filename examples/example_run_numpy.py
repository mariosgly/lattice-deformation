import argparse
import numpy as np
import open3d as o3d
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.numpy_bspline import bspline_2x2x2_deform_fast


def main(json_path, mesh_path, output_path):
    # (1) Read the JSON
    with open(json_path, 'r') as f:
        lattice_deformations = json.load(f)['lattice_deformations']

    # (2) Read a mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)

    # Example: scale from mm->m
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    original_vertices = vertices.copy()

    # (3) Build bounding box
    box_min = original_vertices.min(axis=0) - 1e-12
    box_max = original_vertices.max(axis=0) + 1e-12

    print("box_min =", box_min, "box_max =", box_max)
    print("box_size =", (box_max - box_min))

    # (4) Initialize corners_offset for the 8 corners
    corners_offset = {}
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                corners_offset[(i, j, k)] = np.array([0.0, 0.0, 0.0])

    # Fill corners_offset from JSON
    for idx_str, corner_data in lattice_deformations.items():
        orig_np = np.array(corner_data["original"])      # e.g. [0.5, -0.5, -0.5]
        deform_np = np.array(corner_data["deformation"]) # e.g. [-0.0023, -0.5887, ...]
        # Convert corner space [-0.5..+0.5] => (0 or 1)
        i, j, k = (orig_np + 0.5).astype(int)
        real_offset = deform_np * (box_max - box_min)
        print("Corner:", (i, j, k), " Deform:", real_offset)
        corners_offset[(i, j, k)] = real_offset

    # (5) Deform using the *vectorized* B-spline interpolation for 2x2x2
    deformed_vertices = bspline_2x2x2_deform_fast(
        original_vertices,
        box_min, box_max,
        corners_offset
    )

    # (6) Write out
    mesh.vertices = o3d.utility.Vector3dVector(deformed_vertices)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print("B-spline (2x2x2) Deformation (Vectorized) complete! Saved to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B-spline 2x2x2 deformation script")
    parser.add_argument("--json", required=True, help="Path to lattice deformation JSON file")
    parser.add_argument("--mesh", required=True, help="Path to input mesh file")
    parser.add_argument("--output", required=True, help="Path to save deformed mesh file")
    
    args = parser.parse_args()
    
    main(args.json, args.mesh, args.output)
