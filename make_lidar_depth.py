"""Project ETH3D laser scans into undistorted cameras as inverse-depth maps.

The maps are metric 1/z in the COLMAP camera frame. Empty pixels stay 0 and
are masked out of the depth loss. Scale is 1 because both sides are metric.
"""

import argparse
import json
import os

import numpy as np

from prepare_eth3d import apply_pose, parse_alignment, read_scan_ply, scene_dir

ROOT = os.path.dirname(os.path.abspath(__file__))


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def read_cameras(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            cameras[int(parts[0])] = {
                "width": int(parts[2]),
                "height": int(parts[3]),
                "fx": float(parts[4]),
                "fy": float(parts[5]),
                "cx": float(parts[6]),
                "cy": float(parts[7]),
            }
    return cameras


def read_poses(path):
    poses = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            f.readline()
            poses.append({
                "q": np.array(list(map(float, parts[1:5])), dtype=np.float64),
                "t": np.array(list(map(float, parts[5:8])), dtype=np.float64),
                "camera_id": int(parts[8]),
                "name": parts[9],
            })
    return poses


def train_wh(width, height):
    scale = width / 1600.0 if width > 1600 else 1.0
    return int(width / scale), int(height / scale), scale


def load_world_points(scan_root):
    poses = parse_alignment(os.path.join(scan_root, "scan_alignment.mlp"))
    chunks = []
    for filename, matrix in poses.items():
        xyz, _ = read_scan_ply(os.path.join(scan_root, filename))
        chunks.append(apply_pose(xyz, matrix))
        print(f"{filename}: {chunks[-1].shape[0]} points")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def project(points, pose, camera):
    width, height, scale = train_wh(camera["width"], camera["height"])
    rot = qvec2rotmat(pose["q"])
    cam = (rot @ points.T).T + pose["t"].astype(np.float32)
    z = cam[:, 2]
    fx = camera["fx"] / scale
    fy = camera["fy"] / scale
    cx = camera["cx"] / scale
    cy = camera["cy"] / scale
    valid = z > 0.2
    u = np.full(z.shape, -1, dtype=np.int32)
    v = np.full(z.shape, -1, dtype=np.int32)
    u[valid] = np.floor(fx * (cam[valid, 0] / z[valid]) + cx).astype(np.int32)
    v[valid] = np.floor(fy * (cam[valid, 1] / z[valid]) + cy).astype(np.int32)
    inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(zbuf, (v[inside], u[inside]), z[inside])
    inv = np.zeros((height, width), dtype=np.float32)
    hit = np.isfinite(zbuf)
    inv[hit] = 1.0 / zbuf[hit]
    return inv, int(hit.sum()), width * height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="pipes")
    args = parser.parse_args()
    prepared = os.path.join(ROOT, "data", "prepared", f"{args.scene}_lidar")
    sparse = os.path.join(prepared, "sparse", "0")
    scan_root = os.path.join(scene_dir(args.scene), "scan_clean")
    points = load_world_points(scan_root)
    print(f"world points: {points.shape[0]}")
    cameras = read_cameras(os.path.join(sparse, "cameras.txt"))
    poses = read_poses(os.path.join(sparse, "images.txt"))
    params = {}
    for pose in poses:
        inv, hits, pixels = project(points, pose, cameras[pose["camera_id"]])
        stem = pose["name"].rsplit(".", 1)[0]
        out_path = os.path.join(prepared, "depths", stem + ".npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, inv)
        params[stem] = {"scale": 1.0, "offset": 0.0}
        print(f"{pose['name']}: {hits}/{pixels} pixels ({100.0 * hits / pixels:.2f}%)")
    with open(os.path.join(sparse, "depth_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"wrote {len(poses)} depth maps")


if __name__ == "__main__":
    main()
