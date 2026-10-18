"""Build a COLMAP-style 3DGS scene whose initial points are ETH3D laser scans.

Images and camera poses stay as released. Each scan is transformed by the
4x4 matrix in scan_alignment.mlp, then the merged cloud is subsampled.
"""

import argparse
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def scene_dir(name):
    base = os.path.join(ROOT, "data", name)
    direct = os.path.join(base, name)
    if os.path.isdir(os.path.join(direct, "scan_clean")):
        return direct
    if os.path.isdir(os.path.join(base, "scan_clean")):
        return base
    for child in os.listdir(base):
        candidate = os.path.join(base, child)
        if os.path.isdir(os.path.join(candidate, "scan_clean")):
            return candidate
    raise FileNotFoundError(f"no scan_clean under {base}")


def parse_alignment(path):
    tree = ET.parse(path)
    poses = {}
    for mesh in tree.iter("MLMesh"):
        filename = mesh.attrib["filename"]
        text = mesh.find("MLMatrix44").text.split()
        matrix = np.array([float(v) for v in text], dtype=np.float64).reshape(4, 4)
        poses[filename] = matrix
    if not poses:
        raise RuntimeError(f"no MLMesh entries in {path}")
    return poses


def read_scan_ply(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"missing end_header in {path}")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break
        count = None
        properties = []
        in_vertex = False
        for line in header_lines:
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element ") and in_vertex:
                break
            if in_vertex and line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))
        if count is None:
            raise RuntimeError(f"no vertex element in {path}")
        fields = []
        for typ, name in properties:
            if typ == "float":
                fields.append((name, "<f4"))
            elif typ == "uchar":
                fields.append((name, "u1"))
            else:
                raise RuntimeError(f"unsupported property {typ} {name} in {path}")
        data = np.fromfile(f, dtype=np.dtype(fields), count=count)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def apply_pose(xyz, matrix):
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([xyz, ones], axis=1)
    return (matrix @ hom.T).T[:, :3]


def write_points_ply(path, xyz, rgb):
    dtype = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    out = np.empty(xyz.shape[0], dtype=dtype)
    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]
    out["nx"] = 0
    out["ny"] = 0
    out["nz"] = 0
    out["red"] = rgb[:, 0]
    out["green"] = rgb[:, 1]
    out["blue"] = rgb[:, 2]
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header\n",
    ])
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(out.tobytes())


def link_images(src, dst):
    if os.path.lexists(dst):
        os.remove(dst) if not os.path.isdir(dst) else None
        if os.path.isdir(dst):
            subprocess.check_call(["cmd", "/c", "rmdir", dst])
    subprocess.check_call(["cmd", "/c", "mklink", "/J", dst, src])


def prepare(name, target_points, seed):
    src = scene_dir(name)
    dst = os.path.join(ROOT, "data", "prepared", f"{name}_lidar")
    sparse = os.path.join(dst, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)

    calib = os.path.join(src, "dslr_calibration_undistorted")
    shutil.copy2(os.path.join(calib, "cameras.txt"), os.path.join(sparse, "cameras.txt"))
    shutil.copy2(os.path.join(calib, "images.txt"), os.path.join(sparse, "images.txt"))
    link_images(os.path.join(src, "images"), os.path.join(dst, "images"))

    poses = parse_alignment(os.path.join(src, "scan_clean", "scan_alignment.mlp"))
    chunks_xyz = []
    chunks_rgb = []
    for filename, matrix in poses.items():
        ply_path = os.path.join(src, "scan_clean", filename)
        xyz, rgb = read_scan_ply(ply_path)
        xyz = apply_pose(xyz, matrix)
        print(f"{filename}: {xyz.shape[0]} points")
        chunks_xyz.append(xyz)
        chunks_rgb.append(rgb)

    xyz = np.concatenate(chunks_xyz, axis=0)
    rgb = np.concatenate(chunks_rgb, axis=0)
    print(f"merged: {xyz.shape[0]} points")
    if xyz.shape[0] > target_points:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(xyz.shape[0], target_points, replace=False)
        xyz = xyz[chosen]
        rgb = rgb[chosen]
    write_points_ply(os.path.join(sparse, "points3D.ply"), xyz, rgb)
    print(f"wrote {xyz.shape[0]} points to {os.path.join(sparse, 'points3D.ply')}")
    print(f"scene: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="courtyard")
    parser.add_argument("--points", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(args.scene, args.points, args.seed)


if __name__ == "__main__":
    main()
