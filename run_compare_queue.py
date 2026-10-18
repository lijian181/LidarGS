# -*- coding: utf-8 -*-
"""Finish the six ETH3D lidar scenes that never completed.

Order is increasing chance of filling 16 GB, with the 4M cap on every run.
Each scene gets the pipes control (lidar init + inverse-depth L1) and the
same run plus --depth_gate, so the two can be compared.
"""

import os
import subprocess
import time

ROOT = r"E:\3dgs_ladia"
PY = r"E:\gaussian-splatting-windows\envs\gaussian_splatting\python.exe"
LOG_DIR = os.path.join(ROOT, "output", "logs")
STATUS = os.path.join(ROOT, "output", "STATUS.txt")

# Least likely to exhaust VRAM first. Courtyard already reached 5.06M by
# iteration 7000 with no cap. Meadow has the largest camera extent.
SCENES = ["office", "kicker", "electro", "delivery", "meadow", "courtyard"]


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def child_env():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    prefix = ";".join([
        r"E:\gaussian-splatting-windows\envs\gaussian_splatting",
        r"E:\gaussian-splatting-windows\envs\gaussian_splatting\Library\bin",
        r"E:\gaussian-splatting-windows\envs\gaussian_splatting\Lib\site-packages\torch\lib",
    ])
    env["PATH"] = prefix + ";" + env.get("PATH", "")
    env["PYTHONPATH"] = r"E:\3dgs_ladia\submodules\diff-gaussian-rasterization\build\lib.win-amd64-cpython-37"
    return env


def run(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=child_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def vertex_count(ply):
    with open(ply, "rb") as handle:
        chunk = handle.read(4096)
    text = chunk.decode("ascii", errors="ignore")
    key = "element vertex "
    start = text.find(key)
    if start < 0:
        return "unknown"
    end = text.find("\n", start)
    return text[start + len(key):end].strip()


def final_ply(model):
    return os.path.join(model, "point_cloud", "iteration_30000", "point_cloud.ply")


def failed_oom(log_path):
    if not os.path.isfile(log_path):
        return False
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read().lower()
    return "out of memory" in text


def ensure_depths(scene):
    prepared = os.path.join(ROOT, "data", "prepared", scene + "_lidar")
    params = os.path.join(prepared, "sparse", "0", "depth_params.json")
    if os.path.isfile(params):
        log("DEPTH %s already exists" % scene)
        return True
    log_path = os.path.join(LOG_DIR, scene + "_depth.log")
    log("DEPTH %s" % scene)
    code = run([PY, "-u", os.path.join(ROOT, "make_lidar_depth.py"), "--scene", scene], log_path)
    if code != 0 or not os.path.isfile(params):
        log("DEPTH FAIL %s exit=%s" % (scene, code))
        return False
    log("DEPTH DONE %s" % scene)
    return True


def train_one(scene, tag, gate):
    model = os.path.join(ROOT, "output", "%s_lidar_4m_%s" % (scene, tag))
    ply = final_ply(model)
    if os.path.isfile(ply):
        log("SKIP %s %s gaussians=%s" % (scene, tag, vertex_count(ply)))
        return True
    if os.path.isdir(model):
        log("RESET incomplete %s" % model)
        import shutil
        shutil.rmtree(model)
    log_path = os.path.join(LOG_DIR, "%s_4m_%s.log" % (scene, tag))
    cmd = [
        PY, "-u", os.path.join(ROOT, "train.py"),
        "-s", os.path.join(ROOT, "data", "prepared", scene + "_lidar"),
        "-m", model,
        "-d", "depths",
        "--disable_viewer",
        "--max_gaussians", "4000000",
        "--save_iterations", "7000", "30000",
        "--test_iterations", "7000", "30000",
    ]
    if gate:
        cmd.append("--depth_gate")
    log("TRAIN %s %s" % (scene, tag))
    code = run(cmd, log_path)
    if os.path.isfile(ply):
        log("DONE %s %s exit=%s gaussians=%s bytes=%s" % (
            scene, tag, code, vertex_count(ply), os.path.getsize(ply)))
        return True
    reason = "OOM" if failed_oom(log_path) else "FAIL"
    log("%s %s %s exit=%s see %s" % (reason, scene, tag, code, log_path))
    return False


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log("COMPARE QUEUE start low-risk to high-risk: " + ", ".join(SCENES))
    for scene in SCENES:
        if not ensure_depths(scene):
            continue
        train_one(scene, "depth", gate=False)
        train_one(scene, "depth_gate", gate=True)
    log("COMPARE QUEUE finished")


if __name__ == "__main__":
    main()
