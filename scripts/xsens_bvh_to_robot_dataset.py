# xsens_bvh_to_robot_dataset.py
import argparse
import pathlib
import os
import multiprocessing as mp
import mujoco as mj
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting.utils.xsens import load_xsens_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
import re
HERE = pathlib.Path(__file__).parent

def process_file(bvh_file_path, tgt_file_path, tgt_robot, tgt_folder, total_files, override=False):
    if os.path.exists(tgt_file_path) and not override:
        print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
        return
    # print(f"Target file {tgt_file_path}...")
    # print(f"Processing {bvh_file_path}...")
    pattern = r"^(.*)/[^/]*\.bvh$"
    match = re.match(pattern, bvh_file_path)
    if match:
        path = match.group(1) + "/"
        # print(path)  # 输出：xsens_bvh/251020/
    else:
        print("No match")

    if os.path.exists(path+"offsets.json"):
        offsets_file = path+"offsets.json"
    else:
        offsets_file = None
    try:
        # 加载 XSens BVH 文件
        args = argparse.Namespace(
            bvh_file=bvh_file_path,
            scale=0.01,
            reset_to_zero=False,
            start=None,
            end=None,
            bvh_format="3DSM",
            offsets_file=offsets_file,
        )
        lafan1_data_frames, actual_human_height, frame_time = load_xsens_file(args)
        src_fps = int(1 / frame_time)  # 计算 FPS
    except Exception as e:
        print(f"Error loading {bvh_file_path}: {e}")
        return

    # 初始化 retargeting 系统
    if tgt_robot == "Q1":
        if os.path.exists(path+"xsens_bvh_to_Q1.json"):
            extern_ik_config_path = path+"xsens_bvh_to_Q1.json"
            print("use extern_ik_config_path:", extern_ik_config_path)
        else:
            extern_ik_config_path = None
    elif tgt_robot == "unitree_g1":
        if os.path.exists(path+"xsens_bvh_to_Q1.json"):
            extern_ik_config_path = path+"xsens_bvh_to_g1.json"
            print("use extern_ik_config_path:", extern_ik_config_path)
        else:
            extern_ik_config_path = None
    retarget = GMR(
        src_human="xsens_bvh",
        tgt_robot=tgt_robot,
        actual_human_height=actual_human_height,
        extern_ik_config_path=extern_ik_config_path,
    )
    model = mj.MjModel.from_xml_path(retarget.xml_file)
    data = mj.MjData(model)

    # retarget 所有帧的 qpos
    qpos_list = []
    for curr_frame in range(len(lafan1_data_frames)):
        smplx_data = lafan1_data_frames[curr_frame]

        # 执行 retargeting
        qpos = retarget.retarget(smplx_data)

        qpos_list.append(qpos.copy())

    qpos_list = np.array(qpos_list)

    # 初始化前向 kinematics
    device = "cuda:0"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)

    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]  # 调整根旋转顺序
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]

    # 计算 local body pos
    identity_root_pos = torch.zeros((num_frames, 3), device=device)
    identity_root_rot = torch.zeros((num_frames, 4), device=device)
    identity_root_rot[:, -1] = 1.0
    local_body_pos, _ = kinematics_model.forward_kinematics(
        identity_root_pos, 
        identity_root_rot, 
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )
    body_names = kinematics_model.body_names

    # 可选的高度调整（与参考脚本一致，默认关闭）
    HEIGHT_ADJUST = False
    PERFRAME_ADJUST = False
    if HEIGHT_ADJUST:
        body_pos, _ = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
            torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        )
        ground_offset = 0.00
        # if not PERFRAME_ADJUST:
        #     lowest_height = torch.min(body_pos[..., 2]).item()
        #     root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
        # else:
        #     for i in range(root_pos.shape[0]):
        #         lowest_body_part = torch.min(body_pos[i, :, 2])
        #         root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

    motion_data = {
        "fps": src_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
    }

    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)

    # 进度打印
    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('.pkl')])
    print(f"Processed {done}/{total_files}: {tgt_file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing XSens BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../motion_data/XSENS_gmr",
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        default="unitree_h1_2",
        type=str,
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--num_cpus",
        default=4,
        type=int,
    )
    
    args = parser.parse_args()
    
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if not filename.endswith(".bvh"):
                continue
            bvh_file_path = os.path.join(dirpath, filename)
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".pkl")
            args_list.append((bvh_file_path, tgt_file_path, args.robot, tgt_folder))
    
    print("args_list:", len(args_list))
    
    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")
    
    with mp.Pool(args.num_cpus) as pool:
        pool.starmap(process_file, [arg + (total_files, args.override) for arg in args_list])

    print("Done. Saved to ", tgt_folder)

if __name__ == "__main__":
    main()
