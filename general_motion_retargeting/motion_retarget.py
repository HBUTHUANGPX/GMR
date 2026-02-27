import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print


class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR)."""

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str = "daqp",  # change from "quadprog" to "daqp".
        damping: float = 5e-1,  # change from 1e-1 to 1e-2.
        verbose: bool = False,
        use_velocity_limit: bool = False,
        extern_ik_config_path: str = None,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        # Print DoF names in order
        # print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i]
            )
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")

        # print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        # print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        if extern_ik_config_path is not None:
            ik_config_path = extern_ik_config_path
        else:
            ik_config_path = IK_CONFIG_DICT[src_human][tgt_robot]
        with open(ik_config_path) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", ik_config_path)

        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
        print(f"actual_human_height:{actual_human_height}")
        print(f"human_height_assumption:{ik_config['human_height_assumption']}")
        print(f"ratio:{ratio}")
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = (
                ik_config["human_scale_table"][key] * ratio
            )
        if "human_scale_table_2" in ik_config:
            for key, entry in ik_config["human_scale_table_2"].items():
                if isinstance(entry, dict) and "scale" in entry:
                    entry["scale"] = entry["scale"] * ratio

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.human_scale_table_2 = ik_config.get("human_scale_table_2")
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3 * np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

        self.setup_retarget_configuration()

        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks1 = []
        self.tasks2 = []

        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []

        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data_2(
            human_data, self.human_root_name, self.human_scale_table_2
        )
        # human_data = self.scale_human_data(
        #     human_data, self.human_root_name, self.human_scale_table
        # )
        human_data = self.offset_human_data(
            human_data, self.pos_offsets1, self.rot_offsets1
        )
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )

        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )

    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration,
                self.tasks1,
                dt,
                self.solver,
                self.damping,
                self.ik_limits,
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration,
                    self.tasks1,
                    dt,
                    self.solver,
                    self.damping,
                    self.ik_limits,
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration,
                self.tasks2,
                dt,
                self.solver,
                self.damping,
                self.ik_limits,
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration,
                    self.tasks2,
                    dt,
                    self.solver,
                    self.damping,
                    self.ik_limits,
                )
                self.configuration.integrate_inplace(vel2, dt)

                next_error = self.error2()
                num_iter += 1

        return self.configuration.data.qpos.copy()

    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )

    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )

    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [
                np.asarray(human_data[body_name][0]),
                np.asarray(human_data[body_name][1]),
            ]
        return human_data

    def scale_human_data(self, human_data, human_root_name, human_scale_table):

        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]

        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos

        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (
                    human_data[body_name][0] - root_pos
                ) * human_scale_table[body_name]

        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (
                human_data_local[body_name] + scaled_root_pos,
                human_data[body_name][1],
            )

        return human_data_global

    def scale_human_data_2(self, human_data, human_root_name, human_scale_table):

        # 新格式：按父子链进行缩放（子节点相对父节点的向量进行缩放）
        scaled_pos = {}
        root_pos, root_quat = human_data[human_root_name]
        root_entry = human_scale_table.get(human_root_name, {})
        root_scale = (
            root_entry.get("scale", 1.0) if isinstance(root_entry, dict) else root_entry
        )
        # root 的父节点视为 world，只缩放 root 的全局位置
        scaled_pos[human_root_name] = root_pos * root_scale

        visiting = set()

        def resolve_scaled_pos(body_name):
            # 已经计算过则直接返回
            if body_name in scaled_pos:
                return scaled_pos[body_name]
            if body_name in visiting:
                # cycle detected, fall back to original position
                scaled_pos[body_name] = human_data[body_name][0]
                return scaled_pos[body_name]
            visiting.add(body_name)

            # 没有配置就原样返回
            entry = human_scale_table.get(body_name)
            if entry is None:
                scaled_pos[body_name] = human_data[body_name][0]
                visiting.remove(body_name)
                return scaled_pos[body_name]

            # 支持两种格式：
            # 1) dict: {"scale": x, "parent": "ParentName"}
            # 2) number: 仅 scale，parent 默认为 root
            if isinstance(entry, dict):
                parent = entry.get("parent", human_root_name)
                scale = entry.get("scale", 1.0)
            else:
                parent = human_root_name
                scale = entry

            # parent 为空或为 "world" 时，父节点在世界坐标原点
            if parent is None or str(parent).lower() == "world":
                parent_pos = np.zeros(3)
                parent_scaled = np.zeros(3)
            elif parent not in human_data:
                # 父节点在数据里不存在，回退为原位置
                scaled_pos[body_name] = human_data[body_name][0]
                visiting.remove(body_name)
                return scaled_pos[body_name]
            else:
                # 先递归得到父节点的缩放后位置
                parent_pos = human_data[parent][0]
                parent_scaled = resolve_scaled_pos(parent)

            # 子节点相对父节点的向量
            vec = human_data[body_name][0] - parent_pos
            # 缩放相对向量并加回父节点的缩放后位置
            scaled_pos[body_name] = parent_scaled + vec * scale
            visiting.remove(body_name)
            return scaled_pos[body_name]

        # 计算所有关节的缩放后位置（保留原姿态）
        for body_name in human_data.keys():
            resolve_scaled_pos(body_name)

        human_data_global = {}
        for body_name, (pos, quat) in human_data.items():
            # 姿态不变，仅替换位置
            if body_name not in human_scale_table:
                continue
            human_data_global[body_name] = (scaled_pos.get(body_name, pos), quat)

        return human_data_global

    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (
                R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]
            ).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat

            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(
                local_offset
            )

            offset_human_data[body_name][0] = pos + global_pos_offset

        return offset_human_data

    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = (
                pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
            )
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
