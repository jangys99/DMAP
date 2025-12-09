import os
import time
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_m.envs.mixins import HistoryMixin
from pybullet_m.helpers.xml_generator import perturb_franka_urdf
from definitions import ROOT_DIR


class FrankaGraspEnv(gym.Env, HistoryMixin):
    def __init__(self, render=False, include_adapt_state=True, num_memory_steps=30):
        super(FrankaGraspEnv, self).__init__()
        self.render_mode = render
        
        if self.render_mode:
            self._p = p.connect(p.GUI)
        else:
            self._p = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())        
        
        self.action_dim = 9
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10]
        # 14(joint pos+vel) + 3(EE pos) + 3(Box pos)
        self.obs_dim = 24
        
        self.robot = self
        # perturbation 추가
        self.perturbation_list = ["link_length", "link_thickness", "box_mass"]
        self.current_perturb = {}
        
        # urdf 경로
        self.base_urdf_path = os.path.join(ROOT_DIR, 'pybullet_m', 'xmls', 'franka', 'franka_panda', 'panda.urdf')
        self.temp_dir = os.path.dirname(self.base_urdf_path)        
        
        self._init_addon(include_adapt_state, num_memory_steps)
        
        # 중요: 초기화 직후에 e_t 범위를 무한대로 확장하는 코드 추가
        if include_adapt_state:
             self.observation_space.spaces["e_t"] = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(len(self.perturbation_list),), 
                dtype=np.float32
            )
             
        # 초기화 시 reset 호출
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        p.loadURDF("plane.urdf")
        
        table_height = 0.625
        self.tableId = p.loadURDF("table/table.urdf", [0.5, 0, 0])
        
        
        # perturbation 추가
        length_scale = 0.8
        thickness_scale = 0.8
        # mass_perturb = np.random.uniform(0.3, 1.0)

        self.current_perturb = {
            'link_length': length_scale,
            'link_thickness': thickness_scale,
            #'box_mass': mass_perturb
        }
        
        temp_urdf_name = f"panda_perturbed_{time.time()}.urdf"
        temp_urdf_path = os.path.join(self.temp_dir, temp_urdf_name)
        
        perturb_franka_urdf(
            self.base_urdf_path, 
            temp_urdf_path, 
            length_scale=length_scale, 
            thickness_scale=thickness_scale
        )
        
        
        robot_pos = [0.0, 0.0, table_height]
        robot_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.pandaId = p.loadURDF(temp_urdf_path, robot_pos, robot_orn, useFixedBase=True)
        # self.pandaId = p.loadURDF("franka_panda/panda.urdf", robot_pos, robot_orn, useFixedBase=True)
        
        self.boxId = p.loadURDF("cube_small.urdf", [0.6, 0, table_height + 0.05])
        #p.changeDynamics(self.boxId, -1, mass=mass_perturb)
        
        if os.path.exists(temp_urdf_path):
            os.remove(temp_urdf_path)
        
        self.rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.04, 0.04]
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.pandaId, joint_idx, self.rest_poses[i])

        raw_obs = self._get_obs()
        return self.create_rma_reset_state(raw_obs)

    def step(self, action):
        scaled_action = action * 0.05
        current_joint_states = p.getJointStates(self.pandaId, self.joint_indices)
        current_positions = [state[0] for state in current_joint_states]
        
        target_positions = [pos + act for pos, act in zip(current_positions, scaled_action)]
        p.setJointMotorControlArray(
            self.pandaId, 
            self.joint_indices,  # range(9) -> self.joint_indices
            p.POSITION_CONTROL, 
            targetPositions=target_positions
        )
        
        p.stepSimulation()
        
        raw_obs = self._get_obs()
        reward, done, info = self._compute_reward_done()
        
        obs_dict = self.create_rma_step_state(raw_obs, action)        
        
        return obs_dict, reward, done, info

    def _get_obs(self):
        joint_states = p.getJointStates(self.pandaId, self.joint_indices)
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]
        ee_state = p.getLinkState(self.pandaId, 11)
        ee_pos = ee_state[0]
        box_pos, _ = p.getBasePositionAndOrientation(self.boxId)
        
        return np.concatenate([joint_positions, joint_velocities, ee_pos, box_pos])

    def _compute_reward_done(self):
        ee_pos = p.getLinkState(self.pandaId, 11)[0]
        box_pos = p.getBasePositionAndOrientation(self.boxId)[0]
        distance = np.linalg.norm(np.array(ee_pos) - np.array(box_pos))
        
        reward = -distance
        if box_pos[2] > 0.9: 
            reward += 100
            done = True
        else:
            done = False
        return reward, done, {}

    def get_current_perturb(self):
        # 기본 환경에서는 perturbation_list에 있는 것만 반환
        return np.array([self.current_perturb.get(key, 0.0) for key in self.perturbation_list])

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()