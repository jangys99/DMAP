import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_m.envs.mixins import HistoryMixin


class FrankaGraspEnv(gym.Env, HistoryMixin):
    def __init__(self, render=False, include_adapt_state=False, num_memory_steps=30):
        super(FrankaGraspEnv, self).__init__()
        self.render_mode = render
        
        if self.render_mode:
            self._p = p.connect(p.GUI)
        else:
            self._p = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.action_dim = 7
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        # 14(joint pos+vel) + 3(EE pos) + 3(Box pos)
        self.obs_dim = 20 
        
        # Mixin 호환성: self.robot이 self를 가리키도록 설정
        self.robot = self
        
        # [DMAP 필수] 환경 변화 요소를 정의합니다. (예: 물체 질량)
        self.perturbation_list = ["box_mass"] 
        self.current_perturb = {}
        
        # [중요] Mixin 초기화는 obs_dim 설정 등 모든 준비가 끝난 뒤 마지막에 호출
        # 이 함수가 self.observation_space를 Dict 형태로 자동 변환합니다.
        self._init_addon(include_adapt_state, num_memory_steps)
        
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # 1. 바닥 생성
        p.loadURDF("plane.urdf")
        
        # 2. 테이블 생성 (높이 조정)
        table_height = 0.625
        self.tableId = p.loadURDF("table/table.urdf", [0.5, 0, 0])
        
        # 3. Franka Panda 로봇을 테이블 위에 고정
        # 로봇의 Z 좌표를 테이블 높이(table_height)와 동일하게 설정하여 상판 위에 얹습니다.
        # 로봇 위치: [0, 0, 0.625] (테이블의 한쪽 끝)
        robot_pos = [0.0, 0.0, table_height]
        robot_orn = p.getQuaternionFromEuler([0, 0, 0]) # 회전 없음
        self.pandaId = p.loadURDF("franka_panda/panda.urdf", robot_pos, robot_orn, useFixedBase=True)
        
        # 4. Box 생성 (테이블 위)
        # 박스가 테이블에 파묻히지 않도록 Z축을 살짝 더 올려줍니다 (+0.05)
        # 박스 질량 랜덤화 및 위치 설정
        mass_perturb = np.random.uniform(0.3, 1.0)
        self.current_perturb = {"box_mass": mass_perturb}
        
        self.boxId = p.loadURDF("cube_small.urdf", [0.5, 0, table_height + 0.05])
        p.changeDynamics(self.boxId, -1, mass=mass_perturb)
        
        # 초기 관절 상태 설정
        self.rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        for i in range(7):
            p.resetJointState(self.pandaId, i, self.rest_poses[i])
            
        raw_obs = self._get_obs()
        return self.create_rma_reset_state(raw_obs)

    def step(self, action):
        scaled_action = action * 0.05
        current_joint_states = p.getJointStates(self.pandaId, range(7))
        target_positions = [state[0] + act for state, act in zip(current_joint_states, scaled_action)]
        
        p.setJointMotorControlArray(
            self.pandaId, range(7), p.POSITION_CONTROL, targetPositions=target_positions
        )
        
        p.stepSimulation()
        
        raw_obs = self._get_obs()
        reward, done, info = self._compute_reward_done()
        
        # [DMAP] 상태 반환
        obs_dict = self.create_rma_step_state(raw_obs, action)        
        
        return obs_dict, reward, done, info

    def _get_obs(self):
        joint_states = p.getJointStates(self.pandaId, range(7))
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

    # [DMAP 필수] Mixin이 호출하는 함수 구현
    def get_current_perturb(self):
        return np.array([self.current_perturb[key] for key in self.perturbation_list])

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()