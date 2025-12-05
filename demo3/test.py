import gymnasium as gym
import torch
import mani_skill.envs
env = gym.make(
    "TurnFaucet-v1",
    obs_mode="rgbd",
    control_mode="pd_ee_delta_pose",
    num_envs=1,
    reward_mode="dense",
    render_mode="rgb_array",
    sensor_configs=dict(width=64, height=64),
    human_render_camera_configs=dict(width=384, height=384),
    reconfiguration_freq=1,
    sim_backend="gpu",
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs['sensor_data']['base_camera']['rgb'].shape) 
    print("Reward:", reward,type(reward))
    done = terminated or truncated
env.close()


# import gymnasium as gym
# import mani_skill.envs


# # 3. 创建环境并传入配置
# env = gym.make(
#     "PickCube-v1",
#     obs_mode='rgbd',
#     env_kwargs={  # 尝试使用 env_kwargs 包装配置
#         'camera_cfgs': {'base_camera': {'width': 64, 'height': 64}}
#     },
#     # ... 其他参数 ...
# )

# # 打印观测空间的 RGB 图像形状进行验证
# # 注意：最终形状取决于环境返回的维度顺序 (H, W, C)
# rgb_shape = env.observation_space['image']['base_camera']['rgb'].shape
# print(f"新的 RGB 观测空间形状: {rgb_shape}")

# # 假设 HWC 顺序，输出应为 (64, 64, 3)