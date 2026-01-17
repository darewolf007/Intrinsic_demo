from time import time
from common.logger import timeit
import numpy as np
import torch
from termcolor import colored
from tensordict.tensordict import TensorDict
from functools import partial
from copy import deepcopy

from common.discriminator import Discriminator
from trainer.base import Trainer

class Demo3Trainer(Trainer):
    """Trainer class for DEMO3 training. Assumes semi-sparse reward environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.env.reward_mode in [
            "semi_sparse"
        ], "Reward mode is incompatible with DEMO3"

        self._step = 0
        self._pretrain_step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._alpha = 1

        self.disc = Discriminator(
            self.env,
            self.cfg.discriminator,
            state_shape=(self.cfg.latent_dim,),
            compile=self.cfg.compile,
        )

        print("Discriminator Architecture:", self.disc)
        print(
            "Learnable parameters: {:,}".format(
                self.agent.model.total_params + self.disc.total_params
            )
        )

    # [NEW] RND 权重调度器：实现后期探索
    def get_rnd_scale(self):
        """
        Calculates the current RND intrinsic reward scale based on training steps.
        Logic: 
          - Before 'start_step': 0.0 (Focus on imitation/task)
          - After 'start_step': Linearly ramp up to 'max_scale' over 'ramp_steps'
        """
        start_step = getattr(self.cfg, 'rnd_start_step', self.cfg.steps // 3) 
        ramp_steps = getattr(self.cfg, 'rnd_ramp_steps', self.cfg.steps // 3)
        max_scale = getattr(self.cfg, 'intrinsic_scale', 0.5) 
        
        if self._step < start_step:
            return 0.0
        
        # 线性增长
        progress = (self._step - start_step) / ramp_steps
        current_scale = max_scale * min(1.0, max(0.0, progress))
        
        return current_scale

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self, pretrain=False):
        """Evaluate agent."""
        ep_rewards, ep_max_rewards, ep_successes, ep_seeds = [], [], [], []
        for i in range(max(1, self.cfg.eval_episodes // self.cfg.num_envs)):
            seed = np.random.randint(2**31)
            obs, done, ep_reward, ep_max_reward, t = (
                self.env.reset(seed=seed),
                torch.tensor(False),
                0,
                None,
                0,
            )
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=True)
            while not done.any():
                torch.compiler.cudagraph_mark_step_begin()
                action = (
                    self.agent.policy_action(obs, eval_mode=True)
                    if pretrain
                    else self.agent.act(obs, t0=t == 0, eval_mode=True)
                )
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_max_reward = (
                    torch.maximum(ep_max_reward, reward)
                    if ep_max_reward is not None
                    else reward
                )
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            assert (
                done.all()
            ), "Vectorized environments must reset all environments at once."
            ep_rewards.append(ep_reward)
            ep_max_rewards.append(ep_max_reward)
            ep_successes.append(info["success"].float().mean())
            ep_seeds.append(seed)

        if self.cfg.save_video:
            if pretrain:
                self.logger.video.save(
                    "pretrain/iteration",
                    self._pretrain_step,
                    key="videos/pretrain_video",
                )
            else:
                self.logger.video.save("eval/step", self._step)

        eval_metrics = dict(
            episode_reward=torch.cat(ep_rewards).mean(),
            episode_max_reward=torch.cat(ep_max_rewards).max(),
            episode_success=torch.stack(ep_successes).mean(),
            best_seed=(
                ep_seeds[torch.argmax(torch.stack(ep_rewards).mean(dim=1)).item()]
                if pretrain
                else None
            ),
        )

        stage_success = {
            f"stage_{s}_success": ((torch.cat(ep_max_rewards) >= s).float().mean())
            for s in range(1, self.env.n_stages + 1)
        }
        eval_metrics.update(stage_success)

        return eval_metrics

    def to_td(self, obs, action=None, reward=None, device="cpu"):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=())
        else:
            obs = obs.unsqueeze(0)
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan")).repeat(self.cfg.num_envs)
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(
                1,
                self.cfg.num_envs,
            ),
        )
        return td.to(torch.device(device))

    def pretrain(self):
        """Pretrains agent policy with demonstration data"""
        demo_buffer = self.buffer._offline_buffer
        n_iterations = self.cfg.pretrain.n_epochs
        self.cfg.pretrain.eval_freq = n_iterations // 25
        start_time = time()
        best_model, best_score = deepcopy(self.agent.bc_model.state_dict()), 0

        print(
            colored(
                f"Policy pretraining: {n_iterations} iterations", "red", attrs=["bold"]
            )
        )

        self.agent.bc_model.train()
        for self._pretrain_step in range(n_iterations):
            metrics = self.agent.init_bc(demo_buffer)

            if self._pretrain_step % self.cfg.pretrain.eval_freq == 0:
                eval_metrics = self.eval(pretrain=True)
                eval_metrics.update({"iteration": self._pretrain_step})
                self.logger.log(eval_metrics, category="pretrain")

                if eval_metrics["episode_reward"] > best_score:
                    best_model = deepcopy(self.agent.bc_model.state_dict())
                    best_score = eval_metrics["episode_reward"]
                    best_seed = eval_metrics["best_seed"]

            if self._pretrain_step % self.cfg.pretrain.log_freq == 0:
                metrics.update(
                    {
                        "iteration": self._pretrain_step,
                        "total_time": time() - start_time,
                    }
                )
                self.logger.log(metrics, category="pretrain")

        eval_metrics = self.eval(pretrain=True)
        eval_metrics.update({"iteration": self._pretrain_step})
        self.logger.log(eval_metrics, category="pretrain")

        if best_score == 0:
            best_model = deepcopy(self.agent.bc_model.state_dict())
            best_seed = eval_metrics["best_seed"]

        self.agent.model.eval()
        self.agent.model.load_state_dict(best_model)
        self.agent.bc_model.load_state_dict(best_model)
        self.seed_scheduler.start(init_seed=best_seed, max_seeds=1e4)

    def train(self):
        """Train agent and discriminator"""

        # Policy pretraining
        if self.cfg.get("policy_pretraining", False):
            self.pretrain()

        # Start interactive training
        print(colored("\nReplay buffer seeding", "yellow", attrs=["bold"]))
        train_metrics, done, eval_next = {}, torch.tensor(True), False

        # [NEW] 输出探索调度信息
        start_step = getattr(self.cfg, 'rnd_start_step', self.cfg.steps // 3)
        max_scale = getattr(self.cfg, 'intrinsic_scale', 0.5)

        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Save Disc and Agent periodically
            if self._step % self.cfg.save_freq == 0 and self._step > 0:
                print("Saving agent and discriminator checkpoints...")
                self.logger.save_agent(self.disc, identifier=f"disc_{self._step}")
                self.logger.save_agent(self.agent, identifier=f"agent_{self._step}")

            # Reset environment
            if done.any():
                assert done.all(), "Vectorized environments must reset all environments at once."
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    tds = torch.cat(self._tds)
                    tds["stage"] = (
                        torch.ones_like(tds["reward"])
                        * np.nanmax(tds["reward"], axis=0)
                    ).int()
                    self._ep_idx = self.buffer.add(tds)
                    train_metrics.update(
                        episode_reward=np.nansum(tds["reward"], axis=0).mean(),
                        episode_max_reward=np.nanmax(tds["reward"], axis=0).max(),
                        episode_success=info["success"].float().nanmean(),
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    self.seed_scheduler.step(train_metrics["episode_success"].item())

                obs = self.env.reset(seed=self.seed_scheduler.sample())
                self._tds = [self.to_td(obs, device="cpu")]

            self._alpha = (
                max(0, self.cfg.max_bc_steps - self._step) / self.cfg.max_bc_steps
            )

            # -----------------------------
            # [MODIFIED] 关键：在采样动作前，把 rnd_scale 写入 agent（影响 MPC 规划）
            # -----------------------------
            if self._step >= self.cfg.seed_steps:
                current_rnd_scale = self.get_rnd_scale()
            else:
                current_rnd_scale = 0.0
            self.agent.set_rnd_scale(current_rnd_scale)
            # -----------------------------

            # Collect experience
            if self._step > self.cfg.seed_steps:
                if np.random.random() < self._alpha and self.cfg.get("policy_pretraining", False):
                    # 仍保留 BC 混合（原样）
                    action = self.agent.policy_action(obs, eval_mode=True)
                else:
                    # ==========================
                    # [MODIFIED] 训练阶段 MPC:Policy = 1:1 交替
                    # ==========================
                    # 用全局 step 计算交替位（每次循环 step 增加 num_envs）
                    k = (self._step // self.cfg.num_envs)
                    use_mpc = (k % 2 == 0)

                    if use_mpc:
                        action = self.agent.act(obs, t0=len(self._tds) == 1)          # MPC
                    else:
                        action = self.agent.act_policy(obs, eval_mode=False)          # Policy
                    # ==========================

            elif self.cfg.get("policy_pretraining", False):
                action = self.agent.policy_action(obs, eval_mode=True)
            else:
                action = self.env.rand_act()

            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward, device="cpu"))

            # Update discriminator and agent
            # self.cfg.seed_steps = 2000
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = max(1, int(self.cfg.seed_steps / self.cfg.steps_per_update))
                    print(colored("\nTraining DEMO3 Agent", "green", attrs=["bold"]))
                    print(f"Pretraining agent with {num_updates} update steps on seed data...")
                else:
                    num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))

                # [MODIFIED] 复用上面采样时的 current_rnd_scale（保持一致）
<<<<<<< HEAD
                # ============================================================
                # 修改 train 方法中的更新循环
                # 找到 for _ in range(num_updates): 部分并修改
                # ============================================================

=======
>>>>>>> 2e5d980918e17bccceccde5b04b0c253c54ba7b7
                for _ in range(num_updates):
                    disc_train_metrics = self.disc.update(
                        self.buffer,
                        encoder_function=partial(self.agent.model.encode, task=None),
                    )
<<<<<<< HEAD
                    
                    # [ENSEMBLE] 获取当前disc_loss并传递给agent
                    current_disc_loss = disc_train_metrics.get("discriminator_loss", 0.3)
                    if isinstance(current_disc_loss, torch.Tensor):
                        current_disc_loss = current_disc_loss.item()
                    self.agent.set_disc_loss(current_disc_loss)
                    
=======
>>>>>>> 2e5d980918e17bccceccde5b04b0c253c54ba7b7
                    agent_train_metrics = self.agent.update(
                        self.buffer,
                        modify_reward=self.disc.get_reward,
                        action_penalty=self.cfg.action_penalty,
                        rnd_scale=current_rnd_scale,
                    )
                train_metrics.update(disc_train_metrics)
                train_metrics.update(agent_train_metrics)
                train_metrics.update({"rnd_schedule_value": current_rnd_scale})
                # [ENSEMBLE] 记录不确定性系数
                train_metrics.update({"uncertainty_coef": self.agent._get_uncertainty_coef()})

            self._step += self.cfg.num_envs

        self.logger.finish(self.agent)