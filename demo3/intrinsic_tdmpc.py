import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from tensordict import TensorDict

# [工具类] 用于在线标准化内在奖励
class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-4, clip=10.0):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(1e-4))
        self.epsilon = epsilon
        self.clip = clip

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    def forward(self, x):
        # Normalize and clip
        return ((x - self.mean) / torch.sqrt(self.var + self.epsilon)).clamp(-self.clip, self.clip)

# [RND 模块]
class RND(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Target network: 固定参数，随机初始化
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Predictor network: 训练去拟合 Target
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 冻结 Target 网络参数
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return predict_feature, target_feature

class Intrinsic_TDMPC2(torch.nn.Module):
    """
    TD-MPC2 agent with RND Intrinsic Motivation (Decoupled Reward + Stable Inputs).
    """

    def __init__(self, cfg):
        print("intrinsic tdmpc init")
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        self.model = WorldModel(cfg).to(self.device)
        self.bc_model = WorldModel(cfg).to(self.device)
        
        # [MODIFIED] 1. 计算 Raw State 的总维度并初始化投影层
        # TD-MPC2 的 cfg.obs_shape 是个字典，例如 {'state': [39]}
        self.obs_keys = list(cfg.obs_shape.keys())
        rnd_input_dim = 0
        for k, v in cfg.obs_shape.items():
            if k.startswith("state"):
                rnd_input_dim += v[0]
        
        # 简单的安全检查，防止全是像素输入导致 Linear 报错
        if rnd_input_dim == 0:
            print("[Warning] No 'state' keys found in obs_shape for RND Linear Projection. Using latent_dim fallback.")
            # 如果全是图像，暂时回退到用 latent dim (虽然这可能导致 collapse，但至少能跑)
            # 更好的做法是用随机 CNN，但改动太大。
            self.use_raw_obs_for_rnd = False
            rnd_input_dim = cfg.latent_dim
        else:
            self.use_raw_obs_for_rnd = True

        # 将 Raw State 投影到 512 维，这是一个固定的随机投影
        self.rnd_input_proj = nn.Linear(rnd_input_dim, 512, bias=False).to(self.device)
        for param in self.rnd_input_proj.parameters():
            param.requires_grad = False
            
        # [NEW] Initialize RND Module (Input is now 512 from projection)
        rnd_out_dim = getattr(cfg, 'rnd_out_dim', 128)
        self.rnd = RND(512, 512, rnd_out_dim).to(self.device)
        
        # [MODIFIED] 2. 降低 RND 学习率
        self.rnd_optim = torch.optim.Adam(self.rnd.predictor.parameters(), lr=self.cfg.lr * 0.1)
        
        # [NEW] Initialize Reward Normalizer
        self.rnd_r_norm = RunningMeanStd(shape=(1,), epsilon=1e-4).to(self.device)
        
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Qs.parameters()},
                {
                    "params": (
                        self.model._task_emb.parameters() if self.cfg.multitask else []
                    )
                },
            ],
            lr=self.cfg.lr,
            capturable=True,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True
        )
        self.bc_optim = torch.optim.Adam(self.bc_model.parameters(), lr=self.cfg.lr)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2 * int(
            cfg.action_dim >= 20
        )
        self.discount = (
            torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
                device="cuda:0",
            )
            if self.cfg.multitask
            else self._get_discount(cfg.episode_length)
        )
        self._prev_mean = torch.nn.Buffer(
            torch.zeros(
                self.cfg.num_envs,
                self.cfg.horizon,
                self.cfg.action_dim,
                device=self.device,
            )
        )
        if cfg.compile:
            print("compiling - tdmpc update")
            self._update = torch.compile(self._update, mode="reduce-overhead")
            print("compiling - bc update")
            self._init_bc = torch.compile(self._init_bc, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if False:  # self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        if self.cfg.discount_hardcoded != 0:
            return self.cfg.discount_hardcoded
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
        )

    def save(self, fp):
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    def init_bc(self, buffer):
        obs, action, reward, task = buffer.sample(return_td=False)
        torch.compiler.cudagraph_mark_step_begin()
        return self._init_bc(obs, action, reward, task)

    def _init_bc(self, obs, action, rew, task):
        self.bc_optim.zero_grad(set_to_none=True)
        a = self.bc_model.pi(self.bc_model.encode(obs[:-1], task), task)[0]
        loss = F.mse_loss(a, action, reduce=True)
        loss.backward()
        self.bc_optim.step()
        self.model.load_state_dict(self.bc_model.state_dict())
        metrics = (TensorDict({"bc_loss": loss}).detach().mean())
        return metrics

    @torch.no_grad()
    def policy_action(self, obs, eval_mode=False, task=None):
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.bc_model.encode(obs, task)
        a = self.bc_model.pi(z, task)[int(not eval_mode)]
        return a.cpu()

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            a = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
        else:
            z = self.model.encode(obs, task)
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(
                self.model.reward(z, actions[:, t], task), self.cfg
            )
            z = self.model.next(z, actions[:, t], task)
            G = G + discount * reward
            discount_update = (
                self.discount[torch.tensor(task)]
                if self.cfg.multitask
                else self.discount
            )
            discount = discount * discount_update
        return G + discount * self.model.Q(
            z, self.model.pi(z, task)[1], task, return_type="avg"
        )

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        # Sample policy trajectories
        b_size = obs.shape[0]
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                b_size,
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            _z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[:, t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:, t], task)
            pi_actions[:, -1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
        mean = torch.zeros(
            b_size, self.cfg.horizon, self.cfg.action_dim, device=self.device
        )
        std = torch.full(
            (b_size, self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
            device=self.device,
        )
        if not t0:
            mean[:, :-1] = self._prev_mean[:, 1:]
        actions = torch.empty(
            b_size,
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            actions[:, :, : self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):
            # Sample actions
            r = torch.randn(
                b_size,
                self.cfg.horizon,
                self.cfg.num_samples - self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=std.device,
            )
            actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, :, self.cfg.num_pi_trajs :] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(
                value.squeeze(2), self.cfg.num_elites, dim=1
            ).indices
            elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2))
            elite_actions = torch.gather(
                actions,
                2,
                elite_idxs.unsqueeze(1)
                .unsqueeze(3)
                .expand(-1, self.cfg.horizon, -1, self.cfg.action_dim),
            )

            # Update parameters
            max_value = elite_value.max(1).values
            score = torch.exp(
                self.cfg.temperature * (elite_value - max_value.unsqueeze(1))
            )
            score = score / score.sum(1, keepdim=True)
            mean = (score.unsqueeze(1) * elite_actions).sum(dim=2) / (
                score.sum(1, keepdim=True) + 1e-9
            )
            std = (
                (score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2).sum(
                    dim=2
                )
                / (score.sum(1, keepdim=True) + 1e-9)
            ).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action sequence
        rand_idx = math.gumbel_softmax_sample(
            score.squeeze(-1), dim=1
        )
        actions = torch.stack(
            [elite_actions[i, :, rand_idx[i]] for i in range(rand_idx.shape[0])], dim=0
        )
        a, std = actions[:, 0], std[:, 0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    def update_pi(self, zs, task):
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type="avg", detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)
        return pi_loss.detach(), pi_grad_norm

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        pi = self.model.pi(next_z, task)[1]
        discount = (
            self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        )
        return reward + discount * self.model.Q(
            next_z, pi, task, return_type="min", target=True
        )

    def set_rnd_scale(self, scale: float):
        self._rnd_scale_plan = float(scale)

    @torch.no_grad()
    def act_policy(self, obs, eval_mode=False, task=None):
        """
        Policy-dominant sampling action:
        Always use self.model.pi (NOT MPC), even if cfg.mpc=True.
        """
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        # pi(...) returns (mu, pi, log_pi, log_std)
        mu, pi, _, _ = self.model.pi(z, task)
        a = mu if eval_mode else pi
        return a.cpu()

    # [MODIFIED] Update loop 接收 rnd_scale 并解耦奖励
    def _update(
        self, obs, action, reward, task=None, modify_reward=None, action_penalty=False, rnd_scale=0.0, update_dynamics=True, update_policy=True
    ):
        # Compute targets
        with torch.no_grad():
            ori_reward = reward.clone()
            next_z = self.model.encode(obs[1:], task)
            
            # [Step 1] 应用 Discriminator 奖励 (Demo3 逻辑)
            # 这个奖励用于告诉 Agent "什么是专家行为"，World Model 需要学习它
            if modify_reward:
                reward = modify_reward(next_z, reward)

            if action_penalty:
                penalty = torch.linalg.norm(action, ord=2, dim=-1, keepdim=True).pow(2) / (action.shape[-1] * 5)
                reward -= penalty

            # [CRITICAL] 锁定 Reward Model 的学习目标
            # World Model 必须学会预测 (Extrinsic + Discriminator)
            reward_for_model = reward.clone()

            # [Step 2] 计算 RND 内在奖励并归一化
            # 仅当 rnd_scale > 0 时计算，节省计算资源
            norm_intr_reward = 0.0
            raw_intr_mean = 0.0
            
            if rnd_scale > 0:
                # [MODIFIED] 准备 RND 输入
                stage1_threshold = 1.0
                gate_mask = (ori_reward >= stage1_threshold)  # bool
                gate_mask = gate_mask.cummax(dim=0).values   # bool：达到一次后持续开启
                gate_mask = gate_mask.float()

                if self.use_raw_obs_for_rnd:
                    # 使用原始 Raw State (从 obs 字典中提取并拼接)
                    next_obs_dict = obs[1:] # T+1 的观测
                    state_list = []
                    for k in self.obs_keys:
                        if k.startswith("state"):
                            state_list.append(next_obs_dict[k])
                    
                    # 拼接 shape: [Horizon, Batch, Total_State_Dim]
                    rnd_input_raw = torch.cat(state_list, dim=-1)
                    
                    # 投影到特征空间
                    rnd_input = self.rnd_input_proj(rnd_input_raw)
                else:
                    # 回退到 Latent (如果全是 RGB)
                    rnd_input = next_z

                # RND 前向
                pred_feat, target_feat = self.rnd(rnd_input)
                
                # raw_intr_reward shape: [Horizon, Batch, 1]
                raw_intr_reward = (pred_feat - target_feat).pow(2).mean(dim=-1, keepdim=True)
                
                # 记录原始均值用于 Logging
                raw_intr_mean = raw_intr_reward.mean().item()
                masked_raw = raw_intr_reward[gate_mask.bool()]
                # [FIXED] 展平后更新 Normalizer，解决维度报错
                if masked_raw.numel() > 0:
                    self.rnd_r_norm.update(masked_raw.reshape(-1, 1))
                
                # [MODIFIED] 归一化后 Clamp 到非负
                # 解决“负奖励”导致 Stage 3 无法突破的问题
                norm_intr_reward_tensor = self.rnd_r_norm(raw_intr_reward).clamp(min=0.0)
                norm_intr_reward_tensor = norm_intr_reward_tensor * gate_mask
                # [CRITICAL] 仅将 RND 加到 Critic 的目标上
                total_reward_for_critic = reward + rnd_scale * norm_intr_reward_tensor.detach()
            else:
                total_reward_for_critic = reward

            # [Step 3] 计算 TD Targets
            # Critic 使用包含探索奖励的总分来更新 Value
            td_targets = self._td_target(next_z, total_reward_for_critic, task)

        # Prepare for update
        self.model.train()
        # 只有需要探索时才开启 RND 训练模式
        if rnd_scale > 0:
            self.rnd.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0.0
        rnd_loss = 0.0
        
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            if update_dynamics:
                consistency_loss = (
                    consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
                )
            zs[t + 1] = z
            
            # [MODIFIED] 计算 RND Loss
            if rnd_scale > 0 and update_dynamics:
                # 准备当前时间步的 RND 输入
                stage1_threshold = 1.0
                gate_t = gate_mask[t]
                if self.use_raw_obs_for_rnd:
                    curr_obs_dict = obs[t+1] # 这里的 obs 已经是 batch 切片过的吗？注意 unbind
                    # Wait, obs[1:] was unbound above? No. 
                    # obs 是 TensorDict [H+1, B]. 
                    # 循环中我们已经有了 _next_z (Target). 但我们需要 _next_raw_obs
                    # 正确做法：直接从 obs[t+1] 取
                    
                    state_list = []
                    for k in self.obs_keys:
                        if k.startswith("state"):
                            state_list.append(obs[k][t+1]) # 取第 t+1 个时间步
                    
                    curr_rnd_input_raw = torch.cat(state_list, dim=-1)
                    with torch.no_grad():
                         curr_rnd_input = self.rnd_input_proj(curr_rnd_input_raw)
                else:
                    curr_rnd_input = _next_z.detach()

                curr_pred, curr_target = self.rnd(curr_rnd_input) 
                # 拟合固定 target
                per_loss = (curr_pred - curr_target.detach()).pow(2).mean(dim=-1, keepdim=True)  # [B,1]
                rnd_loss = rnd_loss + (per_loss * gate_t).mean()
        # Predictions
        _zs = zs[:-1]
        
        # 只有更新 Policy 时才需要算 Q，只有更新 Dynamics 时才需要算 Reward Predictor
        # 但 TD-MPC 是一起算的，为了代码简单，我们算出来，但在 Loss 加和时控制
        qs = self.model.Q(_zs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)

        # Compute losses
        reward_loss, value_loss = 0.0, 0.0
        
        # 只有在需要相关 Loss 时才进行循环计算，节省计算图
        if update_dynamics or update_policy:
            for t, (rew_pred_unbind, rew_model_unbind, qs_unbind) in enumerate(
                zip(
                    reward_preds.unbind(0),
                    reward_for_model.unbind(0),
                    qs.unbind(1),
                )
            ):
                # 1. Reward Prediction Loss (属于 World Model)
                if update_dynamics:
                    reward_loss = (
                        reward_loss
                        + math.soft_ce(rew_pred_unbind, rew_model_unbind, self.cfg).mean()
                        * self.cfg.rho**t
                    )
                
                # 2. Value Loss (属于 Critic/Policy)
                if update_policy:
                    td_targets_unbind = td_targets[t] # 注意：td_targets 需要对应索引
                    for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                        value_loss = (
                            value_loss
                            + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean()
                            * self.cfg.rho**t
                        )

        # Normalize Losses
        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        if isinstance(rnd_loss, torch.Tensor):
            rnd_loss = rnd_loss / self.cfg.horizon

        # [CRITICAL] 组合 Total Loss
        total_loss = 0.0
        
        # 仅当 update_dynamics 为真，才优化 consistency 和 reward prediction
        if update_dynamics:
            total_loss += self.cfg.consistency_coef * consistency_loss
            total_loss += self.cfg.reward_coef * reward_loss
            
        # 仅当 update_policy 为真，才优化 value loss
        # 注意：Value Loss 通常也会回传梯度给 Encoder (Representation Learning)。
        # 如果你希望 Policy 更新完全不影响 World Model (包括 Encoder)，你需要在这里 detach _zs
        # 但通常我们希望 Value 信号也能优化 Encoder，只是不要优化 Dynamics (Next state prediction)。
        # 由于我们这里没有加 consistency loss，Dynamics 网络本身不会被 Value Loss 更新，
        # 只有 Encoder 会被 Value Loss 共同更新，这通常是可以接受甚至有益的。
        if update_policy:
            total_loss += self.cfg.value_coef * value_loss

        # Update model
        self.optim.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        
        # [NEW] Update RND Predictor separately
        if rnd_scale > 0 and update_dynamics:
            self.rnd_optim.zero_grad()
            rnd_loss.backward()
            self.rnd_optim.step()
        pi_loss, pi_grad_norm = 0.0, 0.0
        if update_policy:
            # 只有在 Policy 阶段才更新 Actor
            pi_loss, pi_grad_norm = self.update_pi(zs.detach(), task)
            # Update target Q-functions
            self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        
        # [NEW] Enhanced Logging Dictionary
        # logs = {
        #     "consistency_loss": consistency_loss,
        #     "reward_loss": reward_loss,
        #     "value_loss": value_loss,
        #     "pi_loss": pi_loss,
        #     "total_loss": total_loss,
        #     "grad_norm": grad_norm,
        #     "pi_grad_norm": pi_grad_norm,
        #     "pi_scale": self.scale.value,
        # }
        logs = {
            "total_loss": total_loss,
        }        
        # if rnd_scale > 0:
        #     logs.update({
        #         "rnd_loss": rnd_loss,
        #         "intr_reward_raw_mean": raw_intr_mean, # 原始 RND 误差均值
        #         "intr_reward_weighted_mean": (rnd_scale * norm_intr_reward_tensor).mean(), # 实际加到 Q 值里的奖励均值
        #     })
            
        return TensorDict(logs).detach().mean()

    def update(self, buffer, **kwargs):
        reward_weights = {
            0.0: 1.0,   # 初始阶段
            1.0: 1.0,
            2.0: 4.0,
            3.0: 8.0,   # 成功阶段
        }
        obs, action, reward, task = buffer.sample()
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
        self._update(obs, action, reward, update_dynamics=True, update_policy=False, **kwargs)

        obs, action, reward, task = buffer.sample_reward_weighted(reward_weights)
        if task is not None:
            kwargs["task"] = task
        # self._update(obs, action, reward)
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, update_dynamics=False, update_policy=True, **kwargs)