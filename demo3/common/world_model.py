from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from common import layers, math, init
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self.register_buffer(
                "_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim)
            )
            for i in range(len(cfg.tasks)):
                self._action_masks[i, : cfg.action_dims[i]] = 1.0
        self._encoder = layers.enc(cfg)
        # [ENSEMBLE] 创建5个独立的dynamics网络
        self.num_ensemble = getattr(cfg, 'num_ensemble', 5)
        self._dynamics_ensemble = nn.ModuleList([
            layers.mlp(
                cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                2 * [cfg.mlp_dim],
                cfg.latent_dim,
                act=layers.SimNorm(cfg),
            )
            for _ in range(self.num_ensemble)
        ])
        # 为每个ensemble成员使用不同的初始化
        for i, dyn in enumerate(self._dynamics_ensemble):
            self._init_ensemble_member(dyn, seed=i * 1000)
        self._reward = layers.mlp(
            cfg.latent_dim + cfg.action_dim + cfg.task_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        self._pi = layers.mlp(
            cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim
        )
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer(
            "log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min
        )
        self.init()

    def _init_ensemble_member(self, module, seed):
        """为ensemble成员使用不同seed初始化，确保多样性"""
        torch.manual_seed(seed)
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(
            self._Qs.params.data.clone(), no_convert=True
        )

        # Create modules
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules
        self._detach_Qs.params = self._detach_Qs_params
        self._target_Qs.params = self._target_Qs_params

    def __repr__(self):
        repr = "TD-MPC2 World Model\n"
        modules = ["Encoder", "Dynamics (Ensemble)", "Reward", "Policy prior", "Q-functions"]
        for i, m in enumerate(
            [self._encoder, self._dynamics_ensemble, self._reward, self._pi, self._Qs]
        ):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if isinstance(obs, (dict, TensorDict)):
            out = {}
            for k, enc in self._encoder.items():
                if k.startswith("rgb") and obs[k].ndim == 5:
                    out[k] = torch.stack([enc(o) for o in obs[k]])
                else:
                    out[k] = enc(obs[k])
            return torch.stack([out[k] for k in out.keys()]).mean(0)
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task, ensemble_idx=None):
        """
        Predicts the next latent state given the current latent state and action.
        ensemble_idx: 如果指定，只使用该索引的网络；否则使用均值
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z_a = torch.cat([z, a], dim=-1)
        
        if ensemble_idx is not None:
            # 使用指定的ensemble成员
            return self._dynamics_ensemble[ensemble_idx](z_a)
        else:
            # 使用所有ensemble的均值
            predictions = torch.stack([dyn(z_a) for dyn in self._dynamics_ensemble], dim=0)
            return predictions.mean(dim=0)

    def next_with_uncertainty(self, z, a, task):
        """
        预测下一状态并返回不确定性估计
        
        Returns:
            mean: 预测均值 [batch_size, latent_dim]
            uncertainty: 不确定性 (pairwise disagreement) [batch_size]
            all_predictions: 所有预测 [num_ensemble, batch_size, latent_dim]
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z_a = torch.cat([z, a], dim=-1)
        
        # 获取所有ensemble的预测
        all_predictions = torch.stack([dyn(z_a) for dyn in self._dynamics_ensemble], dim=0)
        # shape: [num_ensemble, batch_size, latent_dim]
        
        mean = all_predictions.mean(dim=0)
        
        # 计算pairwise disagreement作为不确定性
        num_ensemble = all_predictions.shape[0]
        pairwise_diffs = []
        for i in range(num_ensemble):
            for j in range(i + 1, num_ensemble):
                diff = (all_predictions[i] - all_predictions[j]).pow(2).mean(dim=-1)
                pairwise_diffs.append(diff)
        
        uncertainty = torch.stack(pairwise_diffs, dim=0).mean(dim=0).sqrt()
        # shape: [batch_size]
        
        return mean, uncertainty, all_predictions

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2