"""Generic RL building blocks (buffers, agents, schedules, logging)."""

from __future__ import annotations

import math
import os
import contextlib
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(f"Torch is required for jormungandr: {exc}")


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.pos = 0
        self.full = False

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, 1), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, next_obs, done) -> None:
        idx = self.pos
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self.priorities[idx] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int, beta: float):
        if len(self) == 0:
            raise ValueError("Replay buffer is empty")
        probs = self.priorities[: len(self)] ** self.alpha
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            probs.fill(1.0 / len(self))
        else:
            probs /= total
        idxs = np.random.choice(len(self), batch_size, p=probs)

        weights = (len(self) * probs[idxs]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        batch = (
            self.obs[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_obs[idxs],
            self.done[idxs],
        )
        return batch, idxs, weights

    def update_priorities(self, idxs, priorities) -> None:
        priorities = np.abs(priorities).astype(np.float32).reshape(-1)
        if priorities.size == 0:
            return
        priorities = np.maximum(priorities, np.float32(1e-6))
        self.priorities[idxs] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "pos": self.pos,
            "full": self.full,
            "obs": self.obs,
            "next_obs": self.next_obs,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "priorities": self.priorities,
            "max_priority": self.max_priority,
        }

    @classmethod
    def load(cls, state: dict) -> "PrioritizedReplayBuffer":
        buf = cls(state["capacity"], state["obs"].shape[1], state["alpha"])
        buf.pos = state["pos"]
        buf.full = state["full"]
        buf.obs = state["obs"]
        buf.next_obs = state["next_obs"]
        buf.action = state["action"]
        buf.reward = state["reward"]
        buf.done = state["done"]
        buf.priorities = state["priorities"]
        buf.max_priority = state["max_priority"]
        return buf


class RunningNormalizer:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = int(dim)
        self.eps = float(eps)
        self.count = 0
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.m2 = np.zeros((self.dim,), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones((self.dim,), dtype=np.float64)
        var = self.m2 / max(self.count - 1, 1)
        return np.sqrt(var + self.eps)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mean = self.mean.astype(np.float32)
        std = self.std().astype(np.float32)
        return (x - mean) / std

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "eps": self.eps,
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
        }

    @classmethod
    def load(cls, state: dict) -> "RunningNormalizer":
        norm = cls(state["dim"], state["eps"])
        norm.count = state["count"]
        norm.mean = state["mean"]
        norm.m2 = state["m2"]
        return norm


def normalize_batch(normalizer: RunningNormalizer, batch):
    obs, action, reward, next_obs, done = batch
    obs_n = normalizer.normalize(obs)
    next_obs_n = normalizer.normalize(next_obs)
    return obs_n, action, reward, next_obs_n, done


class TBLogger:
    @staticmethod
    @contextlib.contextmanager
    def _silence_native_stderr():
        # Some TensorBoard/TensorFlow startup messages are emitted from native code
        # directly to file descriptor 2, bypassing Python's sys.stderr.
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
        except Exception:
            yield
            return
        try:
            saved_fd = os.dup(2)
        except Exception:
            os.close(devnull_fd)
            yield
            return
        try:
            os.dup2(devnull_fd, 2)
            yield
        finally:
            try:
                os.dup2(saved_fd, 2)
            finally:
                os.close(saved_fd)
                os.close(devnull_fd)

    def __init__(self, enabled: bool, logdir: str):
        self._writer = None
        self._writer_cls = None
        self._logdir = logdir
        self._split_writers = {}
        if not enabled:
            return
        # TensorBoard can pull TensorFlow internals that emit noisy startup logs to stderr.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
        try:
            with self._silence_native_stderr():
                from torch.utils.tensorboard import SummaryWriter
            self._writer_cls = SummaryWriter
        except Exception as exc:
            print(f"TensorBoard disabled (import failed): {exc}")
            return
        os.makedirs(logdir, exist_ok=True)
        try:
            with self._silence_native_stderr():
                self._writer = SummaryWriter(logdir)
        except Exception as exc:
            print(f"TensorBoard disabled (writer init failed): {exc}")
            self._writer = None

    def add(self, tag: str, value: Optional[float], step: int) -> None:
        if self._writer is None or value is None:
            return
        self._writer.add_scalar(tag, value, step)

    def add_split(self, split: str, tag: str, value: Optional[float], step: int) -> None:
        if self._writer is None or self._writer_cls is None or value is None:
            return
        split_key = str(split).strip().lower()
        if not split_key:
            return
        writer = self._get_split_writer(split_key)
        if writer is None:
            return
        writer.add_scalar(tag, value, step)

    def _get_split_writer(self, split_key: str):
        if self._writer is None or self._writer_cls is None:
            return None
        writer = self._split_writers.get(split_key)
        if writer is not None:
            return writer
        base = os.path.basename(os.path.normpath(self._logdir))
        parent = os.path.dirname(os.path.normpath(self._logdir))
        split_dir = os.path.join(parent, f"{base}_{split_key}")
        os.makedirs(split_dir, exist_ok=True)
        with self._silence_native_stderr():
            writer = self._writer_cls(split_dir)
        self._split_writers[split_key] = writer
        return writer

    def add_custom_scalars(self, layout: Mapping[str, Any]) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_custom_scalars(layout)
        except Exception:
            pass

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()
        for writer in self._split_writers.values():
            writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        for writer in self._split_writers.values():
            writer.close()
        self._split_writers.clear()

    def add_hist(self, tag: str, values: Optional[np.ndarray], step: int) -> None:
        if self._writer is None or values is None:
            return
        arr = np.asarray(values)
        if arr.size == 0:
            return
        self._writer.add_histogram(tag, arr, step)

    def add_split_hist(self, split: str, tag: str, values: Optional[np.ndarray], step: int) -> None:
        if self._writer is None or self._writer_cls is None or values is None:
            return
        split_key = str(split).strip().lower()
        if not split_key:
            return
        arr = np.asarray(values)
        if arr.size == 0:
            return
        writer = self._get_split_writer(split_key)
        if writer is None:
            return
        writer.add_histogram(tag, arr, step)


class EpsilonSchedule:
    def __init__(self, schedule: str, start: float, end: float, decay_steps: int):
        self.schedule = schedule
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(1, int(decay_steps))

    def value(self, step: int) -> float:
        if self.schedule == "constant":
            return self.start
        if self.schedule == "linear":
            frac = min(1.0, max(0.0, step / self.decay_steps))
            return self.start + frac * (self.end - self.start)
        if self.schedule == "exp":
            return self.end + (self.start - self.end) * math.exp(-step / self.decay_steps)
        return self.start


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        actor_hidden: int = 128,
        critic_hidden: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.actor = MLP(obs_dim, 1, actor_hidden).to(self.device)
        self.critic = MLP(obs_dim + 1, 1, critic_hidden).to(self.device)
        self.actor_t = MLP(obs_dim, 1, actor_hidden).to(self.device)
        self.critic_t = MLP(obs_dim + 1, 1, critic_hidden).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, obs: np.ndarray, noise: float = 0.0) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = torch.tanh(self.actor(obs_t)).cpu().numpy()[0, 0]
        if noise > 0.0:
            action += np.random.normal(0.0, noise)
        return float(np.clip(action, -1.0, 1.0))

    def update(self, batch, weights) -> Tuple[float, float, np.ndarray]:
        obs, action, reward, next_obs, done = batch
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = torch.tanh(self.actor_t(next_obs_t))
            q_next = self.critic_t(torch.cat([next_obs_t, next_action], dim=1))
            target = reward_t + self.gamma * (1.0 - done_t) * q_next

        q_val = self.critic(torch.cat([obs_t, action_t], dim=1))
        td_err = target - q_val
        critic_loss = (weights_t * td_err.pow(2)).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_action = torch.tanh(self.actor(obs_t))
        actor_loss = -(self.critic(torch.cat([obs_t, actor_action], dim=1)) * weights_t).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_t)
        self._soft_update(self.critic, self.critic_t)

        return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu()), td_err.detach().cpu().numpy()

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        for param, target_param in zip(src.parameters(), tgt.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "critic_t": self.critic_t.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "actor_t" in state:
            self.actor_t.load_state_dict(state["actor_t"])
        if "critic_t" in state:
            self.critic_t.load_state_dict(state["critic_t"])
        if "actor_opt" in state:
            self.actor_opt.load_state_dict(state["actor_opt"])
        if "critic_opt" in state:
            self.critic_opt.load_state_dict(state["critic_opt"])


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logp = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.rewards)

    def add(self, obs, action, logp, reward, value, done) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.logp.append(logp)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def finish(self, last_value: float, gamma: float, lam: float) -> None:
        n = len(self.rewards)
        values = np.array(self.values + [last_value], dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        adv = np.zeros((n,), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + values[:-1]

    def get(self):
        obs = np.asarray(self.obs, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32).reshape(-1, 1)
        logp = np.asarray(self.logp, dtype=np.float32).reshape(-1, 1)
        adv = np.asarray(self.advantages, dtype=np.float32)
        ret = np.asarray(self.returns, dtype=np.float32)
        return obs, actions, logp, adv, ret

    def reset(self) -> None:
        self.__init__()

    def state_dict(self) -> dict:
        return {
            "type": "rollout",
            "size": len(self.rewards),
        }


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        hidden: int,
        lr: float,
        clip: float,
        entropy_coef: float,
        value_coef: float,
        max_grad_norm: float,
        log_std_init: float,
        device: str,
        action_values: Optional[List[float]] = None,
        aux_classes: int = 0,
        aux_hidden: int = 0,
    ):
        self.device = torch.device(device)
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.action_values = action_values
        self.aux_classes = aux_classes

        out_dim = 1 if not action_values else len(action_values)
        self.policy = MLP(obs_dim, out_dim, hidden).to(self.device)
        self.value = MLP(obs_dim, 1, hidden).to(self.device)
        self.log_std = nn.Parameter(torch.ones(1, device=self.device) * log_std_init)
        self.aux_head = None
        if aux_classes and aux_classes > 0:
            aux_h = int(aux_hidden) if int(aux_hidden) > 0 else int(hidden)
            self.aux_head = MLP(obs_dim, aux_classes, aux_h).to(self.device)

        params = list(self.policy.parameters()) + list(self.value.parameters()) + [self.log_std]
        if self.aux_head is not None:
            params += list(self.aux_head.parameters())
        self.opt = optim.Adam(params, lr=lr)

    def _atanh(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        x = torch.clamp(x, -1 + eps, 1 - eps)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.policy(obs)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[float, float, float, Optional[int]]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.action_values:
            logits = self.policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                idx = torch.argmax(logits, dim=-1)
            else:
                idx = dist.sample()
            action = torch.tensor(self.action_values, device=self.device)[idx]
            logp = dist.log_prob(idx)
            value = self.value(obs_t)
            return (
                float(action.cpu().numpy()[0]),
                float(logp.detach().cpu().numpy()[0]),
                float(value.detach().cpu().numpy()[0, 0]),
                int(idx.cpu().numpy()[0]),
            )

        dist = self._dist(obs_t)
        if deterministic:
            pre_tanh = dist.mean
        else:
            pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        logp = logp.sum(dim=-1)
        value = self.value(obs_t)
        return (
            float(action.detach().cpu().numpy()[0, 0]),
            float(logp.detach().cpu().numpy()[0]),
            float(value.detach().cpu().numpy()[0, 0]),
            None,
        )

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.action_values:
            logits = self.policy(obs)
            dist = torch.distributions.Categorical(logits=logits)
            idx = actions.long().view(-1)
            logp = dist.log_prob(idx)
            entropy = dist.entropy()
            value = self.value(obs).squeeze(-1)
            return logp, entropy, value

        dist = self._dist(obs)
        pre_tanh = self._atanh(actions)
        logp = dist.log_prob(pre_tanh) - torch.log(1 - actions.pow(2) + 1e-6)
        logp = logp.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs).squeeze(-1)
        return logp, entropy, value

    def update(
        self,
        obs,
        actions,
        old_logp,
        adv,
        ret,
        epochs: int,
        batch_size: int,
        aux_obs: Optional[np.ndarray] = None,
        aux_targets: Optional[np.ndarray] = None,
        aux_weight: float = 0.0,
    ) -> Tuple[float, float, float, Optional[float], Optional[float]]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device).unsqueeze(1)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device).unsqueeze(1)
        aux_loss_val = None
        aux_acc_val = None
        aux_obs_t = None
        aux_targets_t = None
        use_aux = (
            self.aux_head is not None
            and aux_obs is not None
            and aux_targets is not None
            and aux_weight > 0.0
            and len(aux_obs) > 0
        )
        if use_aux:
            aux_obs_t = torch.tensor(aux_obs, dtype=torch.float32, device=self.device)
            aux_targets_t = torch.tensor(aux_targets, dtype=torch.long, device=self.device)
            with torch.no_grad():
                aux_logits = self.aux_head(aux_obs_t)
                aux_loss_val = float(nn.functional.cross_entropy(aux_logits, aux_targets_t).detach().cpu())
                preds = torch.argmax(aux_logits, dim=-1)
                aux_acc_val = float((preds == aux_targets_t).float().mean().detach().cpu())

        n = obs_t.size(0)
        idxs = np.arange(n)
        last_actor = last_critic = last_entropy = 0.0
        aux_scale = 0.0
        if use_aux:
            mb_per_epoch = max(1, math.ceil(n / batch_size))
            aux_scale = aux_weight / float(mb_per_epoch * max(1, epochs))

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, batch_size):
                mb_idx = idxs[start : start + batch_size]
                logp, entropy, value = self.evaluate(obs_t[mb_idx], actions_t[mb_idx])
                ratio = torch.exp(logp - old_logp_t[mb_idx])
                unclipped = ratio * adv_t[mb_idx].squeeze(-1)
                clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_t[
                    mb_idx
                ].squeeze(-1)
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (ret_t[mb_idx].squeeze(-1) - value).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                if use_aux:
                    aux_logits = self.aux_head(aux_obs_t)
                    aux_loss = nn.functional.cross_entropy(aux_logits, aux_targets_t)
                    loss = loss + aux_scale * aux_loss
                self.opt.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                    if self.aux_head is not None:
                        torch.nn.utils.clip_grad_norm_(self.aux_head.parameters(), self.max_grad_norm)
                self.opt.step()

                last_actor = float(policy_loss.detach().cpu())
                last_critic = float(value_loss.detach().cpu())
                last_entropy = float(entropy.detach().mean().cpu())

        return last_actor, last_critic, last_entropy, aux_loss_val, aux_acc_val

    def value_fn(self, obs: np.ndarray) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.value(obs_t).cpu().numpy()[0, 0])

    def state_dict(self) -> dict:
        state = {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "opt": self.opt.state_dict(),
            "log_std": self.log_std.detach().cpu().numpy(),
        }
        if self.aux_head is not None:
            state["aux_head"] = self.aux_head.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])
        if "log_std" in state:
            log_std = torch.tensor(state["log_std"], device=self.device).view_as(self.log_std)
            self.log_std.data.copy_(log_std)
        if self.aux_head is not None and "aux_head" in state:
            self.aux_head.load_state_dict(state["aux_head"])


class C51Net(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, atoms: int, hidden: int):
        super().__init__()
        self.num_actions = num_actions
        self.atoms = atoms
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions * atoms),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return logits.view(-1, self.num_actions, self.atoms)

    def dist(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        return torch.softmax(logits, dim=-1)


class C51Agent:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        action_values: List[float],
        hidden: int,
        lr: float,
        gamma: float,
        v_min: float,
        v_max: float,
        atoms: int,
        target_update: int,
        max_grad_norm: float,
        device: str,
        aux_classes: int = 0,
        aux_weight: float = 0.0,
        aux_hidden: int = 0,
        aux_class_weighting: str = "none",
        aux_label_smoothing: float = 0.0,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.target_update = max(1, int(target_update))
        self.max_grad_norm = max_grad_norm
        self.action_values = action_values
        self.aux_classes = max(0, int(aux_classes))
        self.aux_weight = max(0.0, float(aux_weight))
        self.aux_class_weighting = str(aux_class_weighting or "none").strip().lower()
        self.aux_label_smoothing = min(0.25, max(0.0, float(aux_label_smoothing)))
        self.last_aux_loss: Optional[float] = None
        self.last_aux_acc: Optional[float] = None

        self.q = C51Net(obs_dim, num_actions, atoms, hidden).to(self.device)
        self.target = C51Net(obs_dim, num_actions, atoms, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.aux_head = None
        if self.aux_classes > 0:
            aux_h = int(aux_hidden) if int(aux_hidden) > 0 else int(hidden)
            self.aux_head = MLP(obs_dim, self.aux_classes, aux_h).to(self.device)
        params = list(self.q.parameters())
        if self.aux_head is not None:
            params += list(self.aux_head.parameters())
        self.opt = optim.Adam(params, lr=lr)
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms, device=self.device)
        self.update_steps = 0

    def _q_values(self, dist: torch.Tensor) -> torch.Tensor:
        return (dist * self.support).sum(dim=-1)

    def _project_distribution(
        self,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Project the target distribution onto this agent's fixed support."""
        next_dist_all = self.target.dist(next_obs)
        next_q = self._q_values(next_dist_all)
        next_actions = torch.argmax(next_q, dim=-1)
        row = torch.arange(next_dist_all.size(0), device=self.device)
        next_dist = next_dist_all[row, next_actions]

        tz = reward.unsqueeze(1) + self.gamma * (1.0 - done.unsqueeze(1)) * self.support
        tz = torch.clamp(tz, self.v_min, self.v_max)
        b = (tz - self.v_min) / (self.v_max - self.v_min) * (self.atoms - 1)
        lower = b.floor().long()
        upper = b.ceil().long()

        projected = torch.zeros_like(next_dist)
        offset = (row * self.atoms).unsqueeze(1)
        same_atom = lower == upper
        lower_weight = torch.where(same_atom, torch.ones_like(b), upper.float() - b)
        upper_weight = torch.where(same_atom, torch.zeros_like(b), b - lower.float())
        projected.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_dist * lower_weight).view(-1),
        )
        projected.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_dist * upper_weight).view(-1),
        )
        return projected

    def act(self, obs: np.ndarray, epsilon: float = 0.0, deterministic: bool = False):
        if not deterministic and np.random.rand() < epsilon:
            idx = np.random.randint(0, len(self.action_values))
            return float(self.action_values[idx]), int(idx)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.q.dist(obs_t)
            q_vals = self._q_values(dist)
            greedy_idx = int(torch.argmax(q_vals, dim=-1).item())
        return float(self.action_values[greedy_idx]), greedy_idx

    def update(
        self,
        batch,
        weights,
        *,
        aux_obs: Optional[np.ndarray] = None,
        aux_targets: Optional[np.ndarray] = None,
        aux_weight: Optional[float] = None,
    ) -> Tuple[float, np.ndarray]:
        obs, actions, reward, next_obs, done = batch
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).view(-1)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).view(-1)

        dist = self.q.dist(obs_t)
        dist_a = dist[torch.arange(dist.size(0), device=self.device), actions_t]
        log_dist = torch.log(dist_a + 1e-8)

        with torch.no_grad():
            m = self._project_distribution(next_obs_t, reward_t, done_t)

        loss_per = -(m * log_dist).sum(dim=1)
        loss = (weights_t * loss_per).mean()

        use_aux = (
            self.aux_head is not None
            and aux_obs is not None
            and aux_targets is not None
            and len(aux_obs) > 0
        )
        aux_w = self.aux_weight if aux_weight is None else max(0.0, float(aux_weight))
        self.last_aux_loss = None
        self.last_aux_acc = None
        if use_aux and aux_w > 0.0:
            aux_obs_t = torch.tensor(aux_obs, dtype=torch.float32, device=self.device)
            aux_targets_t = torch.tensor(aux_targets, dtype=torch.long, device=self.device).view(-1)
            aux_logits = self.aux_head(aux_obs_t)
            class_weights = None
            if self.aux_class_weighting in {"balanced", "balanced_batch"}:
                counts = torch.bincount(aux_targets_t, minlength=self.aux_classes).float()
                denom = counts.clamp_min(1.0)
                class_weights = aux_targets_t.numel() / (float(self.aux_classes) * denom)
                class_weights = class_weights.to(device=self.device, dtype=torch.float32)
            aux_loss = nn.functional.cross_entropy(
                aux_logits,
                aux_targets_t,
                weight=class_weights,
                label_smoothing=self.aux_label_smoothing,
            )
            loss = loss + aux_w * aux_loss
            with torch.no_grad():
                self.last_aux_loss = float(aux_loss.detach().cpu())
                aux_pred = torch.argmax(aux_logits, dim=-1)
                self.last_aux_acc = float((aux_pred == aux_targets_t).float().mean().detach().cpu())

        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
            if self.aux_head is not None:
                torch.nn.utils.clip_grad_norm_(self.aux_head.parameters(), self.max_grad_norm)
        self.opt.step()

        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

        td_err = loss_per.detach().cpu().numpy()
        return float(loss.detach().cpu()), td_err

    def evaluate_batch(
        self,
        batch,
        *,
        aux_obs: Optional[np.ndarray] = None,
        aux_targets: Optional[np.ndarray] = None,
    ) -> Mapping[str, float]:
        """Evaluate a held-out batch without changing model or optimizer state."""
        obs, actions, reward, next_obs, done = batch
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).view(-1)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1)

        with torch.no_grad():
            dist = self.q.dist(obs_t)
            rows = torch.arange(dist.size(0), device=self.device)
            chosen = dist[rows, actions_t]
            target = self._project_distribution(next_obs_t, reward_t, done_t)
            loss_per = -(target * torch.log(chosen + 1e-8)).sum(dim=1)
            out = {
                "loss": float(loss_per.mean().cpu()),
                "td_abs_mean": float(loss_per.abs().mean().cpu()),
                "count": float(loss_per.numel()),
            }
            if (
                self.aux_head is not None
                and aux_obs is not None
                and aux_targets is not None
                and len(aux_obs) > 0
            ):
                aux_obs_t = torch.tensor(aux_obs, dtype=torch.float32, device=self.device)
                aux_targets_t = torch.tensor(
                    aux_targets,
                    dtype=torch.long,
                    device=self.device,
                ).view(-1)
                logits = self.aux_head(aux_obs_t)
                aux_loss = nn.functional.cross_entropy(
                    logits,
                    aux_targets_t,
                    label_smoothing=self.aux_label_smoothing,
                )
                out["aux_loss"] = float(aux_loss.cpu())
                out["aux_acc"] = float(
                    (torch.argmax(logits, dim=-1) == aux_targets_t).float().mean().cpu()
                )
                out["aux_count"] = float(aux_targets_t.numel())
        return out

    def state_dict(self) -> dict:
        out = {
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "v_min": self.v_min,
            "v_max": self.v_max,
            "atoms": self.atoms,
            "aux_classes": self.aux_classes,
            "aux_weight": self.aux_weight,
            "aux_class_weighting": self.aux_class_weighting,
            "aux_label_smoothing": self.aux_label_smoothing,
        }
        if self.aux_head is not None:
            out["aux_head"] = self.aux_head.state_dict()
        return out

    def load_state_dict(self, state: dict) -> None:
        self.q.load_state_dict(state["q"])
        if "target" in state:
            self.target.load_state_dict(state["target"])
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])
        if "aux_weight" in state:
            self.aux_weight = max(0.0, float(state["aux_weight"]))
        if "aux_class_weighting" in state:
            self.aux_class_weighting = str(state["aux_class_weighting"] or "none").strip().lower()
        if "aux_label_smoothing" in state:
            self.aux_label_smoothing = min(0.25, max(0.0, float(state["aux_label_smoothing"])))
        if self.aux_head is not None and "aux_head" in state:
            self.aux_head.load_state_dict(state["aux_head"])


@dataclass
class FeatureSpec:
    keys: List[str]

    def extract(self, fv: dict) -> np.ndarray:
        vec = np.zeros((len(self.keys),), dtype=np.float32)
        for i, key in enumerate(self.keys):
            val = fv.get(key, 0.0)
            if val is None:
                val = 0.0
            try:
                vec[i] = float(val)
            except (TypeError, ValueError):
                vec[i] = 0.0
        return vec
