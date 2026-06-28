"""A Gymnasium environment for daily multi-asset portfolio allocation.

State  : the last `window` days of asset returns + the currently held weights.
Action : an unconstrained vector that is softmax-normalized into long-only weights
         summing to 1 (fully invested, no leverage / shorting).
Reward : the differential Sharpe ratio (Moody & Saffell) of the next-day
         portfolio return net of transaction costs — an online, risk-adjusted
         objective that rewards return per unit of volatility rather than raw
         return, discouraging churn and concentration.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, returns: np.ndarray, window: int = 30, cost: float = 0.001,
                 eta: float = 0.02, tilt: float = 0.5):
        super().__init__()
        self.returns = returns.astype(np.float32)        # (T, N) daily simple returns
        self.T, self.N = self.returns.shape
        self.window = window
        self.cost = cost
        self.eta = eta                                   # differential-Sharpe EMA rate
        self.tilt = tilt                                 # how far weights may move off equal-weight
        self.equal = np.ones(self.N, dtype=np.float32) / self.N
        self.action_space = spaces.Box(-10.0, 10.0, shape=(self.N,), dtype=np.float32)
        obs_dim = window * self.N + self.N
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    def _obs(self):
        hist = self.returns[self.t - self.window:self.t].flatten()
        return np.concatenate([hist, self.w]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.w = (np.ones(self.N, dtype=np.float32) / self.N)
        self.A = 0.0          # EMA of returns
        self.B = 0.0          # EMA of squared returns
        self._init = False
        return self._obs(), {}

    def _diff_sharpe(self, R):
        if not self._init:
            self.A, self.B, self._init = R, R * R, True
            return 0.0
        dA, dB = R - self.A, R * R - self.B
        denom = (self.B - self.A ** 2) ** 1.5
        d = (self.B * dA - 0.5 * self.A * dB) / denom if denom > 1e-8 else 0.0
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(d)

    def step(self, action):
        raw = softmax(np.asarray(action, dtype=np.float64)).astype(np.float32)
        # tilt around an equal-weight core so the agent allocates rather than gambles
        new_w = (1.0 - self.tilt) * self.equal + self.tilt * raw
        turnover = float(np.abs(new_w - self.w).sum())
        asset_ret = self.returns[self.t]                  # return realized on day t
        port_ret = float(new_w @ asset_ret) - self.cost * turnover
        self.w = new_w
        self.t += 1
        terminated = self.t >= self.T
        reward = self._diff_sharpe(port_ret)
        info = {"port_ret": port_ret, "weights": new_w, "turnover": turnover}
        return self._obs(), reward, terminated, False, info
