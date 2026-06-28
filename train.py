"""
Deep reinforcement learning for daily portfolio allocation, benchmarked against
classic baselines (equal-weight, rolling mean-variance / max-Sharpe, and
buy-and-hold SPY) on an out-of-sample test period.

Agent: PPO (Stable-Baselines3) on a custom Gymnasium environment whose action is
a softmax-normalized, long-only, fully-invested weight vector and whose reward is
the next-day portfolio log-return net of transaction costs.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from portfolio_env import PortfolioEnv, softmax

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets"); os.makedirs(ASSETS, exist_ok=True)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

WINDOW, COST = 30, 0.001
TRAIN_END = "2019-12-31"
TRADING_DAYS = 252


def load_returns():
    px = pd.read_csv(os.path.join(HERE, "data", "prices.csv"), index_col=0, parse_dates=True)
    rets = px.pct_change().dropna()
    return rets


def backtest(weights, rets, cost=COST):
    """weights, rets: (T, N) aligned. Returns daily net returns + equity curve."""
    w = np.asarray(weights); r = np.asarray(rets)
    prev = np.concatenate([w[:1], w[:-1]], axis=0)          # weights held the prior day
    turnover = np.abs(w - prev).sum(axis=1)
    net = (w * r).sum(axis=1) - cost * turnover
    equity = np.cumprod(1.0 + net)
    return net, equity


def metrics(net):
    ann_ret = float(np.mean(net) * TRADING_DAYS)
    ann_vol = float(np.std(net) * np.sqrt(TRADING_DAYS))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    eq = np.cumprod(1.0 + net)
    max_dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    total = float(eq[-1] - 1.0)
    return {"total_return": round(total, 4), "ann_return": round(ann_ret, 4),
            "ann_vol": round(ann_vol, 4), "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4)}


def max_sharpe_weights(window_rets):
    mu = window_rets.mean(0); cov = np.cov(window_rets.T) + 1e-6 * np.eye(window_rets.shape[1])
    n = len(mu)
    neg_sharpe = lambda w: -(w @ mu) / np.sqrt(w @ cov @ w + 1e-12)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    return res.x if res.success else np.ones(n) / n


def mv_weights_full(rets, lookback=TRADING_DAYS, rebal=21):
    R = rets.values; T, N = R.shape
    W = np.tile(np.ones(N) / N, (T, 1))
    last = np.ones(N) / N
    for t in range(T):
        if t >= lookback and (t - lookback) % rebal == 0:
            last = max_sharpe_weights(R[t - lookback:t])
        W[t] = last
    return W


def drl_weights(model, test_rets):
    env = PortfolioEnv(test_rets.values, window=WINDOW, cost=COST)
    obs, _ = env.reset(); done = False; ws = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        ws.append(info["weights"]); done = term or trunc
    return np.array(ws)


def main(timesteps):
    set_random_seed(SEED)
    rets = load_returns()
    assets = list(rets.columns)
    train_rets = rets.loc[:TRAIN_END]
    test_rets = rets.loc[TRAIN_END:].iloc[1:]
    print(f"assets={assets}\ntrain {train_rets.shape}  test {test_rets.shape}")

    # ---- train PPO ----
    env = PortfolioEnv(train_rets.values, window=WINDOW, cost=COST)
    model = PPO("MlpPolicy", env, seed=SEED, verbose=0, n_steps=2048, batch_size=256,
                gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.001)
    model.learn(total_timesteps=timesteps)

    # ---- aligned test backtest (DRL warms up for `window` days) ----
    aligned = test_rets.iloc[WINDOW:]
    R = aligned.values
    w_drl = drl_weights(model, test_rets)[:len(R)]
    w_eq = np.tile(np.ones(len(assets)) / len(assets), (len(R), 1))
    w_mv_full = mv_weights_full(rets)                         # uses past-only data, no leakage
    w_mv = w_mv_full[-len(R):]
    w_spy = np.zeros_like(w_eq); w_spy[:, assets.index("SPY")] = 1.0

    strategies = {"DRL (PPO)": w_drl, "Equal-Weight": w_eq,
                  "Mean-Variance": w_mv, "Buy&Hold SPY": w_spy}
    results, equities, nets = {}, {}, {}
    for name, w in strategies.items():
        net, eq = backtest(w, R)
        results[name] = metrics(net); equities[name] = eq; nets[name] = net
        print(f"{name:16s} Sharpe={results[name]['sharpe']:.2f}  "
              f"AnnRet={results[name]['ann_return']:.3f}  MaxDD={results[name]['max_drawdown']:.3f}")

    json.dump(results, open(os.path.join(RESULTS, "metrics.json"), "w"), indent=2)
    pd.DataFrame(results).T.to_csv(os.path.join(RESULTS, "metrics.csv"))
    make_figures(results, equities, aligned.index, w_drl, assets)
    print("\nSaved metrics + figures.\n" + json.dumps(results, indent=2))


def make_figures(results, equities, dates, w_drl, assets):
    colors = {"DRL (PPO)": "#56a98c", "Equal-Weight": "#9aa7b8",
              "Mean-Variance": "#6f9bd8", "Buy&Hold SPY": "#dca06a"}
    # 1) equity curves
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for n, eq in equities.items():
        ax.plot(dates, eq, label=f"{n} (Sharpe {results[n]['sharpe']:.2f})",
                color=colors[n], lw=2)
    ax.set_title("Out-of-sample growth of $1 (2020-2024, net of costs)")
    ax.set_ylabel("portfolio value"); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "equity_curves.png")); plt.close(fig)

    # 2) risk/return bars
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, key, title in zip(axes, ["sharpe", "ann_return", "max_drawdown"],
                              ["Sharpe ratio", "Annualized return", "Max drawdown"]):
        ax.bar(names, [results[n][key] for n in names], color=[colors[n] for n in names])
        ax.set_title(title); ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "risk_return.png")); plt.close(fig)

    # 3) DRL allocation over time
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.stackplot(dates, w_drl.T, labels=assets, alpha=0.9)
    ax.set_title("DRL agent: learned portfolio allocation over time")
    ax.set_ylabel("weight"); ax.set_ylim(0, 1)
    ax.legend(loc="upper center", ncol=len(assets), fontsize=8, frameon=False)
    ax.margins(x=0); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "drl_allocation.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(timesteps=3000 if a.smoke else a.timesteps)
