"""
Experiment runner: compare exploration strategies for Q-learning.

    python experiments.py

Protocol
--------
* 8 strategy variants (strategies.comparison_set()) x 5 run seeds (0-4).
* Each run trains for 6,000 episodes (30-day months), with every Q-value
  starting at -20 (see QLearningConfig). With 512 states x 19 actions the
  table needs this much data; at 3,000 episodes the learning curves were
  still falling. The decaying-epsilon schedule reaches its 0.05 floor at
  episode 4,000, leaving 2,000 episodes of mostly exploiting.
* After training, each learned Q-table is evaluated greedily (no exploration)
  on 100 held-out test episodes (seeds 3,000,000-3,000,099). Training seeds
  are all < 1,000,000, so the two never overlap. Every policy, including the
  baselines, sees the same 100 test months.
* Baselines on the same test months:
    - Random policy: a uniformly random action every day (5 seeds).
    - Reorder-to-target heuristic: each day, find the location whose
      inventory position (on hand + in transit) is furthest below a target of
      7 days of supply; ship the largest standard quantity (10/20/40) that
      fits the gap, via the cheap Consolidated truck. Ship nothing if no gap
      is at least 10 units. Deterministic, so all 5 "seeds" are identical.

Metrics
-------
* total_cost        all costs + penalties for the month, in $ (the
                    "distribution cost" the reward minimizes)
* transport_cost    shipping spend alone, in $
* stockout_rate     stockout units / demand units
* delivery_eff      on-time deliveries / total deliveries
* epsilon           current epsilon (epsilon strategies only, else blank)
* non_greedy_rate   share of actions that were NOT the current best-known
                    action, a measure of how much each strategy actually
                    explores, comparable across all strategies

Convergence
-----------
Using the training curve of monthly total cost (with exploration on), take
the 100-episode moving average (MA). Its final value is the MA at the last
episode. The convergence episode is the first episode after which the MA
stays within +/-5% of that final value for the rest of training. This
measures when the learning curve settles. It does not say the settled level
is good: a strategy can converge quickly to a high cost.

Outputs: results/training_log.csv, results/evaluation_results.csv,
results/summary.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from q_learning import (ActionSelector, QLearningConfig, check_disjoint,
                        evaluate, test_seeds, train, _finish, _new_episode_log,
                        _record)
from strategies import comparison_set
from supply_chain_env import (ACTION_TABLE, DEMAND_MEAN, QUANTITIES, ROUTES,
                              PharmaSupplyChainEnv)

RESULTS = Path(__file__).parent / "results"
RUN_SEEDS = [0, 1, 2, 3, 4]
EPISODES = 6000
Q_INIT = -20.0
N_TEST_EPISODES = 100
MA_WINDOW = 100
CONVERGENCE_TOL = 0.05
HEURISTIC_TARGET_DAYS = 7.0


# ---------------------------------------------------------------------------
# Wrapper that measures exploration without changing the strategy
# ---------------------------------------------------------------------------
class ExplorationTracker(ActionSelector):
    """Passes every call through to the real strategy and records, per
    episode, the epsilon in use and how often the chosen action was not the
    current greedy (best-known) action."""

    def __init__(self, inner: ActionSelector):
        self.inner = inner
        self.name = inner.name
        self.records = []
        self._steps = self._non_greedy = 0

    def reset(self, n_states, n_actions):
        self.inner.reset(n_states, n_actions)
        self.records, self._steps, self._non_greedy = [], 0, 0

    def select(self, q_row, state, rng):
        a = self.inner.select(q_row, state, rng)
        self._steps += 1
        self._non_greedy += int(q_row[a] < q_row.max())
        return a

    def end_episode(self, episode):
        self.records.append({
            "epsilon": getattr(self.inner, "epsilon", np.nan),
            "non_greedy_rate": self._non_greedy / max(self._steps, 1),
        })
        self._steps = self._non_greedy = 0
        self.inner.end_episode(episode)


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------
TRUCK = next(r for r in ROUTES if r.startswith("Consolidated"))


def heuristic_action(env) -> int:
    e = env.unwrapped
    position = e.inventory.astype(float).copy()
    for s in e.pipeline:
        position[s["dest"]] += s["qty"]
    gap = HEURISTIC_TARGET_DAYS * DEMAND_MEAN - position
    dest = int(np.argmax(gap))
    fitting = [(n, q) for n, q in QUANTITIES.items() if q <= gap[dest]]
    if not fitting:
        return 0
    q_name = max(fitting, key=lambda x: x[1])[0]
    return ACTION_TABLE.index((dest, q_name, TRUCK))


def run_policy(env, choose, seeds) -> list[dict]:
    """Evaluate an arbitrary policy function on the given seeds."""
    seeds = list(seeds)
    check_disjoint(seeds)
    out = []
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        log = _new_episode_log()
        done = False
        while not done:
            state, reward, term, trunc, info = env.step(choose(state))
            _record(log, reward, info)
            done = term or trunc
        out.append({"seed": seed, **_finish(log)})
    return out


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
def convergence_episode(costs: np.ndarray, window=MA_WINDOW, tol=CONVERGENCE_TOL):
    ma = pd.Series(costs).rolling(window).mean().to_numpy()
    final = ma[-1]
    outside = np.abs(ma - final) > tol * abs(final)
    outside[: window - 1] = True                 # no full window yet
    last_outside = np.flatnonzero(outside)
    return int(last_outside[-1] + 2) if last_outside.size else window


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def eval_rows(strategy, run_seed, episodes):
    return [{"strategy": strategy, "run_seed": run_seed, "test_seed": e["seed"],
             "total_cost": e["total_cost"], "transport_cost": e["transport_cost"],
             "holding_cost": e["holding_cost"],
             "stockout_penalty": e["stockout_penalty"],
             "late_penalty": e["late_penalty"],
             "overstock_penalty": e["overstock_penalty"],
             "stockout_units": e["stockout_units"], "demand_units": e["demand_units"],
             "stockout_rate": e["stockout_rate"],
             "on_time_deliveries": e["on_time_deliveries"],
             "total_deliveries": e["total_deliveries"],
             "delivery_eff": e["on_time_rate"],
             "shipments": e["shipments"], "express_shipments": e["express_shipments"],
             "reward": e["reward"]} for e in episodes]


def pooled(df: pd.DataFrame) -> pd.Series:
    """Per-run test metrics, with rates pooled over the 100 episodes."""
    return pd.Series({
        "final_cost": df["total_cost"].mean(),
        "transport_cost": df["transport_cost"].mean(),
        "stockout_rate": df["stockout_units"].sum() / df["demand_units"].sum(),
        "delivery_eff": df["on_time_deliveries"].sum() / df["total_deliveries"].sum(),
        "express_share": df["express_shipments"].sum() / max(df["shipments"].sum(), 1),
    })


def main():
    RESULTS.mkdir(exist_ok=True)
    env = PharmaSupplyChainEnv()
    config = QLearningConfig(episodes=EPISODES, q_init=Q_INIT)
    tests = test_seeds(N_TEST_EPISODES)
    train_rows, eval_list, conv_rows = [], [], []
    t0 = time.time()

    names = [s.name for s in comparison_set()]
    for i, name in enumerate(names):
        for run_seed in RUN_SEEDS:
            tracker = ExplorationTracker(comparison_set()[i])   # fresh state
            Q, history = train(env, tracker, config, run_seed=run_seed)
            for h, x in zip(history, tracker.records):
                train_rows.append({
                    "strategy": name, "run_seed": run_seed, "episode": h["episode"],
                    "total_reward": h["reward"], "total_cost": h["total_cost"],
                    "transport_cost": h["transport_cost"],
                    "stockout_rate": h["stockout_rate"],
                    "delivery_eff": h["on_time_rate"],
                    "epsilon": x["epsilon"], "non_greedy_rate": x["non_greedy_rate"],
                })
            conv_rows.append({
                "strategy": name, "run_seed": run_seed,
                "convergence_episode": convergence_episode(
                    np.array([h["total_cost"] for h in history])),
                "training_cost_total": sum(h["total_cost"] for h in history),
            })
            episodes, _ = evaluate(env, Q, tests)
            eval_list += eval_rows(name, run_seed, episodes)
        print(f"trained {name:<22} ({time.time() - t0:5.0f}s elapsed)")

    for run_seed in RUN_SEEDS:
        rng = np.random.default_rng(run_seed)
        eps = run_policy(env, lambda s: int(rng.integers(env.action_space.n)), tests)
        eval_list += eval_rows("Random policy (baseline)", run_seed, eps)
        eps = run_policy(env, lambda s: heuristic_action(env), tests)
        eval_list += eval_rows("Reorder-to-target heuristic (baseline)", run_seed, eps)

    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_list)
    conv_df = pd.DataFrame(conv_rows)
    train_df.to_csv(RESULTS / "training_log.csv", index=False)
    eval_df.to_csv(RESULTS / "evaluation_results.csv", index=False)

    # ---------------- summary: mean +/- std across the 5 run seeds ----------
    per_run = (eval_df.groupby(["strategy", "run_seed"]).apply(pooled, include_groups=False)
               .reset_index().merge(conv_df, on=["strategy", "run_seed"], how="left"))
    order = names + ["Reorder-to-target heuristic (baseline)", "Random policy (baseline)"]
    cols = ["final_cost", "stockout_rate", "delivery_eff", "express_share",
            "convergence_episode", "training_cost_total"]
    summary = per_run.groupby("strategy")[cols].agg(["mean", "std"]).reindex(order)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(RESULTS / "summary.csv")

    def pm(m, s, fmt):
        if pd.isna(m):
            return "n/a"
        return f"{fmt(m)} ± {fmt(s)}"

    money = lambda v: f"${v:,.0f}"
    pct = lambda v: f"{v * 100:.1f}%"
    ep = lambda v: f"{v:,.0f}"
    table = pd.DataFrame({
        "Test cost / month": [pm(summary.loc[s, "final_cost_mean"], summary.loc[s, "final_cost_std"], money) for s in order],
        "Stockout rate": [pm(summary.loc[s, "stockout_rate_mean"], summary.loc[s, "stockout_rate_std"], pct) for s in order],
        "Delivery eff. (on-time)": [pm(summary.loc[s, "delivery_eff_mean"], summary.loc[s, "delivery_eff_std"], pct) for s in order],
        "Express share": [pm(summary.loc[s, "express_share_mean"], summary.loc[s, "express_share_std"], pct) for s in order],
        "Episodes to converge": [pm(summary.loc[s, "convergence_episode_mean"], summary.loc[s, "convergence_episode_std"], ep) for s in order],
    }, index=order)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nGreedy evaluation on {N_TEST_EPISODES} held-out months, "
          f"mean ± std across {len(RUN_SEEDS)} run seeds "
          f"({EPISODES:,} training episodes per run)\n")
    print(table.to_string())
    print(f"\nConvergence = first episode after which the {MA_WINDOW}-episode moving "
          f"average of training cost stays within ±{CONVERGENCE_TOL:.0%} of its final value.")
    print(f"Total runtime: {time.time() - t0:.0f}s. "
          f"Saved training_log.csv, evaluation_results.csv, summary.csv to {RESULTS}")


if __name__ == "__main__":
    main()
