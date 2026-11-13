"""
Tabular Q-learning for the pharmaceutical supply chain environment.

The update rule comes from the Bellman optimality equation:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    alpha  learning rate: how far each new experience moves the estimate
    gamma  discount factor: how much future costs count compared to today's
    r + gamma * max_a' Q(s',a')  is the "TD target": today's reward plus the
                                 best value the agent believes is reachable
                                 from tomorrow's state

Q-learning is off-policy: it learns the value of the *greedy* policy no
matter how actions are chosen during training. That is why the exploration
strategy can be swapped out without changing the update. `train()` takes the
strategy as a parameter, so every strategy goes through identical code.

Why gamma close to 1 (default 0.95)
-----------------------------------
In a supply chain, today's decision shows up in inventory days later: a
truck shipment lands in 3-4 days, and stock left today is still on the shelf
tomorrow. With gamma = 0.95, a stockout 4 days from now still counts at
0.95^4 = 81% of its full cost, and the effective planning horizon is about
1 / (1 - 0.95) = 20 days. With gamma = 0.5 the same stockout would count at
only 6%, so the agent would be too short-sighted to ever pay for a shipment
whose benefit arrives later.

Seeds: training and evaluation never overlap
--------------------------------------------
Every episode is reset with an explicit seed taken from a separate range:

    training   : run_seed * 100_000 + episode     (always < 1,000,000)
    validation : 2,000,000 + i
    test       : 3,000,000 + i

So the greedy policy is always scored on demand scenarios it never trained
on, the RL equivalent of a held-out test set. `check_disjoint()` enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Seed ranges (kept disjoint on purpose)
# ---------------------------------------------------------------------------
TRAIN_SEED_STRIDE = 100_000
TRAIN_SEED_LIMIT = 1_000_000
VALIDATION_SEED_START = 2_000_000
TEST_SEED_START = 3_000_000


def training_seed(run_seed: int, episode: int) -> int:
    seed = run_seed * TRAIN_SEED_STRIDE + episode
    assert seed < TRAIN_SEED_LIMIT, "training seed escaped its range"
    return seed


def validation_seeds(n: int) -> list[int]:
    return list(range(VALIDATION_SEED_START, VALIDATION_SEED_START + n))


def test_seeds(n: int) -> list[int]:
    return list(range(TEST_SEED_START, TEST_SEED_START + n))


def check_disjoint(seeds) -> None:
    """Fail loudly if any evaluation seed could also be a training seed."""
    leaked = [s for s in seeds if s < TRAIN_SEED_LIMIT]
    if leaked:
        raise ValueError(f"evaluation seeds overlap the training range: {leaked[:5]}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QLearningConfig:
    alpha: float = 0.1       # learning rate
    gamma: float = 0.95      # discount factor (see module docstring)
    episodes: int = 3000     # 30-day episodes per training run
    q_init: float = 0.0      # initial value of every Q-table entry
    # On optimistic initialization:
    # every reward here is negative (it is a cost), so the true Q-values are
    # all below 0. That means q_init = 0.0 is already *optimistic*: any
    # untried action looks better than one that has been tried and has cost
    # something, so even a purely greedy agent sweeps through actions early
    # on. Raising q_init (e.g. +5) would make that systematic exploration
    # stronger and longer-lasting. Setting it pessimistic (e.g. -50, below
    # the real values) would remove it, leaving all exploration to the
    # strategy. Keeping 0.0 for every strategy keeps the comparison fair.


# ---------------------------------------------------------------------------
# Action-selection interface
# ---------------------------------------------------------------------------
class ActionSelector:
    """Interface every exploration strategy implements.

    select(q_row, state, rng) -> action
        q_row is Q[state, :]; return an action index.
    reset(n_states, n_actions)
        called once at the start of training (e.g. clear visit counts).
    end_episode(episode)
        called after each episode (e.g. decay epsilon).
    """

    name = "base"

    def reset(self, n_states: int, n_actions: int) -> None:
        pass

    def select(self, q_row: np.ndarray, state: int, rng: np.random.Generator) -> int:
        raise NotImplementedError

    def end_episode(self, episode: int) -> None:
        pass


def greedy_action(q_row: np.ndarray) -> int:
    """Best-known action. Ties go to the lowest index (deterministic), so
    evaluation is reproducible; in an unvisited state that is 'No shipment'."""
    return int(np.argmax(q_row))


# ---------------------------------------------------------------------------
# Episode bookkeeping
# ---------------------------------------------------------------------------
_SUM_KEYS = ("transport_cost", "holding_cost", "stockout_penalty",
             "late_penalty", "overstock_penalty", "total_cost",
             "stockout_units", "demand_units", "wasted_units",
             "on_time_deliveries", "late_deliveries", "total_deliveries")


def _new_episode_log() -> dict:
    log = {k: 0 for k in _SUM_KEYS}
    log.update(reward=0.0, shipments=0, express_shipments=0, truck_shipments=0)
    return log


def _record(log: dict, reward: float, info: dict) -> None:
    log["reward"] += reward
    for k in _SUM_KEYS:
        log[k] += info[k]
    if info["route"] is not None:
        log["shipments"] += 1
        if info["route"].startswith("Express"):
            log["express_shipments"] += 1
        else:
            log["truck_shipments"] += 1


def _finish(log: dict) -> dict:
    log["stockout_rate"] = log["stockout_units"] / max(log["demand_units"], 1)
    log["on_time_rate"] = (log["on_time_deliveries"] / log["total_deliveries"]
                           if log["total_deliveries"] else float("nan"))
    return log


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(env, strategy: ActionSelector, config: QLearningConfig = QLearningConfig(),
          run_seed: int = 0, on_episode_end=None):
    """Train a Q-table with the given action-selection strategy.

    Returns (Q, history) where history is a list of per-episode dicts
    (costs, stockouts, deliveries) recorded *while exploring*, i.e. what
    the business would actually experience during the learning period.

    on_episode_end(episode, Q) is an optional hook, e.g. for periodic
    validation, called after each episode.
    """
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = np.full((n_states, n_actions), config.q_init, dtype=float)
    rng = np.random.default_rng(run_seed)      # the agent's own randomness
    strategy.reset(n_states, n_actions)
    history = []

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=training_seed(run_seed, episode))
        log = _new_episode_log()
        done = False
        while not done:
            action = strategy.select(Q[state], state, rng)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Bellman / TD update. Bootstrap from the next state unless the
            # episode reached a true terminal state. Day 30 is a time limit
            # (truncated), not the end of the business, so we still bootstrap.
            future = 0.0 if terminated else config.gamma * Q[next_state].max()
            td_target = reward + future
            Q[state, action] += config.alpha * (td_target - Q[state, action])

            _record(log, reward, info)
            state = next_state
            done = terminated or truncated

        strategy.end_episode(episode)
        history.append({"episode": episode, **_finish(log)})
        if on_episode_end is not None:
            on_episode_end(episode, Q)

    return Q, history


# ---------------------------------------------------------------------------
# Evaluation (greedy, held-out seeds)
# ---------------------------------------------------------------------------
def evaluate(env, Q: np.ndarray, seeds) -> tuple[list[dict], dict]:
    """Run the learned policy greedily (no exploration) on held-out seeds.

    Returns (per_episode, summary). Rates in the summary are pooled over all
    episodes (total stockout units / total demand units), not averages of
    per-episode rates, so long and short months are weighted fairly.
    """
    seeds = list(seeds)
    check_disjoint(seeds)
    episodes = []
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        log = _new_episode_log()
        done = False
        while not done:
            state, reward, terminated, truncated, info = env.step(greedy_action(Q[state]))
            _record(log, reward, info)
            done = terminated or truncated
        episodes.append({"seed": seed, **_finish(log)})

    n = len(episodes)
    tot = {k: sum(e[k] for e in episodes) for k in _SUM_KEYS + ("shipments",
                                                                 "express_shipments")}
    summary = {
        "episodes": n,
        "mean_monthly_cost": tot["total_cost"] / n,
        "mean_transport_cost": tot["transport_cost"] / n,
        "mean_holding_cost": tot["holding_cost"] / n,
        "mean_stockout_penalty": tot["stockout_penalty"] / n,
        "mean_late_penalty": tot["late_penalty"] / n,
        "mean_overstock_penalty": tot["overstock_penalty"] / n,
        "stockout_rate": tot["stockout_units"] / max(tot["demand_units"], 1),
        "on_time_rate": (tot["on_time_deliveries"] / tot["total_deliveries"]
                         if tot["total_deliveries"] else float("nan")),
        "shipments_per_month": tot["shipments"] / n,
        "express_share": (tot["express_shipments"] / tot["shipments"]
                          if tot["shipments"] else float("nan")),
        "mean_reward": sum(e["reward"] for e in episodes) / n,
    }
    return episodes, summary
