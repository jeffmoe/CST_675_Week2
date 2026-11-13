"""
Exploration strategies for tabular Q-learning.

Each class implements the ActionSelector interface from q_learning.py, so any
of them can be passed to train() unchanged:

    select(q_row, state, rng) -> action   choose an action for this state
    reset(n_states, n_actions)            clear internal state before training
    end_episode(episode)                  per-episode hook (e.g. decay)

How each one decides when to explore (plain language)
-----------------------------------------------------
* Fixed epsilon-greedy: "Roll a die every day." With probability epsilon,
  try a random shipment. Otherwise use the best-known one. It explores at
  the same rate forever, even after it has learned.
* Decaying epsilon-greedy: "Explore a lot while new, less with experience."
  It starts out fully random (epsilon = 1.0) and lowers epsilon a little after
  every month until it reaches a 5% floor.
* Softmax (Boltzmann): "Try options in proportion to how good they look."
  Better-valued shipments are picked more often, and poor ones rarely, but
  nothing is ever ruled out. The temperature tau sets how adventurous it is:
  low tau is nearly greedy, high tau is nearly random.
* UCB (Upper Confidence Bound): "Give the benefit of the doubt to what you
  haven't tried enough." Each option gets a bonus that shrinks as it is tried
  more often, so exploration targets uncertainty instead of being random. c
  sets the size of the bonus. UCB1 comes from multi-armed bandits; here it is
  applied separately in each state, a common heuristic extension to MDPs.
"""

from __future__ import annotations

import numpy as np

from q_learning import ActionSelector


def _argmax_random_ties(values: np.ndarray, rng: np.random.Generator) -> int:
    """Best action, with ties broken at random. Without this, all-zero rows
    (unvisited states) would always pick action 0, 'No shipment'."""
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


# ---------------------------------------------------------------------------
# 1. Fixed epsilon-greedy
# ---------------------------------------------------------------------------
class EpsilonGreedy(ActionSelector):
    def __init__(self, epsilon: float):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.name = f"Epsilon-greedy ε={epsilon:g}"

    def select(self, q_row, state, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(len(q_row)))       # explore
        return _argmax_random_ties(q_row, rng)          # exploit


# ---------------------------------------------------------------------------
# 2. Decaying epsilon-greedy
# ---------------------------------------------------------------------------
class DecayingEpsilonGreedy(ActionSelector):
    """epsilon_k = max(floor, start * decay**k) after k completed episodes.

    The decay rate is derived from `episodes_to_floor` so the schedule is
    easy to explain: epsilon falls from 1.0 to 0.05 over the first 2,000
    episodes, then stays at 0.05.
    """

    def __init__(self, start: float = 1.0, floor: float = 0.05,
                 episodes_to_floor: int = 2000):
        self.start, self.floor = start, floor
        self.decay = (floor / start) ** (1.0 / episodes_to_floor)
        self.epsilon = start
        self.name = f"Decaying ε {start:g}→{floor:g}"

    def reset(self, n_states, n_actions):
        self.epsilon = self.start

    def select(self, q_row, state, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(len(q_row)))
        return _argmax_random_ties(q_row, rng)

    def end_episode(self, episode):
        self.epsilon = max(self.floor, self.start * self.decay ** episode)


# ---------------------------------------------------------------------------
# 3. Softmax / Boltzmann
# ---------------------------------------------------------------------------
class Softmax(ActionSelector):
    """P(a | s) = exp(Q(s,a)/tau) / sum_b exp(Q(s,b)/tau)."""

    def __init__(self, tau: float):
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.tau = tau
        self.name = f"Softmax τ={tau:g}"

    def probabilities(self, q_row: np.ndarray) -> np.ndarray:
        # Subtract the max before exponentiating. Mathematically the result is
        # unchanged (the constant cancels), but now the largest exponent is
        # exactly 0, so exp() can't overflow to inf. At least one term is
        # exp(0) = 1, so the sum can't underflow to 0 and divide by zero.
        z = (q_row - q_row.max()) / self.tau
        p = np.exp(z)
        return p / p.sum()

    def select(self, q_row, state, rng):
        return int(rng.choice(len(q_row), p=self.probabilities(q_row)))


# ---------------------------------------------------------------------------
# 4. Upper Confidence Bound (UCB)
# ---------------------------------------------------------------------------
class UCB(ActionSelector):
    """a = argmax_a [ Q(s,a) + c * sqrt( ln N(s) / N(s,a) ) ]

    N(s,a) counts how often action a was taken in state s during training,
    and N(s) = sum over a of N(s,a). Any action with N(s,a) = 0 is tried
    first (its bonus would be infinite).
    """

    def __init__(self, c: float):
        if c < 0:
            raise ValueError("c must be >= 0")
        self.c = c
        self.counts = None
        self.name = f"UCB c={c:g}"

    def reset(self, n_states, n_actions):
        self.counts = np.zeros((n_states, n_actions), dtype=np.int64)

    def select(self, q_row, state, rng):
        n_sa = self.counts[state]
        untried = np.flatnonzero(n_sa == 0)
        if untried.size:
            action = int(rng.choice(untried))
        else:
            bonus = self.c * np.sqrt(np.log(n_sa.sum()) / n_sa)
            action = _argmax_random_ties(q_row + bonus, rng)
        self.counts[state, action] += 1
        return action


# ---------------------------------------------------------------------------
# Strategy sets
# ---------------------------------------------------------------------------
def all_variants() -> list[ActionSelector]:
    """Every variant requested (10). Each call returns fresh objects, since
    strategies keep internal state (counts, epsilon) between episodes."""
    return [
        EpsilonGreedy(0.01), EpsilonGreedy(0.1), EpsilonGreedy(0.3),
        DecayingEpsilonGreedy(start=1.0, floor=0.05, episodes_to_floor=2000),
        Softmax(0.5), Softmax(1.0), Softmax(5.0),
        UCB(0.5), UCB(1.0), UCB(2.0),
    ]


def comparison_set() -> list[ActionSelector]:
    """The 8 variants used for the main comparison.

    All three fixed-epsilon values, since fixed vs. decaying epsilon is the
    core question. For softmax and UCB, keep the low and high ends of each
    range (tau 0.5 / 5.0, c 0.5 / 2.0): they show the clearest contrast between
    cautious and adventurous settings. The middle values (tau 1.0, c 1.0)
    remain available in all_variants().
    """
    return [
        EpsilonGreedy(0.01), EpsilonGreedy(0.1), EpsilonGreedy(0.3),
        DecayingEpsilonGreedy(start=1.0, floor=0.05, episodes_to_floor=2000),
        Softmax(0.5), Softmax(5.0),
        UCB(0.5), UCB(2.0),
    ]
