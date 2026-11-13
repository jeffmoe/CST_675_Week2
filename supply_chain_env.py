"""
Pharmaceutical distribution environment for tabular Q-learning.

Built on Gymnasium (``import gymnasium as gym``), the maintained successor to
OpenAI Gym. OpenAI stopped maintaining Gym, and the Farama Foundation now
develops the same API as Gymnasium. Differences from classic Gym that this
file relies on:
  * the base class is ``gym.Env`` (capital E)
  * ``reset(seed=...)`` seeds ``self.np_random`` for reproducible runs
  * ``step()`` returns ``(obs, reward, terminated, truncated, info)``. A time
    limit is reported as ``truncated``, not as a terminal state.

Scenario
--------
A central warehouse supplies three locations with one temperature-controlled
medication. Each simulated day the agent can send one shipment to one
location: it chooses the destination, the quantity, and the route. The goal
is to keep every location stocked without overpaying for transport or
holding too much perishable stock.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
# Hospitals have higher demand and are less predictable: on a "surge" day
# (e.g. an admissions spike) their Poisson mean doubles. The retail pharmacy
# has lower, steadier demand. Stockouts at hospitals are penalized more
# because they affect patient care directly.
LOCATIONS = ["Hospital A", "Hospital B", "Retail Pharmacy"]
DEMAND_MEAN = np.array([8.0, 6.0, 4.0])        # base units/day
SURGE_PROB = np.array([0.15, 0.15, 0.0])       # chance a day is a surge day
SURGE_MULT = 2.0                               # surge-day multiplier on mean
STOCKOUT_PENALTY = np.array([25.0, 25.0, 10.0])  # $ per unit of unmet demand

# Inventory buckets in "days of supply" (on-hand / base daily mean demand):
#   0 critical: < 1 day     1 low: 1-3 days     2 ok: 3-6 days
#   3 high:     6-9 days    4 over: >= 9 days
BUCKET_NAMES = ["critical", "low", "ok", "high", "over"]
BUCKET_EDGES_DAYS = np.array([1.0, 3.0, 6.0, 9.0])
N_BUCKETS = len(BUCKET_NAMES)

# Storage limit per location (cold-chain fridge space), in days of supply.
# Units delivered above this cap are wasted.
CAPACITY_DAYS = 12.0
CAPACITY = np.floor(DEMAND_MEAN * CAPACITY_DAYS).astype(int)   # [96, 72, 48]

# ---------------------------------------------------------------------------
# Shipments
# ---------------------------------------------------------------------------
QUANTITIES = {"small": 10, "medium": 20, "large": 40}

# route: lead time (days), fixed cost per shipment ($), cost per unit ($),
#        probability of a 1-day delay
ROUTES = {
    "Express (direct)":     dict(lead=1, fixed=60.0, per_unit=3.0, delay_prob=0.0),
    "Consolidated (truck)": dict(lead=3, fixed=20.0, per_unit=1.0, delay_prob=0.15),
}

HOLDING_COST = 0.25    # $ per unit on hand per day (refrigerated storage)
WASTE_COST = 15.0      # $ per unit that arrives with no room to store it
EPISODE_DAYS = 30
REWARD_SCALE = 100.0   # reward = -cost / 100 keeps Q-values in a small range


def _build_action_table():
    """Action 0 = ship nothing. Every other action = (destination, quantity, route)."""
    table = [None]
    for dest in range(len(LOCATIONS)):
        for q_name in QUANTITIES:
            for route in ROUTES:
                table.append((dest, q_name, route))
    return table


ACTION_TABLE = _build_action_table()           # 1 + 3 x 3 x 2 = 19 actions


class PharmaSupplyChainEnv(gym.Env):
    """One warehouse, three locations, one shipment decision per day."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.n_locations = len(LOCATIONS)
        self.n_states = N_BUCKETS ** self.n_locations       # 5^3 = 125
        self.n_actions = len(ACTION_TABLE)                   # 19
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        self.render_mode = render_mode

    # ------------------------------------------------------------------ state
    def bucket(self, loc: int, units: int) -> int:
        """Map one location's on-hand units to a 0-4 bucket index."""
        days_of_supply = units / DEMAND_MEAN[loc]
        return int(np.searchsorted(BUCKET_EDGES_DAYS, days_of_supply, side="right"))

    def encode_state(self) -> int:
        """Map (bucket_A, bucket_B, bucket_R) to one integer in [0, 125)
        so it can index a row of the Q-table (base-5 number)."""
        code = 0
        for loc in range(self.n_locations):
            code = code * N_BUCKETS + self.bucket(loc, self.inventory[loc])
        return code

    @staticmethod
    def decode_state(code: int) -> list[str]:
        """Inverse of encode_state, for readable logs and policy tables."""
        names = []
        for _ in range(len(LOCATIONS)):
            names.append(BUCKET_NAMES[code % N_BUCKETS])
            code //= N_BUCKETS
        return names[::-1]

    # ----------------------------------------------------------------- action
    @staticmethod
    def decode_action(action: int) -> dict:
        """Map an action index to {destination, quantity, route}."""
        entry = ACTION_TABLE[int(action)]
        if entry is None:
            return {"destination": None, "quantity": 0, "route": None,
                    "label": "No shipment"}
        dest, q_name, route = entry
        qty = QUANTITIES[q_name]
        return {"destination": dest, "quantity": qty, "route": route,
                "label": f"{qty} units to {LOCATIONS[dest]} via {route}"}

    # --------------------------------------------------------------- Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)          # seeds self.np_random
        self.day = 0
        # start each location with 3-6 days of supply (the "ok" band)
        self.inventory = np.array([
            int(self.np_random.integers(int(3 * m), int(6 * m) + 1))
            for m in DEMAND_MEAN
        ])
        # in-transit shipments: dicts with dest, qty, promised_day, arrival_day
        self.pipeline: list[dict] = []
        return self.encode_state(), self._info_base()

    def step(self, action: int):
        assert self.action_space.contains(int(action)), f"invalid action {action}"
        act = self.decode_action(action)

        # 1. dispatch today's shipment (if any) and pay for transport
        transport_cost = 0.0
        if act["route"] is not None:
            r = ROUTES[act["route"]]
            transport_cost = r["fixed"] + r["per_unit"] * act["quantity"]
            promised = self.day + r["lead"]
            delayed = self.np_random.random() < r["delay_prob"]
            self.pipeline.append({
                "dest": act["destination"], "qty": act["quantity"],
                "promised_day": promised,
                "arrival_day": promised + (1 if delayed else 0),
            })

        # 2. advance to the next day; shipments due arrive before demand
        self.day += 1
        on_time, total_deliveries = 0, 0
        wasted = np.zeros(self.n_locations, dtype=int)
        remaining = []
        for s in self.pipeline:
            if s["arrival_day"] <= self.day:
                total_deliveries += 1
                on_time += int(s["arrival_day"] <= s["promised_day"])
                d = s["dest"]
                self.inventory[d] += s["qty"]
                overflow = max(0, self.inventory[d] - CAPACITY[d])
                self.inventory[d] -= overflow
                wasted[d] += overflow
            else:
                remaining.append(s)
        self.pipeline = remaining

        # 3. stochastic demand at each location (each location is
        #    handled separately, so a shipment only changes its destination)
        surge = self.np_random.random(self.n_locations) < SURGE_PROB
        means = DEMAND_MEAN * np.where(surge, SURGE_MULT, 1.0)
        demand = self.np_random.poisson(means)
        served = np.minimum(demand, self.inventory)
        unmet = demand - served
        self.inventory = self.inventory - served

        # 4. costs
        holding_cost = HOLDING_COST * float(self.inventory.sum())
        stockout_cost = float((STOCKOUT_PENALTY * unmet).sum())
        waste_cost = WASTE_COST * float(wasted.sum())
        total_cost = transport_cost + holding_cost + stockout_cost + waste_cost
        reward = -total_cost / REWARD_SCALE

        # 5. fixed-length episode: never terminate on stockout, only truncate
        terminated = False
        truncated = self.day >= EPISODE_DAYS

        info = self._info_base()
        info.update({
            "action": act["label"],
            "route": act["route"],
            "quantity": act["quantity"],
            "destination": act["destination"],
            "transport_cost": transport_cost,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "waste_cost": waste_cost,
            "total_cost": total_cost,
            "stockout_units": int(unmet.sum()),
            "stockout_units_by_location": unmet.tolist(),
            "demand_units": int(demand.sum()),
            "demand_units_by_location": demand.tolist(),
            "wasted_units": int(wasted.sum()),
            "on_time_deliveries": on_time,
            "total_deliveries": total_deliveries,
        })
        return self.encode_state(), reward, terminated, truncated, info

    def _info_base(self) -> dict:
        return {
            "day": self.day,
            "inventory": self.inventory.tolist(),
            "buckets": [BUCKET_NAMES[self.bucket(i, u)]
                        for i, u in enumerate(self.inventory)],
            "in_transit_units": int(sum(s["qty"] for s in self.pipeline)),
        }

    def render(self):
        if self.render_mode == "ansi":
            parts = [f"{n}: {u} ({BUCKET_NAMES[self.bucket(i, u)]})"
                     for i, (n, u) in enumerate(zip(LOCATIONS, self.inventory))]
            return f"Day {self.day:2d} | " + " | ".join(parts)
        return None
