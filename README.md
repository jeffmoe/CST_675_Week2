# Pharmaceutical Supply Chain RL: Exploration vs. Exploitation

A proof of concept in which a Q-learning agent decides daily shipments from a central warehouse to two hospitals and a retail pharmacy. It compares four exploration strategies (fixed ε-greedy, decaying ε-greedy, softmax and UCB) on distribution cost, stockout rate and delivery efficiency. Every number below comes from `results/summary.csv`.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install numpy gymnasium pandas matplotlib
python supply_chain_env.py        # prints the reward sanity-check table
python experiments.py             # trains and evaluates all strategies (~8 min)
python make_figures.py            # writes the PNG charts to figures/
```

All runs use fixed seeds, so rerunning reproduces the same results.

| File | Purpose |
|---|---|
| `supply_chain_env.py` | Gymnasium environment and reward function |
| `q_learning.py` | Q-table, Bellman update, `train()` and `evaluate()` |
| `strategies.py` | ε-greedy, decaying ε, softmax and UCB |
| `experiments.py` | Runs every strategy across 5 seeds and saves the CSVs |
| `make_figures.py` | Builds the four figures from the CSVs |

## How it works

**Environment.** Built on Gymnasium, the maintained successor to OpenAI Gym (Johnny Code, 2023).
- **Locations:** Hospital A, Hospital B and a Retail Pharmacy, with Poisson demand averaging 8, 6 and 4 units per day. On 15% of days, hospital demand doubles.
- **State (512):** for each location, a stock level in days of supply (critical <1, low 1–3, ok 3–7, high ≥7) plus a flag for whether a shipment is already on its way.
- **Actions (19):** ship nothing, or choose a destination × a quantity (10, 20 or 40 units) × a route. The routes are:
  - Express: arrives in 1 day, costs $60 + $3 per unit.
  - Consolidated truck: arrives in 3 days, costs $20 + $1 per unit, with a 15% chance of a one-day delay.
- **Episodes:** fixed 30-day months. A stockout is penalized but never ends the episode.

**Reward.** −(transport + holding + stockout + late + overstock) / 100. Stockout is the largest penalty because a missed dose is the costliest failure in pharma.

| Cost term | Weight |
|---|---|
| Stockout | $50 per unit at a hospital, $20 at the pharmacy |
| Overstock above capacity | $15 per unit |
| Late shipment | $40 per shipment |
| Holding | $0.25 per unit per day |

Dividing by 100 keeps Q-values in the tens, which matters for softmax.

**Q-learning.** Q(s,a) ← Q(s,a) + α[r + γ·maxₐ′ Q(s′,a′) − Q(s,a)], derived from the Bellman equation (Mahesh Huddar, 2022; CodeEmporium, 2023).
- **Learning rate α = 0.1.**
- **Discount factor γ = 0.95.** A truck ordered today lands 3–4 days later, so future costs must still count.
- **Starting Q-values of −20.** That's a realistic starting guess, since every true value is negative.
- **6,000 training episodes per run.**

**Exploration strategies** (Rocca, 2021; Deeplizard, 2018)

| Strategy | Settings | How it decides to explore |
|---|---|---|
| Fixed ε-greedy | ε = 0.01, 0.1, 0.3 | Random action with a fixed probability ε, forever |
| Decaying ε-greedy | 1.0 → 0.05 by episode 4,000 | Explores heavily at first, less with experience |
| Softmax | τ = 0.5, 5 | Chooses actions in proportion to exp(Q/τ) |
| UCB | c = 0.5, 2 | Adds a bonus to rarely tried actions (ritvikmath, 2020) |

**Evaluation.**
- Each strategy trains on 5 seeds.
- Each trained policy is then scored greedily (no exploration) on 100 held-out months. Training and test seeds never overlap (Kozyrkov, 2020; Buhl, 2024).
- Two baselines are scored on the same months:
  - a random policy
  - a reorder-to-target heuristic: ship by truck to whichever location is furthest below 7 days of supply

**How we got here.** In the first version (125 states, Q-values starting at 0, 3,000 episodes), the best agent cost $4,724 a month against the heuristic's $3,075. The agents couldn't see shipments already in transit, so they kept re-ordering. Three changes fixed most of that:
- adding the in-transit flag
- starting Q-values at −20
- training for 6,000 episodes

## Results

Greedy evaluation on 100 held-out months, mean ± standard deviation across 5 seeds.

| Policy | Cost / month | Stockout rate | On-time delivery |
|---|---|---|---|
| ε-greedy 0.01 | $4,185 ± $319 | 6.0% ± 1.2% | 91.5% |
| ε-greedy 0.1 | $3,708 ± $249 | 4.7% ± 1.5% | 91.9% |
| **ε-greedy 0.3** | **$3,486 ± $223** | **4.1% ± 0.7%** | 91.5% |
| Decaying ε | $3,923 ± $396 | 5.3% ± 1.8% | 91.5% |
| Softmax τ=0.5 | $4,035 ± $295 | 5.6% ± 1.4% | **94.6%** |
| Softmax τ=5 | $12,368 ± $4,242 | 42.0% ± 18.2% | 93.9% |
| UCB c=0.5 | $3,670 ± $204 | 5.3% ± 1.6% | 92.2% |
| UCB c=2 | $4,243 ± $380 | 8.6% ± 2.8% | **94.6%** |
| Reorder heuristic | $3,075 | 7.0% | 84.9% |
| Random policy | $9,895 ± $234 | 16.3% ± 0.7% | 92.5% |

Figures: [learning curves](figures/1_learning_curves.png) · [evaluation bars](figures/2_evaluation_bars.png) · [ε sensitivity](figures/3_epsilon_sensitivity.png) · [exploration over time](figures/4_exploration_over_time.png)

## Analysis

**Best performer by metric.**
- **Cost and stockouts:** fixed ε-greedy at ε = 0.3 was best on both ($3,486 ± $223 a month; 4.1% ± 0.7%).
- **Delivery efficiency:** softmax τ = 0.5 and UCB c = 2 tied at 94.6%.

These leads are small compared with the seed-to-seed variation. ε = 0.3, UCB c = 0.5 ($3,670 ± $204) and ε = 0.1 ($3,708 ± $249) overlap within one standard deviation, so five seeds can't separate the top three.

The differences that do exceed the noise are against the baselines:
- Every policy except softmax τ = 5 beat the random policy.
- ε = 0.3 cut stockouts from the heuristic's 7.0% to 4.1%, a gap of about four standard deviations.

The heuristic still cost less ($3,075 a month) because it ships only by truck, while the stronger agents spent about twice as much on transport to buy fewer stockouts.

**Exploration vs. exploitation.** Strategies that kept exploring paid more during training but learned better policies:
- **Fixed ε:** at the end of training, ε = 0.3 had the highest cost of the three fixed values ($4,514 a month, against about $4,040 for ε = 0.01 and 0.1), yet it produced the best test result. ε = 0.01 explored least and tested worst ($4,185; 6.0%). With 512 states, wider exploration reaches more state–action pairs, which fits Rocca's (2021) point that too little exploration locks in early estimates.
- **Decaying ε:** paid the most early on and finished between the fixed values ($3,923).
- **UCB:** aimed its exploration at rarely tried actions (Huang, 2025). Its share of non-greedy actions fell from 57.7% to 28.9% (c = 0.5).
- **Softmax:** never settled. Even τ = 0.5 still picked a non-greedy action 73.9% of the time.

**Trade-offs.**
- **Early exploration is a real operating cost.** The best variants averaged $9,226–$9,854 a month over their first 100 training months, more than twice their final test cost. A production agent should therefore learn in simulation before it touches live orders (EITCA Academy, 2024).
- **Results were sensitive to hyperparameters (Kılıç, 2023).** Softmax went from $4,035 at τ = 0.5 to $12,368 at τ = 5, worse than random. UCB stockouts rose from 5.3% to 8.6% when c went from 0.5 to 2.
- **Softmax depends on reward scale because it exponentiates Q/τ.** With rewards divided by 100, the best actions' Q-values differ by fractions of a unit, so τ = 5 flattens the choice to near-uniform (92% non-greedy). With rewards in raw dollars, the same τ would be almost purely greedy.

**Limitations and next steps.** This proof of concept is deliberately simplified:
- one product and one shipment per day
- a warehouse that never runs out
- synthetic Poisson demand
- a state that shows stock levels and an in-transit flag, but not the quantities in transit or any demand forecast

The learning curves were still falling at 6,000 episodes, and every metric comes from simulated months, not real data, so avoiding overfitting to the simulator matters (*How Do We Avoid Overfitting in Reinforcement Learning?*, n.d.). Next steps:
- a deep Q-network (DQN) to handle richer states (Habib, 2019)
- demand-forecast inputs
- multi-warehouse networks modeled with graph neural networks (Abu Dhabi Machine Learning, 2022)
- continuous validation against live outcomes (Iguazio, 2023)

## References

Abu Dhabi Machine Learning. (2022, October 2). *ADML S2E5 - Optimizing supply chain with reinforcement learning and graph NNs* [Video]. YouTube. https://www.youtube.com/watch?v=fEXmU8cSn14

Buhl, N. (2024, June 28). *Training, validation, test split for machine learning datasets*. Encord. https://encord.com/blog/train-val-test-split/

CodeEmporium. (2023, November 7). *Q-learning - Explained!* [Video]. YouTube. https://www.youtube.com/watch?v=TiAXhVAZQl8

Deeplizard. (2018, October 11). *Exploration vs. exploitation - Learning the optimal reinforcement learning policy* [Video]. YouTube. https://www.youtube.com/watch?v=mo96Nqlo1L8

EITCA Academy. (2024, June 11). *How does the concept of exploration and exploitation trade-off manifest in bandit problems, and what are some of the common strategies used to address this trade-off?* https://eitca.org/artificial-intelligence/eitc-ai-arl-advanced-reinforcement-learning/deep-reinforcement-learning/advanced-topics-in-deep-reinforcement-learning/examination-review-advanced-topics-in-deep-reinforcement-learning/how-does-the-concept-of-exploration-and-exploitation-trade-off-manifest-in-bandit-problems-and-what-are-some-of-the-common-strategies-used-to-address-this-trade-off/

Habib, N. (2019). *Hands-on Q-learning with Python*. Packt Publishing. https://learning.oreilly.com/library/view/hands-on-q-learning-with/9781789345803/

How do we avoid overfitting in reinforcement learning? (n.d.). Quora. https://www.quora.com/How-do-we-avoid-overfitting-in-reinforcement-learning

Huang, Z. (2025, August 15). *Reinforcement learning #1: Multi-armed bandits, explore vs exploit, epsilon-greedy, UCB* [Video]. YouTube. https://www.youtube.com/watch?v=p13V-93aG1c

Iguazio. (2023, November 15). *What is continuous validation*. https://www.iguazio.com/glossary/continuous-validation/

Johnny Code. (2023, June 25). *How to use Q-learning to train Gymnasium FrozenLake-v1 | Python reinforcement learning tutorial #1* [Video]. YouTube. https://www.youtube.com/watch?v=ZhoIgo3qqLU

Kılıç, İ. (2023, September 5). *Understanding hyperparameters in machine learning*. Medium. https://medium.com/@ilyurek/understanding-hyperparameters-in-machine-learning-6ae699fcbdd1

Kozyrkov, C. [Cassie Kozyrkov]. (2020, September 16). *Machine learning: Validation vs testing* [Video]. YouTube. https://www.youtube.com/watch?v=pGlQLMPI46g

Mahesh Huddar. (2022, November 29). *How to use Bellman equation reinforcement learning | Bellman equation machine learning Mahesh Huddar* [Video]. YouTube. https://www.youtube.com/watch?v=3Rx2x2traxw

ritvikmath. (2020, October 5). *Best multi-armed bandit strategy? (feat: UCB method)* [Video]. YouTube. https://www.youtube.com/watch?v=FgmMK6RPU1c

Rocca, J. (2021, April 18). *The exploration-exploitation trade-off: Intuitions and strategies*. Towards Data Science. https://medium.com/data-science/the-exploration-exploitation-dilemma-f5622fbe1e82
