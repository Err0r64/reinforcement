# CSS 382 - Reinforcement Learning Project Report



## Q1 - Value Iteration

Implemented core value iteration in `ValueIterationAgent`.

- `runValueIteration`:
   - Iterates for the configured number of passes.
   - Uses a copy of previous values each iteration (batch update behavior).
   - Computes each state's new value as the max expected Q-value over legal actions.
   - Assigns terminal/no-action states value 0.
- `computeQValueFromValues`:
   - Computes $Q(s,a)=\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V(s')\right]$.
- `computeActionFromValues`:
   - Returns the action with highest Q-value.
   - Returns `None` if no legal actions exist.

## Q2 - Bridge Crossing Analysis

Completed `question2()` in `analysis.py` with:

- `answerDiscount = 0.9`
- `answerNoise = 0`

These settings prioritize future reward while removing transition randomness so the optimal policy is willing to take the narrow high-reward path.

## Q3 - Policy Preference Analysis

Completed `question3a` through `question3e` in `analysis.py` by choosing discount, noise, and living reward values for each target policy.

- `question3a`: `(1, 0.1, -4)`
   - Strongly negative living reward encourages quick termination.
   - Some noise discourages dangerous long routes.
- `question3b`: `(0.2, 0.2, -2)`
   - More short-sighted and somewhat risk-averse.
   - Still favors ending sooner due to negative living reward.
- `question3c`: `(0.9, 0, 0)`
   - Long-term planning with deterministic transitions.
   - No per-step penalty or bonus.
- `question3d`: `(0.9, 0.2, 0)`
   - Same long-term planning as 3c, but with uncertainty that shifts policy away from risky transitions.
- `question3e`: `(1, 0, 4)`
   - Positive living reward makes continuing to live attractive, favoring avoidance of terminal exits.

## Q4 - Asynchronous Value Iteration

Implemented `AsynchronousValueIterationAgent.runValueIteration` in `valueIterationAgents.py`.

- Cycles through states one at a time using modulo indexing.
- Updates only one state per iteration.
- Skips terminal and no-action states.
- Uses current value table for updates (in-place asynchronous behavior).

## Q5 - Prioritized Sweeping

Implemented `PrioritizedSweepingValueIterationAgent.runValueIteration` in `valueIterationAgents.py`.

- Builds predecessor sets for each state.
- Initializes a priority queue with priority based on Bellman error magnitude.
- Repeatedly pops highest-priority state and updates it.
- Recomputes predecessor priorities when their error exceeds `theta`.

This targets updates where value changes matter most, improving convergence efficiency.

## Q6 - Q-Learning Core

Implemented core tabular Q-learning methods in `QLearningAgent` (`qlearningAgents.py`).

- `getQValue`: returns stored Q-value (defaults to 0 for unseen pairs through `util.Counter`).
- `computeValueFromQValues`: returns $\max_a Q(s,a)$ or 0 if terminal.
- `computeActionFromQValues`: selects a best action; breaks ties randomly.
- `update`: performs
   - $Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha\left[r + \gamma\max_{a'}Q(s',a')\right]$.

## Q7 - Epsilon-Greedy Behavior

Implemented exploration policy in `QLearningAgent.getAction`.

- Returns `None` when no legal action exists.
- With probability `epsilon`, chooses a random legal action.
- Otherwise chooses greedy action from current Q-values.

This provides the required exploration/exploitation tradeoff.

## Q8 - Epsilon and Learning Rate Analysis

Completed `question8()` in `analysis.py` as:

- `return 'NOT POSSIBLE'`

Reason: no single fixed pair `(epsilon, alpha)` can satisfy the required guarantee under the assignment conditions. Which I am assuming to be: Find the optimal policy in under 50 episodes.

## Q9 - Pacman Q-Learning Performance

This question evaluates trained Pacman behavior using the implemented Q-learning logic.

Relevant work comes from:

- `QLearningAgent` update and action-selection methods.
- `PacmanQAgent` parameter wiring (`epsilon`, `gamma`, `alpha`, `numTraining`) and action reporting.

No additional new method implementation was required beyond the Q-learning methods above.

## Q10 - Approximate Q-Learning

Implemented `ApproximateQAgent` methods in `qlearningAgents.py`.

- `getQValue`:
   - Computes weighted feature dot product: $Q(s,a)=\sum_i w_i f_i(s,a)$.
- `update`:
   - Computes temporal-difference error:
      - $\delta = \left[r + \gamma V(s')\right] - Q(s,a)$
   - Updates each weight:
      - $w_i \leftarrow w_i + \alpha\,\delta\,f_i(s,a)$

This enables generalization across state-action pairs via features rather than a fully tabular representation.
