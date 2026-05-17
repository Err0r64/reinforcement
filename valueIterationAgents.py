# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
        Run batch value iteration for the configured number of iterations.

        The method performs synchronous updates: each iteration computes a
        complete new value function from the previous iteration's values, then
        replaces self.values after all states are processed.

        Update rule for each non-terminal state s:
            V_{k+1}(s) = max_a sum_{s'} P(s'|s,a)
                                * (R(s,a,s') + discount * V_k(s'))

        Terminal states, and states with no legal actions, are assigned value 0
        for that iteration.
        """
        states = self.mdp.getStates()

        for _ in range(self.iterations):
            # Snapshot previous iteration values for synchronous (batch) updates.
            oldValues = util.Counter()
            oldValues.update(self.values)
            newValues = util.Counter()

            for state in states:
                # Terminal states have fixed value 0.
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue

                actions = self.mdp.getPossibleActions(state)
                # States with no legal actions are treated as terminal.
                if len(actions) == 0:
                    newValues[state] = 0
                    continue

                qValues = []
                for action in actions:
                    qValue = 0
                    # Bellman expectation over stochastic next states.
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, nextState)
                        qValue += prob * (reward + self.discount * oldValues[nextState])
                    qValues.append(qValue)

                # Greedy policy improvement step for this state.
                newValues[state] = max(qValues)

            # Commit all state updates simultaneously.
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                reward = self.mdp.getReward(state, action, nextState)
                qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
                return None

        bestAction = None
        bestValue = float('-inf')
        for action in actions:
                qValue = self.computeQValueFromValues(state, action)
                if qValue > bestValue:
                        bestValue = qValue
                        bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        if len(states) == 0:
            return

        for i in range(self.iterations):
            state = states[i % len(states)]
            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            if len(actions) == 0:
                continue

            self.values[state] = max(self.computeQValueFromValues(state, action) for action in actions)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        predecessors = {}
        for state in states:
            predecessors[state] = set()

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

        priorityQueue = util.PriorityQueue()
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            if len(actions) == 0:
                continue
            bestQValue = max(self.computeQValueFromValues(state, action) for action in actions)
            diff = abs(self.values[state] - bestQValue)
            priorityQueue.update(state, -diff)

        for _ in range(self.iterations):
            if priorityQueue.isEmpty():
                break

            state = priorityQueue.pop()
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                if len(actions) > 0:
                    self.values[state] = max(self.computeQValueFromValues(state, action) for action in actions)

            for predecessor in predecessors[state]:
                if self.mdp.isTerminal(predecessor):
                    continue
                actions = self.mdp.getPossibleActions(predecessor)
                if len(actions) == 0:
                    continue
                bestQValue = max(self.computeQValueFromValues(predecessor, action) for action in actions)
                diff = abs(self.values[predecessor] - bestQValue)
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)

