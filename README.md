# tinyRL

## Policy

Policy is simply a function that tells us what action to take in a given state.

We usually write it as:

$\pi(a \mid s)$

Which means: probability of taking action `a` given state `s`.

So policy is not the action itself.
It is a probability distribution over actions.

If we use a neural network, it takes state as input and outputs probabilities for each action.
That network is our policy.

## Policy Gradient

Policy gradient means we directly optimize the policy.

Instead of learning value first and then deriving policy (like Q-learning), we directly change policy parameters to increase expected return.

We define objective:

$$J(\theta) = \mathbb{E}[R]$$

We want to maximize this.

So we compute:

$$
\nabla_\theta J(\theta)
$$

Core idea:

* Increase probability of actions that give high return
* Decrease probability of actions that give low return

We update parameters in direction of gradient.


## REINFORCE

REINFORCE is the simplest policy gradient algorithm.

It is:

* Monte Carlo based (uses full episode return)
* No value function (unless baseline added)
* Unbiased gradient estimator
* High variance

The update rule:

$$
\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t
$$

Where:

* $( G_t )$ = return from time step t
* $( \log \pi_\theta(a_t \mid s_t) )$ = log probability of taken action

We sample a full episode.
For each step, we compute return from that step onward.
Then multiply log-probability with that return.

This increases probability of actions that resulted in high return.


### Loss in REINFORCE

In REINFORCE, our loss function is basically:

$$
\log \pi_\theta(a_t \mid s_t) \cdot G_t
$$

**Why probability?**

Because we want to increase probability of good actions.
If an action gives high return, we want the model to choose it more often.

Returns give us overall idea of how the agent performed after taking action `a` in state `s`.

But there is problem.

There is no relative reference point to know whether reward is actually good or bad.

We only see raw returns.

Because of that:

* Gradients have high variance
* Training becomes unstable
* Loss oscillates


### Baseline

To reduce variance, we introduce baseline.

Baseline is a base reward / return.

Instead of using raw return, we subtract baseline:

$$
G_t - b
$$

New update:

$$
\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b)
$$

This gives us relative improvement.

Now we update based on:

* How much better the return is compared to expectation
* Not absolute reward

**Important**

Subtracting baseline does NOT change expected gradient.
It only reduces variance.

But baseline does not have full information about future.
If it is just mean return or simple estimate, it cannot predict next step properly.

So REINFORCE still has:

* High variance
* Slow learning

Baseline helps.
But it is not best possible solution.

That is why Actor-Critic methods use value function:

$$
A(s, a) = G_t - V(s)
$$

Where ( V(s) ) tries to approximate expected return from that state.

This gives better signal than simple mean baseline.

**Core limitation of REINFORCE**

Unbiased but high variance.

And in RL, most things are about bias–variance tradeoff.
