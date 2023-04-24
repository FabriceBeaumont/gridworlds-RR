# Attainable Utility Preservation

A test-bed for the [Attainable Utility Preservation](https://arxiv.org/abs/1902.09725) method for quantifying and penalizing the change an agent has on the world around it. This repository further augments [this expansion](https://github.com/side-grids/ai-safety-gridworlds) to DeepMind's [AI safety gridworlds](https://github.com/deepmind/ai-safety-gridworlds). For discussion of AUP's potential contributions to long-term AI safety, see [here](https://www.lesswrong.com/s/7CdoznhJaLEKHwvJW).

## Installation
1. Using Python 2.7 as the interpreter, acquire the libraries in `requirements.txt`.
2. Clone using `--recursive` to snag the `pycolab` submodule:
`git clone --recursive https://github.com/alexander-turner/attainable-utility-preservation.git`.
3. Run `python -m experiments.charts` or `python -m experiments.ablation`, tweaking the code to include the desired environments. 

## Environments

>Our environments are Markov Decision Processes. All environments use a grid of
size at most 10x10. Each cell in the grid can be empty, or contain a wall or
other objects... The agent is located in one cell on
the grid and in every step the agent takes one of the actions from the action
set A = {`up`, `down`, `left`, `right`, `no-op`}. Each action modifies the agent's position to
the next cell in the corresponding direction unless that cell is a wall or
another impassable object, in which case the agent stays put.

>The agent interacts with the environment in an episodic setting: at the start of
each episode, the environment is reset to its starting configuration (which is
possibly randomized). The agent then interacts with the environment until the
episode ends, which is specific to each environment. We fix the maximal episode
length to 20 steps. Several environments contain a goal cell... If
the agent enters the goal cell, it receives a reward of +1 and the episode
ends.

>In the classical reinforcement learning framework, the agent's objective is to
maximize the cumulative (visible) reward signal. While this is an important part
of the agent's objective, in some problems this does not capture everything that
we care about. Instead of the reward function, we evaluate the agent on the
performance function *that is not observed by the agent*. The performance
function might or might not be identical to the reward function. In real-world
examples, the performance function would only be implicitly defined by the
desired behavior the human designer wishes to achieve, but is inaccessible to
the agent and the human designer.


### `Box`
![](https://i.imgur.com/lfPdzOB.png)
![](https://i.imgur.com/Khg8gQV.gif)
---

### `Dog`
![](https://i.imgur.com/Iy8RcrL.png)
![](https://i.imgur.com/4xwQqNr.gif)
---

### `Survival`
![](https://i.imgur.com/wyGnyql.png)
![](https://i.imgur.com/SEhU3Jx.gif)
---

### `Conveyor`
![](https://i.imgur.com/wR9KiaQ.png)
![](https://i.imgur.com/9B2yebO.gif)
---

### `Vase`
![](https://i.imgur.com/Xnox0zO.png)
![](https://i.imgur.com/N8a1FsA.gif)
---

### `Sushi`
![](https://i.imgur.com/Nz0EVuY.png)
![](https://i.imgur.com/DEIOM03.gif)

The `Conveyor-Sushi` variant induces similar behavior:

![](https://i.imgur.com/5QE0sao.gif)

_Due to the larger state space, the attainable set Q-values need more than the default 4,000 episodes to converge and induce interference behavior in Starting state._
***

### `Burning`
![](https://i.imgur.com/fLzCzX2.png)
![](https://i.imgur.com/WeD5xUx.gif)

