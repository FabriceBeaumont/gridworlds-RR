# Gym AI Safety Gridworlds
This is a repository that serves as a base for RL algorithm testing.
It consists of following sub repos:
  - [gym-ai-safety-gridworlds](https://github.com/n0p2/gym_ai_safety_gridworlds)
  - [ai-safety-gridworlds](https://github.com/deepmind/ai-safety-gridworlds)
  - [ai-safety-gridworlds-viewer](https://github.com/n0p2/ai-safety-gridworlds-viewer)



## Get started
```shell
# Create virtual environment
python3 -m venv .venv
# Windows
.\venv\Scripts\activate
# OR Linux
source ./venv/bin/acivate


# Install dependencies
pip install -r requirements.txt
# OR
pip install absl-py numpy pycolab tensorflow gym


# Run example
python -m gym_ai_safety_gridworlds.examples.env_example
```