import plotly.express as px
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])   # x-axis values
y = np.array([10, 8, 6, 4, 2])  # y-axis values

diff = x-y

discount = 0.9
print(str(discount).replace(".", "-"))


env_path: str = f"A/B/CD/9034?38901_0:2323.323"
path_names = env_path.split("/")
for i, _ in enumerate(path_names):
    print("/".join(path_names[0:i]))

# rr = np.avg(np.max(coverage_table[_baseline_state_id, :] - coverage_table[_current_state_id, :], 0, axis=0))
