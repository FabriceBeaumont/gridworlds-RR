import plotly.express as px
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])   # x-axis values
y = np.array([10, 8, 6, 4, 2])  # y-axis values

diff = x-y
print(diff)

diff[diff<0] = 0
print(diff)
print(np.sum(diff))
print(len(diff))

print(np.mean(diff))


# rr = np.avg(np.max(coverage_table[_baseline_state_id, :] - coverage_table[_current_state_id, :], 0, axis=0))
