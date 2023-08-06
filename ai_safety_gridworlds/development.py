from typing import List, Dict, Set, Tuple
import plotly.express as px
import numpy as np
from collections import Counter
import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import re


def gif_frame_creation():
    states_list = np.load("/home/fabrice/Documents/coding/ML/GridworldResults/sokocoin0/RRLearning/2023_08_06-21_20_e100_lr0-1_SEst_blStar_g0-99_b0-1/agent_journey_n5.npy")
    states: List[np.array] = []
    for state in states_list:
        # Eliminate matrix encoding strings.
        state_rows: List[str] = state.replace('[','').replace(']','').replace(' ','').split("\n")
        # Remove the tailing float-dot and split the other values using them.
        state_matrix: List[List[str]] = [row[:-1].split(".") for row in state_rows]
        # Convert the strings into integers and save each state as np.array.
        state_matrix: np.array = np.array([[int(x) for x in row] for row in state_matrix])
        states.append(state_matrix)

    # Define the colormap for the states.
    colors = ["black", "silver", "gold", "peru", "green"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    v_min: int = min([state.min() for state in states])
    v_max: int = max([state.max() for state in states])
    
    state = states[0]
    lr_str    = f"Learning rate: x.x"
    sss_str   = f"State set space strategy 'X'"
    bl_str    = f"Baseline: 'X'"
    dic_str   = f"Discount factor: x.x"
    beta_str  = f"Beta: x.x"
    lperf_str = f"Last performance: 100x0"
    lref_str  = f"Last reward: 45x0"
    info_text = f"{lr_str}\n{sss_str}\n{bl_str}\n{dic_str}\n{beta_str}\n{lperf_str}\n{lref_str}"


    ############################################################################################################

    fig, (ax_state, ax_info) = plt.subplots(1, 2, figsize=(5, 3))
    fig.suptitle(f"RR-Learning Agent envname")
    ax_state.matshow(state, cmap=cmap1, vmin=v_min, vmax=v_max)
    
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            value = state[i,j]
            ax_state.text(j, i, str(value), va='center', ha='center')
    
    for frame in [ax_state.axes, ax_info.axes]:
        frame.get_xaxis().set_ticks([])
        frame.get_yaxis().set_ticks([])        
    ax_state.set_xlabel(f"State: 0/{len(states)-1}")

    border = 0.05
    ax_info.text(border, 1-border, info_text, va='top', ha='left')


    ax_info.spines['top'].set_visible(False)
    ax_info.spines['right'].set_visible(False)
    ax_info.spines['bottom'].set_visible(False)
    ax_info.spines['left'].set_visible(False)

    # Optionally, you can remove the ticks as well
    # ax_info.tick_params(axis='both', which='both', length=0)
    


    plt.tight_layout()
    plt.savefig('Test.png')
    plt.close()

if __name__ == "__main__":
    gif_frame_creation()
