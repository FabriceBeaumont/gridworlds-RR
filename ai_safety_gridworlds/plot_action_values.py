from typing import List, Dict, Set, Tuple, Any
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import save_npz, load_npz, csc_matrix, csr_matrix
import pandas as pd

from os import listdir
from os.path import isfile, join

import constants as c

def visualize_qtable(save_path: str = None, path: str = None, q_table: np.array = None):
    if q_table is None:
        q_table = np.load(path)

    plt.figure(figsize=(14, 10))
    plt.imshow(q_table, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Q-Value')
    plt.title('Q-Table Visualization')
    plt.xlabel('Actions')
    plt.ylabel('States')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_topNq_states(path: str = '', num_states: int = 100, q_table: np.array = None):
    if q_table is None:
        q_table = np.load(path, allow_pickle=True)

    # Compute the average Q-value for each state
    avg_q_values = q_table.mean(axis=1)

    # Get the indices of the states with the highest average Q-values
    top_states = np.argpartition(avg_q_values, -num_states)[-num_states:]

    # Extract the corresponding rows from the Q-table
    top_q_table = q_table[top_states, :]

    visualize_qtable(q_table)

def get_percentage_of_zero(path: str = '', q_table: np.array = None) -> Tuple[float, float]:
    """Returns the percentage of entries which are not zero, 
    and the percentage of rows which do contain at least one non zero value.
    """
    if q_table is None:
        q_table: np.array = np.load(path, allow_pickle=True)

    n = q_table.size
    nzero_percent = np.count_nonzero(q_table) / n * 100
    not_null_rows = q_table[np.any(q_table, axis=1)]
    nzero_rows_percent = len(not_null_rows) / n * 100
    return nzero_percent, nzero_rows_percent

    

if __name__ == "__main__":

    path = f"{c.RESULTS_DIR}/env_name/method_name/dir_time_tag/qtable.npy"
    path = "/home/fabrice/Documents/coding/ML/Results/Environments.SOKOCOIN2/RRLearning/2023_07_20-22_40_e10000_b0-1_blStarting/qtable.npy"

    # TODO: get percentage of non zero rows

    q_table: np.array = np.load(path, allow_pickle=True)
    x: csc_matrix = None # np.load_npz(path,  allow_pickle=True)
    # q_table[:100,:]
    not_null_rows = q_table[np.any(q_table, axis=1)]
    fig = px.imshow(not_null_rows)
    fig.show()

    # # TODO: GOAL: Bar Chart for differences between two qtables. Hope: Vanishing bar sizes.

    # q_tables_dir: str   = "QTables"
    # env_name: str       = "TestEnv"
    # selection_frequency: int    = 10

    # # TODO: Read in the nr episodes from the file name
    # qtable_files = [f for f in listdir(q_tables_dir) if isfile(join(q_tables_dir, f))]
    # qtable_selected_files = qtable_files[0::len(qtable_files)//selection_frequency]

    # for id0, file_name0 in enumerate(qtable_selected_files[:-1]):
    #     episode0 = id0 * selection_frequency
    #     episode1 = (id0 + 1) * selection_frequency
    #     file_name1 = qtable_selected_files[id0 + 1]
    #     q_table0 = pd.DataFrame(np.load(f"{q_tables_dir}/{file_name0}"))
    #     q_table1 = pd.DataFrame(np.load(f"{q_tables_dir}/{file_name1}"))

    #     diff_df = q_table0 - q_table1
    #     # Add an index column to count/name the number of states.
    #     diff_df = diff_df.reset_index()
    #     # Rename the columns based on the Actions Dict: 0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "NOOP".
    #     actions: List[str] = ["Up", "Down", "Left", "Right", "NOOP"]
    #     state_ids: str = "State ids"
    #     diff_df.columns.values[0] = state_ids
    #     diff_df.columns.values[1] = actions[0]
    #     diff_df.columns.values[2] = actions[1]
    #     diff_df.columns.values[3] = actions[2]
    #     diff_df.columns.values[4] = actions[3]
    #     diff_df.columns.values[5] = actions[4]
    #     # Plot the performance data and store it to image.
    #     plot_filename = f"qtable_diff_{episode0}_{episode1}.jpeg"

    #     fig = px.bar(diff_df, x=state_ids, y=actions, title=f"Action-Value differences - e{episode0}-e{episode1} - {env_name}")
    #     fig.write_image(f"{q_tables_dir}/{plot_filename}")
