from typing import List, Dict, Set, Tuple, Any
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import save_npz, load_npz, csc_matrix, csr_matrix
import pandas as pd

from os import listdir
from os.path import isfile, join

import constants as c

Q_PATH = '/home/fabrice/Documents/coding/ML/Results/sokocoin0/RRLearning/2023_07_23-17_44_e100_b0-1_blStarting/qtable.npy'
#"/home/fabrice/Documents/coding/ML/Results/Environments.SOKOCOIN2/RRLearning/2023_07_20-22_40_e10000_b0-1_blStarting/qtable.npy"
C_PATH = "/home/fabrice/Documents/coding/ML/Results/Environments.SOKOCOIN2/RRLearning/2023_07_20-22_40_e10000_b0-1_blStarting/ctable.npy"

def visualize_matrix_px(save_path: str = None, path: str = None, table: np.array = None):
    if table is None:
        table = np.load(path)

    fig = px.imshow(table)#, width=800, height=400)        
    # fig.update_layout(
    #     margin=dict(l=20, r=20, t=20, b=20),
    #     paper_bgcolor="LightSteelBlue",
    # )
    fig.write_image(save_path)
    fig.show()    

def visualize_matrix_np(save_path: str = None, path: str = None, table: np.array = None):
    if table is None:
        table = np.load(path)

    plt.figure(figsize=(14, 10))
    plt.imshow(table, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Q-Value')
    plt.title('Q-Table Visualization')
    plt.xlabel('Actions')
    plt.ylabel('States')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_topNc_states(path: str = '', num_states: int = 100, c_table: np.array = None):
    if c_table is None:
        c_table = np.load(path, allow_pickle=True)

    # Compute the average c-value for each state.
    avg_c_row = c_table.mean(axis=1)
    # Get the indices of the states with the highest average c-values.
    top_rows = np.argpartition(avg_c_row, -num_states)[-num_states:]

    avg_c_col = c_table[top_rows, :].mean(axis=0)
    # Get the indices of the states with the highest average c-values.
    top_cols = np.argpartition(avg_c_col, -num_states)[-num_states:]
    
    # Extract the corresponding rows and columns.
    top_q_table = c_table[np.ix_(top_rows, top_cols)]

    visualize_matrix_px(table = top_q_table, save_path='Test.jpg')

def visualize_topNq_states(path: str = '', num_states: int = 100, q_table: np.array = None):
    if q_table is None:
        q_table = np.load(path, allow_pickle=True)

    # Compute the average Q-value for each state
    avg_q_values = q_table.mean(axis=1)

    # Get the indices of the states with the highest average Q-values
    top_states = np.argpartition(avg_q_values, -num_states)[-num_states:]

    # Extract the corresponding rows from the Q-table
    top_q_table = q_table[top_states, :]

    visualize_matrix_px(table = top_q_table, save_path='Test.jpg')

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

# TODO: Delete or refactor
def visualize_action_values():
    q_tables_dir: str   = "QTables"
    env_name: str       = "TestEnv"
    selection_frequency: int    = 10

    # TODO: Read in the nr episodes from the file name
    qtable_files = [f for f in listdir(q_tables_dir) if isfile(join(q_tables_dir, f))]
    qtable_selected_files = qtable_files[0::len(qtable_files)//selection_frequency]

    for id0, file_name0 in enumerate(qtable_selected_files[:-1]):
        episode0 = id0 * selection_frequency
        episode1 = (id0 + 1) * selection_frequency
        file_name1 = qtable_selected_files[id0 + 1]
        q_table0 = pd.DataFrame(np.load(f"{q_tables_dir}/{file_name0}"))
        q_table1 = pd.DataFrame(np.load(f"{q_tables_dir}/{file_name1}"))

        diff_df = q_table0 - q_table1
        # Add an index column to count/name the number of states.
        diff_df = diff_df.reset_index()
        # Rename the columns based on the Actions Dict: 0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "NOOP".
        actions: List[str] = ["Up", "Down", "Left", "Right", "NOOP"]
        state_ids: str = "State ids"
        diff_df.columns.values[0] = state_ids
        diff_df.columns.values[1] = actions[0]
        diff_df.columns.values[2] = actions[1]
        diff_df.columns.values[3] = actions[2]
        diff_df.columns.values[4] = actions[3]
        diff_df.columns.values[5] = actions[4]
        # Plot the performance data and store it to image.
        plot_filename = f"qtable_diff_{episode0}_{episode1}.jpeg"

        fig = px.bar(diff_df, x=state_ids, y=actions, title=f"Action-Value differences - e{episode0}-e{episode1} - {env_name}")
        fig.write_image(f"{q_tables_dir}/{plot_filename}")

if __name__ == "__main__":    
    visualize_topNc_states(path=C_PATH)
    
    # path = f"{c.RESULTS_DIR}/env_name/method_name/dir_time_tag/qtable.npy"
    pass