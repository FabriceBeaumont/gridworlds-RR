from typing import List, Dict, Set, Tuple, Any
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import save_npz, load_npz, csc_matrix, csr_matrix
import pandas as pd
import os

import imageio
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


import constants as c

Q_PATH = '/home/fabrice/Documents/coding/ML/Results/sokocoin0/RRLearning/2023_07_23-17_44_e100_b0-1_blStarting/qtable.npy'
#"/home/fabrice/Documents/coding/ML/Results/Environments.SOKOCOIN2/RRLearning/2023_07_20-22_40_e10000_b0-1_blStarting/qtable.npy"
C_PATH = "/home/fabrice/Documents/coding/ML/Results/sokocoin0/RRLearning/2023_07_25-14_37_e10000_lr0-1_SEst_blStar_g1-0_b0-05"


def visualize_matrix_px(save_path: str = None, path: str = None, table: np.array = None, show_in_explorer: bool = False):
    if table is None:
        table = np.load(path)

    fig = px.imshow(table, title=save_path)
    fig.write_image(save_path)
    if show_in_explorer:
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
    num_states = min(c_table.shape[0], num_states)
    # Get the indices of the states with the highest average c-values.
    top_rows = np.argpartition(avg_c_row, -num_states)[-num_states:]

    avg_c_col = c_table[top_rows, :].mean(axis=0)
    # Get the indices of the states with the highest average c-values.
    top_cols = np.argpartition(avg_c_col, -num_states)[-num_states:]
    
    # Extract the corresponding rows and columns.
    top_q_table = c_table[np.ix_(top_rows, top_cols)]

    visualize_matrix_px(table = top_q_table, save_path=path.replace('.npy', '.jpg'))

def visualize_topNq_states(q_table_path: str = None, q_table: np.array = None, num_states: int = 100, save_path: str = None, show_in_explorer:bool = False):
    if q_table is None:
        q_table = np.load(q_table_path, allow_pickle=True)

    if save_path is None:
        save_path = ''
        if q_table_path is not None:
            save_path = q_table_path.replace('.npy', '.jpg')
    
    # Compute the average Q-value for each state
    avg_q_values = q_table.mean(axis=1)
    num_states = min(q_table.shape[0], num_states)
    # Get the indices of the states with the highest average Q-values
    top_states = np.argpartition(avg_q_values, -num_states)[-num_states:]
    # Extract the corresponding rows from the Q-table
    top_q_table = q_table[top_states, :]

    visualize_matrix_px(table = top_q_table, save_path=save_path, show_in_explorer=show_in_explorer)

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
    qtable_files = [f for f in os.listdir(q_tables_dir) if os.path.isfile(os.path.join(q_tables_dir, f))]
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

def plot_states_to_png(states: List[np.array], env_name: str, file_path_prefix: str, info_text: str = '') -> List:
    frames = []        
    # Define the colormap for the states.
    colors = ["black", "silver", "gold", "peru", "green"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    v_min: int = min([state.min() for state in states])
    v_max: int = max([state.max() for state in states])

    # Iterate over all states-matrices and convert them to image.
    for nr, state in enumerate(states):
        fig, (ax_gif, ax_info) = plt.subplots(1, 2, figsize=(5, 3))
        fig.suptitle(f"RR-Learning Agent{env_name if env_name is not None else ''}")
        fig.tight_layout()
        ax_gif.matshow(state, cmap=cmap1, vmin=v_min, vmax=v_max)
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                value = state[i,j]
                ax_gif.text(j, i, str(value), va='center', ha='center')
        
        for frame in [ax_gif.axes, ax_info.axes]:
            frame.get_xaxis().set_ticks([])
            frame.get_yaxis().set_ticks([])        
        ax_gif.set_xlabel(f"State: {nr}/{len(states)-1}")

        border = 0.05
        ax_info.text(border, 1-border, info_text, va='top', ha='left')
        
        # Save the image.
        file_path = f"{file_path_prefix}_n{nr}.png"
        plt.savefig(file_path)
        frames.append(imageio.v2.imread(file_path))
        plt.close()

    return frames

def convert_images_to_gif(frames: List, save_path: str, duration: int = 600, loop: int = 3) -> None:
    imageio.mimsave(f"{save_path}.gif", 
        frames, 
        duration = duration, 
        loop = loop
    )

def parse_matrix_str_to_int(states: List[List[str]]) -> List[np.array]:
    state_matrices: List[np.array] = []
    for state in states:
        # Eliminate matrix encoding strings.
        state_rows: List[str] = state.replace('[','').replace(']','').replace(' ','').split("\n")
        # Remove the tailing float-dot and split the other values using them.
        state_matrix: List[List[str]] = [row[:-1].split(".") for row in state_rows]
        # Convert the strings into integers and save each state as np.array.
        state_matrix: np.array = np.array([[int(x) for x in row] for row in state_matrix])
        state_matrices.append(state_matrix)

    return state_matrices

def render_state_list_to_pngs(states: List[List[str]], env_name: str, file_path_prefix: str, info_text: str = '') -> None:
    # Convert the string representations of the states into matrices with integer values.
    state_matrices: List[np.array] = parse_matrix_str_to_int(states)
    # Now plot all these states and save the plots in a dir.
    plot_states_to_png(state_matrices, env_name, file_path_prefix, info_text)

def render_agent_journey_gif(directory: str, env_name: str = None, states: List[str] = None, info_text: str = '') -> str:
    # The real environment renderer uses 'saftey_ui' ('_display') and 'courses'. We can simplify this rendering process, since
    # no rendering in realtime, depending on user actions are required.
   
    # Convert the journey environments into images.    
    save_path: str          = f"{directory}/{c.fn_agent_journey}_figs"
    file_path_prefix: str   = f"{save_path}/{c.fn_agent_journey}"
    if not os.path.exists(save_path): os.mkdir(save_path)
    
    # Read all files in the directory. Find the file, which name starts with {c.fn_agent_journey}.    
    if states is None:
        agents_journey_file = None
        for filename in os.listdir(directory):
            if c.fn_agent_journey in filename and os.path.isfile(os.path.join(directory,filename)):
                agents_journey_file = filename
                break

        states = np.load(f"{directory}/{agents_journey_file}")
    
    # Convert the string representations of the states into matrices with integer values.
    state_matrices: List[np.array] = parse_matrix_str_to_int(states)
    # Now plot all these states and save the plots in a dir.
    gif_frames: List = plot_states_to_png(state_matrices, env_name, file_path_prefix, info_text)
    convert_images_to_gif(gif_frames, f"{save_path}/{c.fn_agent_journey}")
    
    return save_path

if __name__ == "__main__":
    C_PATH = "/home/fabrice/Downloads"
    # visualize_topNc_states(path=f"{C_PATH}/{c.fn_ctable_npy}")        
    # visualize_topNq_states(q_table_path=f"{C_PATH}/qtable_lr0-1_gamma1_beta0-1.npy")
    visualize_topNq_states(q_table_path=f"/home/fabrice/Downloads/qtable.npy")

    
    # path = f"{c.RESULTS_DIR}/env_name/method_name/dir_time_tag/qtable.npy"
    pass