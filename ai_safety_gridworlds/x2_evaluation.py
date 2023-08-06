from typing import List, Dict, Set, Tuple, Any
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.sparse import save_npz, load_npz, csc_matrix, csr_matrix
from io import BytesIO
import pandas as pd
import os

import imageio

import constants as c
import helper_fcts as hf

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

def plot_states_to_image(states: List[np.array], returns: List[float] = None, performances: List[float] = None, env_name: str = None, save_images_to_dir: bool = False, file_path_prefix: str = 'Image', info_text: str = '') -> List:
    frames = []        
    # Define the colormap for the states.
    colors = ["black", "silver", "gold", "peru", "green"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    v_min: int = min([state.min() for state in states])
    v_max: int = max([state.max() for state in states])

    # Iterate over all states-matrices and convert them to image.
    for nr, state in enumerate(states):
        fig, (ax_state, ax_info) = plt.subplots(1, 2, figsize=(5, 3))
        agent_name: str = f" '{env_name}'" if env_name is not None else ''
        plot_title: str = f"RR-Learning Agent{agent_name}"
        fig.suptitle(plot_title)
        ax_state.matshow(state, cmap=cmap1, vmin=v_min, vmax=v_max)
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                value = state[i,j]
                ax_state.text(j, i, str(value), va='center', ha='center')
        
        for frame in [ax_state.axes, ax_info.axes]:
            frame.get_xaxis().set_ticks([])
            frame.get_yaxis().set_ticks([])        
        ax_state.set_xlabel(f"State: {nr}/{len(states)-1}")

        text = f"{info_text}\n\nCurrent return: {returns[nr]}\nCurrent hidden reward: {performances[nr]}"
        border = 0.05
        ax_info.text(border, 1-border, text, va='top', ha='left')
        # Remove the black border of the text-subplot.
        ax_info.spines['top'].set_visible(False)
        ax_info.spines['right'].set_visible(False)
        ax_info.spines['bottom'].set_visible(False)
        ax_info.spines['left'].set_visible(False)

        # Terminate the plot creation. Set layout to tight.
        plt.tight_layout()
        # Save the image.
        #frames.append(imageio.v2.imread(file_path)) # plt.figure
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        frames.append(imageio.v2.imread(buffer))

        if save_images_to_dir:
            file_path = f"{file_path_prefix}_n{nr}.png"
            plt.savefig(file_path)
        plt.close()

    return frames

def save_images_as_gif(frames: List, save_path: str, duration: int = 600, loop: int = 3) -> None:
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

def render_agent_journey_gif(data_path: str, save_path: str, env_name: str, q_table: np.array=None, states_dict: Dict[str, int]=None, states_list: List[str] = None, info_text: str = '', save_images_to_dir: bool = False) -> str:
    """_summary_

    Args:
        directory (str): _description_
        env_name (str, optional): _description_. Defaults to None.
        states (List[str], optional): _description_. Defaults to None.
        info_text (str, optional): _description_. Defaults to ''.

    Returns:
        str: _description_
    """
    # The real environment renderer uses 'saftey_ui' ('_display') and 'courses'. We can simplify this rendering process, since
    # no rendering in realtime, depending on user actions are required.
   
    # Convert the journey environments into images.
    file_path_prefix: str   = ''
    agent_journey_figs: str = f"{save_path}/{c.fn_agent_journey}_figs"
    if save_images_to_dir:
        file_path_prefix: str   = f"{agent_journey_figs}/{c.fn_agent_journey}"
        if not os.path.exists(agent_journey_figs): os.mkdir(agent_journey_figs)
    
    # Read all files in the directory. Find the file, which name starts with {c.fn_agent_journey}.
    states_list, returns, performances = hf.run_agent_on_env(results_path=data_path, env_name=env_name, q_table=q_table, states_dict=states_dict)
    
    # Convert the string representations of the states into matrices with integer values.
    state_matrices: List[np.array] = parse_matrix_str_to_int(states_list)
    # Now plot all these states and save the plots in a dir.
    gif_frames: List = plot_states_to_image(states=state_matrices, returns=returns, performances=performances, env_name=env_name, save_images_to_dir=save_images_to_dir, file_path_prefix=file_path_prefix, info_text=info_text)
    save_images_as_gif(gif_frames, f"{save_path}/{c.fn_agent_journey}")    
    
    return agent_journey_figs

def evaluate(data_path: str):
    if data_path[-1] == "/": data_path = data_path[:-1]
    save_path: str = f"{data_path}/{c.dir_evaluation}"
    if not os.path.exists(save_path): os.mkdir(save_path)
    # Load data from directory.
    settings = dict()
    filenname_qtable: str           = f"{data_path}/{c.fn_qtable_npy}"
    filenname_states_dict: str      = f"{data_path}/{c.fn_states_dict_npy}"
    filenname_coverage_table: str   = f"{data_path}/{c.fn_ctable_npy}"
    filenname_general: str          = f"{data_path}/{c.fn_general_csv}"
    filenname_perf: str             = f"{data_path}/{c.fn_performances_csv}"

    print("Loading files ...", end="")
    settings: Dict[str, str] = pd.read_csv(filenname_general).iloc[0].to_dict()
    q_table         = np.load(filenname_qtable, allow_pickle=True)
    states_dict     = np.load(filenname_states_dict, allow_pickle=True).item()
    coverage_table  = np.load(filenname_coverage_table, allow_pickle=True)
    results_df      = pd.read_csv(filenname_perf)
                    
    method_name: str        = settings.get(c.PARAMETRS.METHOD_NAME.value)
    env_name: str           = settings.get(c.PARAMETRS.ENV_NAME.value)
    nr_episodes: int        = settings.get(c.PARAMETRS.NR_EPISODES.value)
    max_nr_steps: int       = settings.get(c.PARAMETRS.MAX_NR_STEPS.value)
   
    learning_rate: float    = settings.get(c.PARAMETRS.LEARNING_RATE.value)
    strategy: str           = settings.get(c.PARAMETRS.STATE_SPACE_STRATEGY.value)
    baseline: str           = settings.get(c.PARAMETRS.BASELINE.value)
    q_discount: float       = settings.get(c.PARAMETRS.Q_DISCOUNT.value)
    beta: float             = settings.get(c.PARAMETRS.BETA.value)

    # Set up plot names.
    filenname_perf_plot: str            = f"{save_path}/{c.fn_plot1_performance_jpeg}"
    filenname_results_plot: str         = f"{save_path}/{c.fn_plot2_results_jpeg}"
    filenname_smooth_results_plot: str  = f"{save_path}/{c.fn_plot3_results_smooth_jpeg}"
    filenname_tde_plot: str             = f"{save_path}/{c.fn_plot4_tde_jpeg}"
    
    # Prepare a subtitle containing the parameter settings.
    metho_str = f"{c.PARAMETRS.METHOD_NAME.value}: {method_name}"
    lr_str    = f"{c.PARAMETRS.LEARNING_RATE.value}: {learning_rate}"
    sss_str   = f"{c.PARAMETRS.STATE_SPACE_STRATEGY.value}: '{strategy}'"
    bl_str    = f"{c.PARAMETRS.BASELINE.value}: '{baseline}'"
    dic_str   = f"{c.PARAMETRS.Q_DISCOUNT.value}: {q_discount}"
    beta_str  = f"{c.PARAMETRS.BETA.value}: {beta}"
    epis_str  = f"{c.PARAMETRS.NR_EPISODES.value}: {nr_episodes}"
    steps_str = f"{c.PARAMETRS.MAX_NR_STEPS.value}: {max_nr_steps}"
    lperf_str = f"Last performance: {results_df[c.results_col_performances].iloc[-1]}"
    lref_str  = f"Last reward: {results_df[c.results_col_rewards].iloc[-1]}"
    sub_title: str = f"<br><sup>"
    sub_title += f"{metho_str}"
    sub_title += f", {lr_str}"
    sub_title += f", {sss_str}"
    sub_title +=f', {bl_str}' if baseline is not None else ''
    sub_title += f", {dic_str}"
    sub_title += "<br>"
    sub_title +=f'{beta_str}' if beta is not None else ''    
    sub_title += f", {epis_str}"
    sub_title += f", {steps_str}"    
    sub_title += f", {lperf_str}"
    sub_title += f", {lref_str}"
    sub_title += f"</sup>"

    print("\rCreating plots ...", end="")
    # Plot the raw performance data and store it to image.
    title: str = f"Performance & Reward - '{env_name}'\n{sub_title}"
    legend_settings = dict(title="Legend")
    fig = px.line(results_df, x=c.results_col_episodes, y=[c.results_col_rewards, c.results_col_performances], title=title)
    fig.update_layout(legend=legend_settings)
    fig.write_image(filenname_perf_plot)
        
    # Plot the raw squared temporal difference error and store it to image.
    title: str = f"Squared Temporal Difference Error - '{env_name}'\n{sub_title}"
    fig = px.line(results_df, x=c.results_col_episodes, y=[c.results_col_tdes], title=title)
    fig.update_layout(legend=legend_settings)
    fig.write_image(filenname_tde_plot)
    
    # Standardize the data and plot it.
    cols_to_standardize = [c.results_col_rewards, c.results_col_performances, c.results_col_tdes, c.results_col_rewards + "_smooth", c.results_col_performances + "_smooth", c.results_col_tdes + "_smooth"]
    results_df[cols_to_standardize] = results_df[cols_to_standardize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Plot the standardized performance data and store it to image.
    title: str = f"Standardized Evaluations - '{env_name}'\n{sub_title}"
    fig = px.line(results_df, x=c.results_col_episodes, y=[c.results_col_rewards, c.results_col_performances, c.results_col_tdes], title=title)
    fig.update_layout(legend=legend_settings)
    fig.write_image(filenname_results_plot)

    # Plot the standardized smoothed performance data and store it to image.
    title: str = f"Smoothed Evaluations - '{env_name}'\n{sub_title}"
    fig = px.line(results_df, x=c.results_col_episodes, y=[c.results_col_rewards + "_smooth", c.results_col_performances + "_smooth", c.results_col_tdes + "_smooth"], title=title)
    fig.update_layout(legend=legend_settings)
    fig.write_image(filenname_smooth_results_plot)

    print("\rCreating a gif ...", end="")
    # Render the learned q-table strategy to gif.
    text = f"{lr_str}\n{sss_str}\n{bl_str}\n{dic_str}\n{beta_str}\n{epis_str}\n{steps_str}\n\n{lperf_str}\n{lref_str}"
    render_agent_journey_gif(data_path=data_path, save_path=save_path, env_name=env_name, q_table=q_table, states_dict=states_dict, info_text=text)

    print("\rEvaluation complete.")

if __name__ == "__main__":
    C_PATH = "/home/fabrice/Documents/coding/ML/GridworldResults/sokocoin2/RRLearning/2023_07_30-16_01_e5000_lr0-2_SExp_blStar_g0-9_b0-2"
    # visualize_topNc_states(path=f"{C_PATH}/{c.fn_ctable_npy}")        
    # visualize_topNq_states(q_table_path=f"{C_PATH}/qtable_lr0-1_gamma1_beta0-1.npy")
    # visualize_topNq_states(q_table_path=f"/home/fabrice/Downloads/qtable.npy")
    evaluate(C_PATH)
    
    # path = f"{c.RESULTS_DIR}/env_name/method_name/dir_time_tag/qtable.npy"
    pass