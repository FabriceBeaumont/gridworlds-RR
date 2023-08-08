from typing import List, Dict, Set, Tuple, Any
from scipy.sparse import save_npz, load_npz, csc_matrix, csr_matrix
import numpy as np
from math import ceil
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import imageio

import constants as c
import helper_fcts as hf

def visualize_qtable(env_name: str, q_table_path: str = None, q_table: np.array = None, save_path: str = None) -> None:
    if q_table is None:
        q_table = np.load(q_table_path, allow_pickle=True)

    if save_path is None:
        save_path = 'QTable.jpg'
        if q_table_path is not None:
            save_path = q_table_path.replace('.npy', '.jpg')

    dim_wzeros = q_table.shape[0]
    # Only consider non zero rows.
    q_table = q_table[np.any(q_table, axis=1)]
    dim_wozeros = q_table.shape[0]
    
    # Compute a suitable height for balanced plots.    
    h = max(50, q_table.shape[0]//50)
    # Fill up the matrix with dummy values to fit the new shape perfectly.
    w = ceil(q_table.shape[0] / h)
    # Add missing rows with a dummy value for more balanced plots.
    if w > 1:
        dummz_value = -10
        dummy_rows = dummz_value * np.ones((w*h - q_table.shape[0], len(c.ACTIONS)))
        q_table = np.vstack([q_table, dummy_rows])
        
    # Rearrange the possibly very long q-table, into a more square-shaped matrix.
    q_table_split = np.vsplit(q_table, w)
    q_table = np.concatenate(tuple(q_table_split), axis=1)

    # Construct the plot title.
    env_name_str: str = f" '{env_name}'" if env_name is not None else ''
    plot_title: str = f"Q-Table - Non-Zero State Evaluations - {env_name_str}"
    sub_title: str = "<br>"
    sub_title += "<sup>"
    sub_title += f"Original dimension: {dim_wzeros}. Nr. of non zero rows (which are displayed here): {dim_wozeros} ({round(dim_wozeros/dim_wzeros *100,2)} %)."
    sub_title += "<br>"
    sub_title += f"Order of actions: {', '.join(list(c.ACTIONS.values()))}"
    sub_title += "</sup>"   

    layout = go.Layout(
        title_text=plot_title + sub_title, 
        legend_title_text='Heatbar',
        title = '',
        yaxis = go.layout.YAxis(
            showticklabels=False,
            title = 'State Rows'
        )
    )
    
    fig = px.imshow(q_table, color_continuous_scale='RdBu_r', origin='lower')        
    fig.update_layout(layout)
    fig.write_image(save_path)

def save_images_as_gif(frames: List, save_path: str, duration: int = 800, loop: int = 5) -> None:
    imageio.mimsave(f"{save_path}.gif", 
        frames, 
        duration = duration, 
        loop = loop
    )

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
        env_name_str: str = f" '{env_name}'" if env_name is not None else ''
        plot_title: str = f"RR-Learning Agent{env_name_str}"
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
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        frames.append(imageio.v2.imread(buffer))

        if save_images_to_dir:
            file_path = f"{file_path_prefix}_n{nr}.png"
            plt.savefig(file_path)
        plt.close()

    return frames

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
    """Load a q-table and execute an agent based on it. Render the visited states as a gif.
    Also diplay a bunch of information side by side to the render.
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
    save_images_as_gif(gif_frames, f"{save_path}/{c.fn_agent_journey}", duration=1000)    
    
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
    filenname_qtable_plot: str          = f"{save_path}/{c.fn_plot5_qtable_heatmap}"

    # Prepare a subtitle containing the parameter settings.
    metho_str = f"{c.PARAMETRS.METHOD_NAME.value}: {method_name}"
    lr_str    = f"{c.PARAMETRS.LEARNING_RATE.value}: {learning_rate}"
    sss_str   = f"{c.PARAMETRS.STATE_SPACE_STRATEGY.value}: '{strategy}'"
    bl_str    = f"{c.PARAMETRS.BASELINE.value}: '{baseline}'"
    dic_str   = f"{c.PARAMETRS.Q_DISCOUNT.value}: {q_discount}"
    beta_str  = f"{c.PARAMETRS.BETA.value}: {beta}"
    epis_str  = f"{c.PARAMETRS.NR_EPISODES.value}: {nr_episodes}"
    steps_str = f"{c.PARAMETRS.MAX_NR_STEPS.value}: {max_nr_steps}"
    lperf_str = f"Last episodes performance: {results_df[c.results_col_performances].iloc[-1]}"
    lref_str  = f"Last episodes reward: {results_df[c.results_col_rewards].iloc[-1]}"
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

    # Plot the q table.
    visualize_qtable(env_name=env_name, q_table=q_table, save_path=filenname_qtable_plot)
    
    print("\rCreating a gif ...", end="")
    # Render the learned q-table strategy to gif.
    text = f"{lr_str}\n{sss_str}\n{bl_str}\n{dic_str}\n{beta_str}\n{epis_str}\n{steps_str}\n\n{lperf_str}\n{lref_str}"
    render_agent_journey_gif(data_path=data_path, save_path=save_path, env_name=env_name, q_table=q_table, states_dict=states_dict, info_text=text)

    print("\rEvaluation complete.")

if __name__ == "__main__":
    # C_PATH = "/home/fabrice/Documents/coding/ML/GridworldResults/sokocoin2/RRLearning/2023_07_30-16_01_e5000_lr0-2_SExp_blStar_g0-9_b0-2"
    C_PATH = "/home/fabrice/Documents/coding/ML/GridworldResults/sokocoin0/RRLearning/2023_08_08-03_21_e100_lr0-1_SEst_blStar_g0-99_b0-1"
    evaluate(C_PATH)
    pass