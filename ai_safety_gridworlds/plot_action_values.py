from typing import List
import plotly.express as px
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join


if __name__ == "__main__":
    # TODO: GOAL: Bar Chart for differences between two qtables. Hope: Vanishing bar sizes.

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
