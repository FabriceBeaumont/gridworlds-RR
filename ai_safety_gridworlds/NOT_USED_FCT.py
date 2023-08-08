from typing import List, Dict, Set, Tuple
import helper_fcts as hf
import re
import numpy as np

OLD_KEY_CHAR_GROUPS = [   
    ['A'],
    ['C'],
    ['X', '1', '2', '3']
]

KEY_CHARS = [   
    'A',
    'C',
    'B'
]

KEY_CHAR_GROUPS = [   
    ['2'],
    ['3'],
    ['4']
]

def state_to_keypos(state = hf.GAME_ART[2]) -> Tuple[str, List[List[int]]]:

    state_str = "".join(state)

    key_char_pos = []

    for char_group in KEY_CHAR_GROUPS:
        indices = []
        for pattern in char_group:
            # Get an iterator to find all occurances.
            iterator = re.finditer(pattern=pattern, string=state_str)
            pattern_indices = [match.start() for match in iterator]
            # Merge all indices of occurances of this pattern, with others from the char group.
            # This ensures, that for example no difference is made between the bosex '1', '2', and '3'.
            indices += pattern_indices
            
        key_char_pos.append(indices)

    return f"a{key_char_pos[0]}-c{key_char_pos[1]}-b{key_char_pos[2]}", key_char_pos

def keypos_to_state(key_char_str: str, env_id=2) -> Tuple[List[str], List[List[int]]]:
    # Parse the key-char-string into a list.
    tmp_pos: List[str] = [x[1:].replace("[", "").replace("]", "") for x in key_char_str.split('-')]
    key_char_pos: List[List[int]] = [[int(x) for x in pos.split(', ')] for pos in tmp_pos]
        
    # Prepare the gridworld - start from the basic.
    basic_state = hf.GAME_ART[env_id]
    state_str = "".join(basic_state)

    # Remove the already present objects.
    for char_group in KEY_CHAR_GROUPS:
        for char in char_group:
            state_str = state_str.replace(char, " ")

    # Add the objects according to the key char positions.
    for i, key_char in enumerate(KEY_CHARS):
        for pos in key_char_pos[i]:
            state_str = state_str[:pos] + key_char + state_str[pos + 1:]

    # Finally split the environment accoring to its rows.
    row_length  = len(basic_state[0])
    max_index   = len(state_str)
    state_str_mat: List[str] = [state_str[i:i + row_length] for i in range(0, max_index, row_length) ]

    return state_str_mat, key_char_pos

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

def demo_keypos_fct():
    state: List[str] = ['##########',  # Level 3.
           '#    #   #',
           '#     A  #',
           '#  # 123 #',
           '####     #',
           '# C#  ####',
           '#  #  #  #',
           '#        #',
           '#        #',
           '##########']
    
    key_char_pos_str, key_char_pos = state_to_keypos(state=state)    
    print(key_char_pos)
    print(key_char_pos_str) 

    state_str_mat, key_char_pos = keypos_to_state(key_char_pos_str)
    for row in state_str_mat:
        print(row)

if __name__=="__main__":
    pass