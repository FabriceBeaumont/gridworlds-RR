import plotly.express as px
import numpy as np
from collections import Counter
import math

import re

GAME_ART = [['######',  # Level 0.
            '# A###',
            '# X  #',
            '##   #',
            '### G#',
            '######'],
            ['#########',  # Level 2.
            '#       #',
            '#  1A   #',
            '# C# ####',
            '#### #C #',
            '#     2 #',    
            '#       #',
            '#########'],
            ['##########',  # Level 3.
            '#    #   #',
            '#  1 A   #',
            '# C#     #',
            '####     #',
            '# C#  ####',
            '#  #  #C #',
            '# 3    2 #',
            '#        #',
            '##########'],
]  

TEST_STATES = [['######',  # Level 0.
            '#  ###',
            '#   X#',
            '## A #',
            '### G#',
            '######'],
            ['#########',  # Level 2.
            '#     A #',
            '#       #',
            '# C# ####',
            '#### #  #',
            '#     2 #',    
            '#  1    #',
            '#########'],
            ['##########',  # Level 3.
            '#  2 #   #',
            '#  1     #',
            '#  #     #',
            '####     #',
            '# C#  ####',
            '#  #  #  #',
            '# 3      #',
            '#   A    #',
            '##########'],
]  

KEY_CHAR_GROUPS = [   
    ['A'],
    ['C'],
    ['X', '1', '2', '3']
]

KEY_CHARS = [   
    'A',
    'C',
    'B'
]

def state_to_keypos(state = GAME_ART[2]):

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

    print(key_char_pos)
    key_char_pos_str = f"a{key_char_pos[0]}-c{key_char_pos[1]}-b{key_char_pos[2]}"
    print(key_char_pos_str)

def state_to_keypos(key_char_pos = [[84], [52], [23, 13, 72]], env_id=2) -> str:
    # Prepare the gridworld - start from the basic and remove the objects.
    basic_state = GAME_ART[env_id]
    state_str = "".join(basic_state)

    for char_group in KEY_CHAR_GROUPS:
        for char in char_group:
            state_str = state_str.replace(char, " ")

    # Add the objects according to the key char positions.
    for i, key_char in enumerate(KEY_CHARS):
        for pos in key_char_pos[i]:
            state_str = state_str[:pos] + key_char + state_str[pos + 1:]

    # Finally split the environment accoring to its rows.
    row_length = len(basic_state[0])
    max_index = len(state_str)
    state_str_mat = [state_str[i:i + row_length] for i in range(0, max_index, row_length) ]
    
    print(state_str)
    print(key_char_pos)
    print()
    for row in state_str_mat:
        print(row)

    print()
    for row in basic_state:
        print(row)

# ctr0 = Counter({'#': 25, ' ': 8,  'A': 1, 'X': 1, 'G': 1})
# ctr1 = Counter({'#': 38, ' ': 29, 'C': 2, '1': 1, 'A': 1, '2': 1})
# ctr2 = Counter({'#': 47, ' ': 46, 'C': 3, '1': 1, 'A': 1, '3': 1, '2': 1})
# states0 =           60
# states1 =       29.830
# states2 = ? >  237.350

def my_comb(n, k) -> int:
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n-k)))

def print_state_set_size_estimations():
    """ To count the number of all possible states, we have to consider all possible places, where the boxes and the agent can be.
    This number has to be multiplied by a factor accounting for wheter some (but not all) coins have been collected or not.
    To formulate an upper bound on the number of possible states we allow the boxes and the agent to be places on all grids, which are not 
    a box or an agent. 
    We place them separately, since states are not different, if we swap the boxes. Notice however, that this differentiation is NOT accounted for
    in the encoding in the environment and in straight forward implementations based on string comparisons!!!!!!!!!

    Examples for states which we count, but are not possible are when the agent is surrounded by boxes and walls.
    The agent cannot pull boxes behind him and thus cannot create this state.
    """
    for env in GAME_ART:
        counter = Counter("".join(env))
        nr_whitespaces = counter[' ']
        nr_coins = counter['C'] + counter['G']
        nr_boxes = np.sum([counter[str(x)] for x in range(5)]) + counter['X']
        nr_agents = 1

        print(f"Nr whitespaces:\t{nr_whitespaces}\nNr coins:\t{nr_coins}\nNr boxes:\t{nr_boxes}")

        nr_states = 0

        for nr_collected_coins in range(nr_coins):
            free_coin_grids = nr_coins - nr_collected_coins
            
            # Agent and box configurations.            
            nr_box_placements   = my_comb(nr_whitespaces + free_coin_grids + nr_boxes,  nr_boxes)
            nr_agent_placements = my_comb(nr_whitespaces + free_coin_grids + nr_agents, nr_agents)
            
            # Compte the esimtate.
            nr_states += nr_box_placements * nr_agent_placements * my_comb(nr_coins, nr_collected_coins)

        print(nr_states)
        print()
        
if __name__ == "__main__":
    # print_state_set_size_estimations()
    # for state in TEST_STATES:
    #     state_to_int(state)
    state_to_keypos()

    pass