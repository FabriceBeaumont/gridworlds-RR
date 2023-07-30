from typing import List, Dict, Set, Tuple
import plotly.express as px
import numpy as np
from collections import Counter
import math

import re

if __name__ == "__main__":
    bl_str = "rest"
    baseline = None
    sub_title = "Bla"
    sub_title +=f', {bl_str}' if baseline is not None else ''
    print(sub_title)
