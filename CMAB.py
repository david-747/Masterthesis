# checking python executable in case of runtime error
# import sys
# print("Python executable:", sys.executable)

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.optimize import linprog
from collections import defaultdict
import random

import MiscShipping

p1 = MiscShipping.Product("Hi")

print(p1)