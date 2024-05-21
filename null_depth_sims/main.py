import numpy as np


# Overall plan:
# define inputs from baldr as a distribution of zernikies
# apply arbitrary correction from PL loop
# assume first n LP modes are injected (for n=1,3)
# Correction due to kernel nuller chip
# look at overall null depth

tscope_type = "UT" # UT or AT


if tscope_type == "UT":
    diameter = 8.2
elif tscope_type == "AT":
    diameter = 1.8
else:
    raise ValueError("Invalid scope type")

# Inputs

import lanternfiber

lanternfiber.lanternfiber()
