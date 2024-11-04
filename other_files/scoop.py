from scoop_env import IsaacSim
from UserDefinedSettings import UserDefinedSettings


import numpy as np


def test():
    userDefinedSettings = UserDefinedSettings()
    weighingEnvironment = IsaacSim(userDefinedSettings=userDefinedSettings)
    weighingEnvironment.data_collection()


if __name__ == "__main__":
    test()
