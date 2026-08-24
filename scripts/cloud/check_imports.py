import os
import sys

for p in [
    "/workspace/isaaclab/source/isaaclab",
    "/workspace/isaaclab/source/isaaclab_assets",
    "/workspace/isaaclab/source/isaaclab_tasks",
    "/workspace/isaaclab/source/isaaclab_rl",
]:
    sys.path.insert(0, p)

import torch

print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import isaaclab
import isaaclab_assets
import isaaclab_rl
import isaaclab_tasks

print("isaaclab", isaaclab.__file__)
import rsl_rl

print("rsl_rl", rsl_rl.__file__)
print("IMPORTS_OK")
