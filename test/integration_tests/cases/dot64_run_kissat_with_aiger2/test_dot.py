from typing import List

import torch
import torch.nn as nn

from torch_mlir import fx

import os
import sys

class DotModule(torch.nn.Module):
  def mm(self, a, b):
    return torch.dot(a, b)
  
  def forward(self, a, b):
    return self.mm(a, b)
  
model = DotModule()

x = torch.randint(low=-128, high=128, size=(64,), dtype=torch.int16)
y = torch.randint(low=-128, high=128, size=(64,), dtype=torch.int16)

m = fx.export_and_import(model, x, y, output_type="linalg-on-tensors", func_name="mm")

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)

with open(os.path.join(current_script_dir, "mm.mlir"), "w") as f:
    f.write(str(m))
