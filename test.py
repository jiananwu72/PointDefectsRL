# To see whether the environment is successfully set up

try:
	import torch
	print("Pytorch version:", torch.__version__)
except Exception:
	print("PyTorch is not installed or failed to import.")