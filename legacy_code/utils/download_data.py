import subprocess
from pathlib import Path
from huggingface_hub import login, hf_hub_download
login(token="token_key")

# Specify parameters
dataset_name = "TCGA"  # or "CPTAC", "PANDA"
project_name =  "TCGA-COAD" # or other projects
local_dir = r"C:\Users\Amaya\Documents\PhD\Data\TGCA_data\TCGA-COADREAD"  # where to save downloaded features

# Ensure local directory exists
Path(local_dir).mkdir(parents=True, exist_ok=True)

# Build the command as a list of arguments
command = [
    "huggingface-cli", "download",
    "MahmoodLab/UNI2-h-features",
    f"{dataset_name}/{project_name}.tar.gz",
    "--repo-type", "dataset",
    "--local-dir", local_dir
]

# Run the command
subprocess.run(command, check=True)
