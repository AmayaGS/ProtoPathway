
## Setup

First clone the repository to the desired location and enter the directory:

```bash
# clone project to desired location
git clone https://github.com/AmayaGS/ProtoPathway
cd ProtoPathway
```

Then create a virtual environememt and install the requirements.txt

#### General Requirements
- Python 3.11.7
- PyTorch 2.5
- NVIDIA GPU with CUDA 12.4

```bash
# Virtual Environment
python -m venv bioxcpath
source protopath/bin/activate

# PyTorch with cuda capabilities
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt  

```

## Usage

### Data Preprocessing