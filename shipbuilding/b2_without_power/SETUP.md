# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# RTX PRO 6000 Blackwell (sm_120) with a CUDA 13-compatible driver:
pip3 install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

# Clone this repo and install the SO-101 copy in so101/ (the repo root is the unmodified upstream code)
git clone https://github.com/YoungK82/OpenVLA-oft.git
cd OpenVLA-oft/so101
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

Run training, serving and the scripts below from inside `so101/`. `pip install -e .` there also installs the
`moojink/transformers-openvla-oft` fork, whose sdpa attention is bidirectional (the paper's parallel decoding); stock
transformers would silently train a causal model. Check both with:

```bash
python -c "import prismatic, importlib.metadata as m; print(prismatic.__file__); print(m.distribution('transformers').read_text('direct_url.json'))"
```

The first line must end in `so101/prismatic/__init__.py` and the second must name `transformers-openvla-oft`.

`flash-attn==2.5.5` does not contain Blackwell `sm_120` kernels. On RTX PRO 6000 Blackwell, skip that installation
and use the Transformers SDPA attention path unless a newer Blackwell-compatible FlashAttention build has been verified.

## SO-101

SO-101 robot control uses a separate LeRobot client environment so hardware dependencies do not complicate the GPU
training environment. Follow [`SO101.md`](SO101.md) for recording, RLDS conversion, fine-tuning, and safe dry-run
evaluation instructions.
