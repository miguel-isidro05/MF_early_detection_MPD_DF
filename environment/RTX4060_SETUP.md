# RTX 4060 Environment

Target: Linux x86_64, Python 3.11, NVIDIA RTX 4060.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/capture_environment.py --output environment/system_info_rtx4060.txt
```

The CUDA wheel index is explicit, but the exact resolved versions must be frozen
on the RTX host after installation:

```bash
python -m pip freeze > environment/requirements-lock-rtx4060.txt
```

Do not accept a training run unless `torch.cuda.is_available()` is `True` and
the captured GPU name identifies the RTX 4060.
