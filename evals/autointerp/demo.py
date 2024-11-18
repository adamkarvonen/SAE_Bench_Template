import argparse
from pathlib import Path

import torch

from evals.autointerp.config import AutoInterpEvalConfig
from evals.autointerp.main import run_eval

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Run AutoInterp evaluation.")
parser.add_argument("--api_key", type=str, required=True, help="API key for the evaluation.")
args = parser.parse_args()

api_key = args.api_key  # Use the API key supplied via command line

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

selected_saes = [("gpt2-small-res-jb", "blocks.7.hook_resid_pre")]
torch.set_grad_enabled(False)

# ! Demo 1: just 4 specially chosen latents
cfg = AutoInterpEvalConfig(model_name="gpt2-small", override_latents=[9, 11, 15, 16873])
save_logs_path = Path(__file__).parent / "logs_4.txt"
save_logs_path.unlink(missing_ok=True)
results = run_eval(cfg, selected_saes, str(device), api_key, save_logs_path=save_logs_path)
print(results)

# ! Demo 2: 100 randomly chosen latents
cfg = AutoInterpEvalConfig(model_name="gpt2-small", n_latents=100)
save_logs_path = Path(__file__).parent / "logs_100.txt"
save_logs_path.unlink(missing_ok=True)
results = run_eval(cfg, selected_saes, str(device), api_key, save_logs_path=save_logs_path)
print(results)

# python demo.py --api_key "YOUR_API_KEY"
