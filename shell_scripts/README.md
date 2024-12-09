I recommend running these from the root directory of the repo. Using these, you can run the full set of evaluations (or a subset) on a selection of SAE Lens SAEs.

We currently have 3 provided shell scripts:

- `run.sh`: Runs all SAE Bench evals on all Gemma-Scope 16k width Gemma-2-2B SAEs on layers 5, 12, and 19. Batch sizes are set for a 24GB VRAM GPU.
- `run_reduced_memory.sh`: Batch sizes and evaluation flags have been modified to lower memory usage, especially for wider SAEs. Runs all SAE Bench evals on all Gemma-Scope 65k width Gemma-2-2B SAEs on layers 5, 12, and 19 on a GPU with 24GB of VRAM.
- `run_reduced_memory_1m_width.py`: For 1M width SAEs, we have to be careful about memory allocation and the order in which we load SAEs and models to avoid memory issues, especially on GPUs with less memory. To avoid introducing extra complexity into the main evals, we instead just evaluate 1 SAE at a time. For a single eval, we evaluate all SAEs on a given layer and then clean up associated artifacts at the end. This script will successfully run on a GPU with 48 GB of VRAM.