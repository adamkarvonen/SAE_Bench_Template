import sys
import os
import argparse
import numpy as np

# Append root directory to system path
sys.path.append('/root')

from .feature_activation import get_shuffled_forget_retain_tokens, get_feature_activation_sparsity
from unlearning.tool import get_hf_model
from SAEBench.utils import load_sae, download_sae_weights

# Constants
ROOT_DIR = '/root/unlearning/SAEBench'
RESULT_DIR = os.path.join(ROOT_DIR, 'results')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')

FORGET_FILENAME = 'feature_sparsity_forget.txt'
RETAIN_FILENAME = 'feature_sparsity_retain.txt'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run SAE analysis with specified folder')
    parser.add_argument('--sae_folder', type=str, required=True, help='Path to the SAE folder')
    return parser.parse_args()


def main():
    args = parse_arguments()

    if check_existing_results(args.sae_folder):
        print(f"Sparsity calculation for {args.sae_folder} is already done")
        return

    ensure_sae_weights(args.sae_folder)
    sae = load_sae(args.sae_folder)
    
    model = get_hf_model('google/gemma-2-2b-it')
    
    forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(model, batch_size=2048, seq_len=1024)
    
    feature_sparsity_forget, feature_sparsity_retain = calculate_sparsity(model, sae, forget_tokens, retain_tokens)
    
    save_results(args.sae_folder, feature_sparsity_forget, feature_sparsity_retain)

if __name__ == "__main__":
    main()