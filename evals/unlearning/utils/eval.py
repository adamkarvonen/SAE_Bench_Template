import argparse
import os
import numpy as np
from transformer_lens import HookedTransformer
from sae.activation_store import ActivationsStore

from .feature_activation import get_top_features
from SAE_Bench_Template.evals.unlearning.utils.metrics import calculate_metrics_list
from SAEBench.utils import load_sae

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

def load_model_and_sae(sae_folder):
    sae = load_sae(sae_folder)
    model = HookedTransformer.from_pretrained('google/gemma-2-2b-it')
    return model, sae

def setup_activation_store(sae, model):
    sae.cfg.dataset = "Skylion007/openwebtext"
    sae.cfg.n_batches_in_store_buffer = 8
    return ActivationsStore(sae.cfg, model, create_dataloader=False)

def load_sparsity_data(sae_folder):
    forget_sparsity = np.loadtxt(os.path.join(RESULT_DIR, sae_folder, FORGET_FILENAME), dtype=float)
    retain_sparsity = np.loadtxt(os.path.join(RESULT_DIR, sae_folder, RETAIN_FILENAME), dtype=float)
    return forget_sparsity, retain_sparsity

def run_metrics_calculation(model, sae, activation_store, forget_sparsity, retain_sparsity, sae_folder):
    save_metrics_dir = os.path.join(RESULT_DIR, sae_folder)
    all_dataset_names = ['loss_added', 'wmdp-bio', 'high_school_us_history', 'college_computer_science', 
                         'high_school_geography', 'human_aging', 'college_biology']

    for retain_threshold in [0.01]:
        top_features_custom = get_top_features(forget_sparsity, retain_sparsity, retain_threshold=retain_threshold)

        main_ablate_params = {
            'intervention_method': 'clamp_feature_activation',
        }
        
        n_features_lst = [10, 20, 50]
        # n_features_lst = [10, 20, 50, 100, 500, 1000, 2000, 5000]
        multipliers = [0, 1, 5, 10, 50, 100]
        # multipliers = [0]
        
        sweep = {
            'features_to_ablate': [np.array(top_features_custom[:n]) for n in n_features_lst],
            'multiplier': multipliers,
        }

        calculate_metrics_list(
            model,
            sae,
            main_ablate_params,
            sweep,
            all_dataset_names,
            n_batch_loss_added=50,
            activation_store=activation_store,
            target_metric='correct',
            save_metrics=True,
            save_metrics_dir=save_metrics_dir,
            notes=f'_sparsity_thres{retain_threshold}'
        )

def main():
    args = parse_arguments()
    model, sae = load_model_and_sae(args.sae_folder)
    activation_store = setup_activation_store(sae, model)
    forget_sparsity, retain_sparsity = load_sparsity_data(args.sae_folder)
    run_metrics_calculation(model, sae, activation_store, forget_sparsity, retain_sparsity, args.sae_folder)

if __name__ == "__main__":
    main()