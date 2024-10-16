import argparse
import os
import numpy as np
from transformer_lens import HookedTransformer
from evals.unlearning.utils.activation_store import ActivationsStore

from evals.unlearning.utils.feature_activation import get_top_features, load_sparsity_data, save_feature_sparsity
from evals.unlearning.utils.metrics import calculate_metrics_list



# def setup_activation_store(sae, model):
#     sae.cfg.dataset = "Skylion007/openwebtext"
#     sae.cfg.n_batches_in_store_buffer = 8
#     return ActivationsStore(sae.cfg, model, create_dataloader=False)

def run_metrics_calculation(model, sae, activation_store, forget_sparsity, retain_sparsity, sae_folder):
    all_dataset_names = ['wmdp-bio', 'high_school_us_history', 'college_computer_science', 
                         'high_school_geography', 'human_aging', 'college_biology']

    for retain_threshold in [0.01]:
        top_features_custom = get_top_features(forget_sparsity, retain_sparsity, retain_threshold=retain_threshold)

        main_ablate_params = {
            'intervention_method': 'clamp_feature_activation',
        }
        
        n_features_lst = [10, 20, 50]
        # n_features_lst = [10, 20, 50, 100, 500, 1000, 2000, 5000]
        multipliers = [50, 100]
        # multipliers = [0, 1, 5, 10, 50, 100]
        
        sweep = {
            'features_to_ablate': [np.array(top_features_custom[:n]) for n in n_features_lst],
            'multiplier': multipliers,
        }
        
        save_metrics_dir = os.path.join('results/metrics', sae_folder)
        
        metrics_lst = calculate_metrics_list(
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
            retain_threshold=retain_threshold,
        )
        
        return metrics_lst



def run_eval_single_sae(model, sae, sae_name):
    
    # calculate feature sparsity
    save_feature_sparsity(model, sae, sae_name)
    forget_sparsity, retain_sparsity = load_sparsity_data(sae_name)
    
    # do intervention and calculate eval metrics
    # activation_store = setup_activation_store(sae, model)
    activation_store = None
    results = run_metrics_calculation(model, sae, activation_store, forget_sparsity, retain_sparsity, sae_name)
    
    return results