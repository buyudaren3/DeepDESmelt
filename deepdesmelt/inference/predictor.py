import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from deepdesmelt import MOE
from deepdesmelt.utils.visualization import calculate_metrics

def load_smiles_file(smiles_file):
    smiles_pairs = []
    with open(smiles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                smiles_pairs.append(line)
    return smiles_pairs


def inference(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_molar_ratios = []
    all_gates = []
    all_expert_outputs = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Extract molar ratio (last feature in input data)
            molar_ratio = data[:, -1].cpu().numpy()
            all_molar_ratios.append(molar_ratio)
            
            # Model forward pass
            output, aggregate_features, experts_raw_outputs, gate, _, _ = model(data, compute_load_balance_loss=False)
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_gates.append(gate.cpu().numpy())
            all_expert_outputs.append(experts_raw_outputs.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    molar_ratios = np.concatenate(all_molar_ratios, axis=0).flatten()
    gates = np.concatenate(all_gates, axis=0)
    expert_outputs = np.concatenate(all_expert_outputs, axis=0)
    return predictions, targets, molar_ratios, gates, expert_outputs


def main(args):
    # Check if files exist
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    
    # Load data
    print(f'\nLoading inference data from: {args.data_file}')
    inference_dataset = torch.load(args.data_file, weights_only=False)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    print(f'Number of samples: {len(inference_dataset)}')
    
    # Load SMILES - try to auto-detect if not provided
    smiles_pairs = None
    smiles_file_to_use = args.smiles_file
    
    # If smiles_file not provided, try to auto-detect based on data_file name
    if not smiles_file_to_use:
        # Try to find corresponding _smiles.txt file
        # e.g., if data_file is "inference_data.pt", look for "inference_data_smiles.txt"
        base_path = args.data_file.replace('.pt', '_smiles.txt')
        if os.path.exists(base_path):
            smiles_file_to_use = base_path
            print(f'\nAuto-detected SMILES file: {smiles_file_to_use}')
    
    if smiles_file_to_use and os.path.exists(smiles_file_to_use):
        print(f'\nLoading SMILES from: {smiles_file_to_use}')
        smiles_pairs = load_smiles_file(smiles_file_to_use)
        print(f'Loaded {len(smiles_pairs)} SMILES pairs')
    else:
        print('\nNote: No SMILES file found. Results will not include molecular structures.')
    
    # Initialize model
    print('\nInitializing model...')
    input_size = 1536  # UniMol representation size
    model = MOE(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=1,
        dropout=0,
        num_hidden_layers=args.num_hidden_layers,
        norm=args.norm,
        activation=args.activation,
        num_experts=args.num_experts,
        load_balance_alpha=args.load_balance_alpha,
        adaptive_weight=args.adaptive_weight.lower() == 'true',
        target_ratio=args.target_ratio
    )
    
    # Load model weights
    print(f'\nLoading model weights from: {args.model_path}')
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model = model.to(device)
    print('Model loaded successfully!')
    
    # Load normalization parameters if provided
    molar_ratio_min = None
    molar_ratio_max = None
    norm_file_to_use = args.norm_file
    if not norm_file_to_use:
        # Auto-detect data_range.npz in the same directory as data_file
        auto_norm_file = os.path.join(os.path.dirname(args.data_file), 'data_range.npz')
        if os.path.exists(auto_norm_file):
            norm_file_to_use = auto_norm_file
            print(f'\nAuto-detected normalization file: {norm_file_to_use}')
    if norm_file_to_use and os.path.exists(norm_file_to_use):
        print(f'\nLoading normalization parameters from: {norm_file_to_use}')
        norm_data = np.load(norm_file_to_use)
        min_val = norm_data['min_val']
        max_val = norm_data['max_val']
        # Molar ratio is the last feature (index 3072)
        molar_ratio_min = min_val[-1] if min_val.ndim == 1 else min_val
        molar_ratio_max = max_val[-1] if max_val.ndim == 1 else max_val
        print(f'  Molar ratio range: [{molar_ratio_min}, {molar_ratio_max}]')
    
    # Perform inference
    predictions, targets, molar_ratios, gates, expert_outputs = inference(model, inference_loader, device)
    
    # Denormalize molar ratios if normalization parameters are available
    if molar_ratio_min is not None and molar_ratio_max is not None:
        molar_ratios = molar_ratios * (molar_ratio_max - molar_ratio_min) + molar_ratio_min
    
    # Check if targets are valid (not all zeros or NaN)
    # Targets are considered invalid if they are all zeros or all NaN
    has_valid_targets = not (np.all(targets == 0) or np.all(np.isnan(targets)))
    
    # Calculate metrics if targets are available
    if has_valid_targets:
        print('\n' + '=' * 80)
        print('EVALUATION METRICS')
        print('=' * 80)
        
        # Filter out NaN and zero targets if any
        valid_mask = ~(np.isnan(targets) | (targets == 0))
        if np.sum(valid_mask) < len(targets):
            print(f'\nNote: {len(targets) - np.sum(valid_mask)} samples have invalid targets (NaN or 0) and will be excluded from metrics calculation')
        
        valid_targets = targets[valid_mask]
        valid_predictions = predictions[valid_mask]
        
        if len(valid_targets) > 0:
            metrics = calculate_metrics(valid_targets, valid_predictions)
            
            # Print metrics with custom precision
            print(f'R2: {metrics["R2"]:.3f}')
            print(f'MAE: {metrics["MAE"]:.1f}')
            print(f'RMSE: {metrics["RMSE"]:.1f}')
            print(f'AARD: {metrics["AARD"]:.1f}%')
        else:
            print('No valid targets found for evaluation')
            has_valid_targets = False
    else:
        print('\nNote: No valid target values detected, skipping evaluation metrics')
    
    # Prepare results DataFrame
    results_data = {
        'Prediction': predictions
    }
    
    # Add targets if valid
    if has_valid_targets:
        results_data['Target'] = targets
    
    # Add SMILES if available (split into HBA and HBD)
    if smiles_pairs is not None and len(smiles_pairs) == len(predictions):
        hba_smiles = []
        hbd_smiles = []
        for pair in smiles_pairs:
            parts = pair.split('\t') if '\t' in pair else pair.split(',')
            hba = parts[0].strip() if len(parts) > 0 else ''
            hbd = parts[1].strip() if len(parts) > 1 else ''
            # Remove quotes and parentheses
            hba = hba.strip("'\"()")
            hbd = hbd.strip("'\"()")
            hba_smiles.append(hba)
            hbd_smiles.append(hbd)
        results_data['HBA_SMILES'] = hba_smiles
        results_data['HBD_SMILES'] = hbd_smiles
    
    # Add molar ratio after HBD_SMILES
    results_data['Molar_Ratio'] = molar_ratios
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    results_df.to_csv(args.output_file, index=False)
    print(f'\nResults saved to: {args.output_file}')
    print('\n' + '=' * 80)
    print('INFERENCE COMPLETED SUCCESSFULLY!')
    print('=' * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Perform inference with trained DeepDESmelt model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file (best_model.pth)')
    parser.add_argument('--data_file', type=str, required=True, help='Path to inference dataset (.pt file)')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path for predictions (.csv)')
    
    # Optional arguments    
    parser.add_argument('--smiles_file', type=str, default=None, help='Path to SMILES file (optional)')
    parser.add_argument('--norm_file', type=str, default=None, help='Path to normalization parameters file (data_range.npz) for denormalizing molar ratio')
    
    # Model parameters (must match training configuration)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size (must match training)')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='Number of hidden layers (must match training)')
    parser.add_argument('--norm', type=str, default='LayerNorm', help='Normalization type (must match training)')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='Activation function (must match training)')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts (must match training)')
    parser.add_argument('--load_balance_alpha', type=float, default=100.0, help='Load balance alpha (must match training)')
    parser.add_argument('--adaptive_weight', type=str, choices=['true', 'false'], default='true', help='Adaptive weight (must match training)')
    parser.add_argument('--target_ratio', type=float, default=0.8, help='Target ratio (must match training)')
    
    args = parser.parse_args()
    
    main(args)
