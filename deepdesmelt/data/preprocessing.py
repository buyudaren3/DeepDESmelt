import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import TensorDataset
from unimol_tools import UniMolRepr
from deepdesmelt.utils import _dataset_name, normalize, mix_out


def _load_smiles_pairs(smiles_file):
    pairs = []
    if not os.path.exists(smiles_file):
        return pairs
    with open(smiles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(ast.literal_eval(line))
    return pairs

def _save_train_data_full(HBA_smiles_list, HBD_smiles_list, HBA_molar_fraction,
                          HBA_repr, HBD_repr, save_path, 
                          normalized_repr=None, min_val=None, max_val=None):
    save_dict = {
        'HBA_smiles': np.array(HBA_smiles_list, dtype=object),
        'HBD_smiles': np.array(HBD_smiles_list, dtype=object),
        'HBA_molar_fraction': HBA_molar_fraction,
        'HBA_repr': HBA_repr,
        'HBD_repr': HBD_repr
    }
    
    # Also save normalized repr if provided (for exact matching in inference)
    if normalized_repr is not None:
        save_dict['normalized_repr'] = normalized_repr
    if min_val is not None:
        save_dict['min_val'] = min_val
    if max_val is not None:
        save_dict['max_val'] = max_val
        
    np.savez(save_path + 'train_data_full.npz', **save_dict)
    print(f"Saved full train data: {len(HBA_smiles_list)} entries")


def _load_train_data_full(load_path):
    file_path = load_path + 'train_data_full.npz'
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    data = np.load(file_path, allow_pickle=True)
    result = {
        'HBA_smiles': data['HBA_smiles'].tolist(),
        'HBD_smiles': data['HBD_smiles'].tolist(),
        'HBA_molar_fraction': data['HBA_molar_fraction'],
        'HBA_repr': data['HBA_repr'],
        'HBD_repr': data['HBD_repr']
    }
    
    # Load normalized repr if available (for exact matching)
    if 'normalized_repr' in data:
        result['normalized_repr'] = data['normalized_repr']
        print("  - Found pre-computed normalized repr")
    if 'min_val' in data:
        result['min_val'] = data['min_val']
    if 'max_val' in data:
        result['max_val'] = data['max_val']
        
    print(f"Loaded full train data: {len(result['HBA_smiles'])} entries")
    return result

def _build_sample_index(HBA_smiles_list, HBD_smiles_list, HBA_molar_fraction):
    index_map = {}
    for i in range(len(HBA_smiles_list)):
        key = (HBA_smiles_list[i], HBD_smiles_list[i], float(HBA_molar_fraction[i, 0]))
        index_map[key] = i
    return index_map


def _UniMol_repr(csv_file):
    df = pd.read_csv(csv_file)
    _HBA_smiles_list = df['Smiles#1'].tolist()
    _HBD_smiles_list = df['Smiles#2'].tolist()
    clf = UniMolRepr(data_type='molecule', model_name='unimolv2', model_size='1.1B')
    HBA_repr = np.array(clf.get_repr(_HBA_smiles_list)['cls_repr'])
    HBD_repr = np.array(clf.get_repr(_HBD_smiles_list)['cls_repr'])
    
    _HBA_molar_fraction = df['X#1 (molar fraction)'].to_numpy().reshape(-1, 1)
    _target_array = df[['Tmelt, K']].to_numpy()

    assert len(_HBA_smiles_list) == len(_HBD_smiles_list) == len(HBA_repr) == len(HBD_repr) == len(_HBA_molar_fraction) == len(_target_array), "length not match"
    return _target_array, HBA_repr, HBD_repr, _HBA_molar_fraction, _HBA_smiles_list, _HBD_smiles_list


def _UniMol_repr_single(smiles):
    clf = UniMolRepr(data_type='molecule', model_name='unimolv2', model_size='1.1B')
    repr_result = clf.get_repr([smiles])['cls_repr']
    return np.array(repr_result[0])

def save_txt(save_path, name, data):
    with open(save_path + f"{name}.txt", 'w') as f:
        for item in data:
            f.write(str(item) + '\n')

def make_torch_dataset(x: np.ndarray, y: np.ndarray, smiles: list, save_path: str, name: str):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    # Create and save the dataset
    dataset = TensorDataset(x, y)
    torch.save(dataset, save_path + name + ".pt")
    file_name = save_path + name + ".pt"

    save_txt(save_path, name + '_smiles', smiles)

    print(f'save files: {file_name}')

def data_cleaning(repr, target, HBA_smiles, HBD_smiles):
    assert len(repr) == len(target) == len(HBA_smiles) == len(HBD_smiles), "length not match"

    # Find indices where target values are not NaN
    valid_indices = np.where(~np.isnan(target))[0]  # ~np.isnan(target) returns indices of non-NaN values

    # Filter out non-empty repr, target, HBA_smiles and HBD_smiles based on these indices
    cleaned_repr = repr[valid_indices]
    cleaned_target = target[valid_indices]
    cleaned_HBA_smiles = [HBA_smiles[i] for i in valid_indices]
    cleaned_HBD_smiles = [HBD_smiles[i] for i in valid_indices]

    if np.isnan(cleaned_repr).any():
        print("Warning: NaN values exist in the cleaned data!")
    if np.isinf(cleaned_repr).any():
        print("Warning: Infinite values exist in the cleaned data!")

    print(f'Data cleaning: {len(target)} -> {len(cleaned_target)} samples')
    return cleaned_repr, cleaned_target, cleaned_HBA_smiles, cleaned_HBD_smiles


def generate_unique_combination_index(HBA_smiles_list, HBD_smiles_list):
    # Get unique HBA and HBD combinations
    smiles_combinations = list(zip(HBA_smiles_list, HBD_smiles_list))

    # Create a DataFrame of unique combinations and assign indices
    df_unique = pd.DataFrame(smiles_combinations, columns=['HBA', 'HBD'])
    df_unique = df_unique.drop_duplicates().reset_index(drop=True)
    df_unique = df_unique.reset_index()  # Add index column
    df_unique = df_unique.rename(columns={'index': 'unique_index'})  # Rename index column to unique_index

    # Create a dictionary mapping combinations to unique indices
    combination_to_index = {
        (row['HBA'], row['HBD']): row['unique_index']
        for _, row in df_unique.iterrows()
    }

    # Generate unique indices for each (HBA, HBD) pair using the mapping
    indices = [combination_to_index[comb] for comb in smiles_combinations]
    unique_indices = np.array(indices).reshape(-1, 1)
    return unique_indices

def make_training_dataset(config):
    save_path = config['save_path']
    seed = config['seed']
    data_file = config.get('data_file', '../dataset/DES_melting_point_dataset.csv')

    dataset_path = _dataset_name(save_path)
    target, HBA_repr, HBD_repr, HBA_molar_fraction, HBA_smiles_list, HBD_smiles_list = _UniMol_repr(data_file)
    target = target.flatten()

    repr = np.concatenate((HBA_repr, HBD_repr, HBA_molar_fraction), axis=1)

    # Data split using strict mode (group by unique HBA-HBD combinations)
    train_val_test_index = generate_unique_combination_index(HBA_smiles_list, HBD_smiles_list)
    mix_train_val_test = mix_out(repr, target, train_val_test_index, 1, 0.1, seed=seed)
    print('Splitting into train, validation and test')
    for train_val_idx, test_idx in mix_train_val_test:
        train_val_repr = repr[train_val_idx]
        train_val_target = target[train_val_idx]
        test_repr = repr[test_idx]
        test_target = target[test_idx]
        train_val_index = train_val_test_index[train_val_idx]

        # Convert indices to lists
        train_val_smiles_HBA = [HBA_smiles_list[i] for i in train_val_idx]
        test_smiles_HBA = [HBA_smiles_list[i] for i in test_idx]
        train_val_smiles_HBD = [HBD_smiles_list[i] for i in train_val_idx]
        test_smiles_HBD = [HBD_smiles_list[i] for i in test_idx]

        # Merge two lists using zip and convert to list of lists
        test_smiles = [(hba, hbd) for hba, hbd in zip(test_smiles_HBA, test_smiles_HBD)]

    mix_train_val = mix_out(train_val_repr, train_val_target, train_val_index, 1, 0.1, seed=seed)
    for train_idx, val_idx in mix_train_val:
        train_repr = train_val_repr[train_idx]
        train_target = train_val_target[train_idx]
        val_repr = train_val_repr[val_idx]
        val_target = train_val_target[val_idx]

        train_smiles_HBA = [train_val_smiles_HBA[i] for i in train_idx]
        val_smiles_HBA = [train_val_smiles_HBA[i] for i in val_idx]
        train_smiles_HBD = [train_val_smiles_HBD[i] for i in train_idx]
        val_smiles_HBD = [train_val_smiles_HBD[i] for i in val_idx]

        train_smiles = [(hba, hbd) for hba, hbd in zip(train_smiles_HBA, train_smiles_HBD)]
        val_smiles = [(hba, hbd) for hba, hbd in zip(val_smiles_HBA, val_smiles_HBD)]

    train_repr, min_val, max_val = normalize(train_repr)
    val_repr, _, _ = normalize(val_repr, min_val, max_val)
    test_repr, _, _ = normalize(test_repr, min_val, max_val)
    
    # Normalize the full repr array using the same min/max from train subset
    # This ensures exact matching in inference mode
    full_repr_normalized, _, _ = normalize(repr, min_val, max_val)
    
    # Save full training data with normalized repr for inference mode
    _save_train_data_full(HBA_smiles_list, HBD_smiles_list, HBA_molar_fraction,
                          HBA_repr, HBD_repr, dataset_path,
                          normalized_repr=full_repr_normalized,
                          min_val=min_val, max_val=max_val)

    if np.isnan(val_repr).any():
        print("-----NaN values exist in the validation set!")
    if np.isinf(val_repr).any():
        print("-----Infinite values exist in the validation set!")

    # Save min_val and max_val to binary file (npz format for full precision)
    np.savez(dataset_path + 'data_range.npz', min_val=min_val, max_val=max_val)
    print(f"Normalization parameters saved to: {dataset_path}data_range.npz")
    
    # Also save to text file for human readability (optional)
    min_val_str = np.array2string(min_val, threshold=np.inf)
    max_val_str = np.array2string(max_val, threshold=np.inf)
    with open(dataset_path + 'data_range.txt', 'w') as f:
        f.write(f"min_val: {min_val_str}\n")
        f.write(f"max_val: {max_val_str}\n")
    print(f"Normalization parameters (text, for reference) saved to: {dataset_path}data_range.txt")

    make_torch_dataset(train_repr, train_target, train_smiles, dataset_path, 'train')
    make_torch_dataset(val_repr, val_target, val_smiles, dataset_path, 'val')
    make_torch_dataset(test_repr, test_target, test_smiles, dataset_path, 'test')


    # Output data to readme
    with open(dataset_path + 'readme.txt', 'w') as f:
        f.write('dataset info:\n')
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write(f'train size: {len(train_repr)}\n')
        f.write(f'val size: {len(val_repr)}\n')
        f.write(f'test size: {len(test_repr)}\n')
        print("Dataset split summary written to readme.txt.")

def make_inference_dataset(config):
    data_file = config['data_file']
    train_data_path = config['train_data_path']
    save_path = config.get('save_path', train_data_path)
    output_name = config.get('output_name', 'inference')
    
    # Ensure save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Load full training data
    print("Loading training data...")
    train_data = _load_train_data_full(train_data_path)
    
    if train_data is None:
        raise FileNotFoundError(
            f"Train data file not found in {train_data_path}. "
            f"Please run train mode first."
        )
    
    train_HBA_smiles = train_data['HBA_smiles']
    train_HBD_smiles = train_data['HBD_smiles']
    train_HBA_molar = train_data['HBA_molar_fraction']
    train_HBA_repr = train_data['HBA_repr']
    train_HBD_repr = train_data['HBD_repr']
    
    # Check if pre-computed normalized repr is available
    has_precomputed_normalized = 'normalized_repr' in train_data
    
    # Build sample index for exact matching (HBA, HBD, molar) -> index
    sample_index = _build_sample_index(train_HBA_smiles, train_HBD_smiles, train_HBA_molar)
    
    # Build molecule pair index for partial matching (HBA, HBD) -> index
    pair_index = {}
    for i, (hba, hbd) in enumerate(zip(train_HBA_smiles, train_HBD_smiles)):
        pair_key = (hba, hbd)
        if pair_key not in pair_index:
            pair_index[pair_key] = i
    
    # Build individual molecule indices for HBA and HBD
    hba_molecule_index = {}
    hbd_molecule_index = {}
    for i, hba in enumerate(train_HBA_smiles):
        hba_molecule_index[hba] = i
    for i, hbd in enumerate(train_HBD_smiles):
        hbd_molecule_index[hbd] = i
    
    # Load normalization parameters from train mode
    if 'min_val' in train_data and 'max_val' in train_data:
        min_val = train_data['min_val']
        max_val = train_data['max_val']
    else:
        norm_file = train_data_path + 'data_range.npz'
        if not os.path.exists(norm_file):
            raise FileNotFoundError(f"Normalization parameters not found: {norm_file}")
        norm_data = np.load(norm_file)
        min_val = norm_data['min_val']
        max_val = norm_data['max_val']
    
    # Read inference CSV
    print("Loading inference data...")
    df = pd.read_csv(data_file)
    HBA_smiles_list = df['Smiles#1'].tolist()
    HBD_smiles_list = df['Smiles#2'].tolist()
    HBA_molar_fraction = df['X#1 (molar fraction)'].to_numpy().reshape(-1, 1)
    
    # Check if target column exists (optional for inference)
    has_target = 'Tmelt, K' in df.columns
    if has_target:
        target = df['Tmelt, K'].to_numpy()
    else:
        target = np.zeros(len(HBA_smiles_list))
    
    n_samples = len(HBA_smiles_list)
    repr_dim = 3073  # 1536 HBA + 1536 HBD + 1 molar fraction
    
    # Final normalized repr array
    repr_normalized = np.zeros((n_samples, repr_dim))
    
    # Track statistics
    samples_exact_match = 0
    samples_pair_match = 0
    samples_molecule_match = 0
    samples_computed = 0
    
    # Cache for newly computed reprs
    new_HBA_cache = {}
    new_HBD_cache = {}
    
    # Track which samples need normalization
    new_sample_indices = []
    new_sample_reprs = []
    
    print(f"Processing {n_samples} samples...")
    
    # Process each sample
    for i in range(n_samples):
        hba_smiles = HBA_smiles_list[i]
        hbd_smiles = HBD_smiles_list[i]
        molar = float(HBA_molar_fraction[i, 0])
        
        # Try to find exact match in training data
        exact_key = (hba_smiles, hbd_smiles, molar)
        pair_key = (hba_smiles, hbd_smiles)
        
        if exact_key in sample_index:
            # Case 1: Exact match found
            orig_idx = sample_index[exact_key]
            
            if has_precomputed_normalized:
                repr_normalized[i] = train_data['normalized_repr'][orig_idx]
            else:
                hba_repr = train_HBA_repr[orig_idx]
                hbd_repr = train_HBD_repr[orig_idx]
                raw_repr = np.concatenate([hba_repr, hbd_repr, [molar]])
                new_sample_indices.append(i)
                new_sample_reprs.append(raw_repr)
            
            samples_exact_match += 1
            
        elif pair_key in pair_index:
            # Case 2: HBA-HBD pair exists but with different molar fraction
            orig_idx = pair_index[pair_key]
            hba_repr = train_HBA_repr[orig_idx]
            hbd_repr = train_HBD_repr[orig_idx]
            raw_repr = np.concatenate([hba_repr, hbd_repr, [molar]])
            new_sample_indices.append(i)
            new_sample_reprs.append(raw_repr)
            samples_pair_match += 1
            
        else:
            # Case 3: No pair match - try to find individual molecules
            hba_found = hba_smiles in hba_molecule_index
            hbd_found = hbd_smiles in hbd_molecule_index
            
            # Get HBA repr
            if hba_found:
                hba_repr = train_HBA_repr[hba_molecule_index[hba_smiles]]
            elif hba_smiles in new_HBA_cache:
                hba_repr = new_HBA_cache[hba_smiles]
            else:
                hba_repr = _UniMol_repr_single(hba_smiles)
                new_HBA_cache[hba_smiles] = hba_repr
            
            # Get HBD repr
            if hbd_found:
                hbd_repr = train_HBD_repr[hbd_molecule_index[hbd_smiles]]
            elif hbd_smiles in new_HBD_cache:
                hbd_repr = new_HBD_cache[hbd_smiles]
            else:
                hbd_repr = _UniMol_repr_single(hbd_smiles)
                new_HBD_cache[hbd_smiles] = hbd_repr
            
            # Store for later normalization
            raw_repr = np.concatenate([hba_repr, hbd_repr, [molar]])
            new_sample_indices.append(i)
            new_sample_reprs.append(raw_repr)
            
            if hba_found or hbd_found:
                samples_molecule_match += 1
            else:
                samples_computed += 1
    
    # Normalize new samples
    if new_sample_reprs:
        new_reprs_array = np.array(new_sample_reprs)
        new_reprs_normalized, _, _ = normalize(new_reprs_array, min_val, max_val)
        for idx, norm_repr in zip(new_sample_indices, new_reprs_normalized):
            repr_normalized[idx] = norm_repr
    
    # Create smiles pairs list
    smiles_pairs = [(hba, hbd) for hba, hbd in zip(HBA_smiles_list, HBD_smiles_list)]
    
    # Save dataset
    make_torch_dataset(repr_normalized, target, smiles_pairs, save_path, output_name)
    return repr_normalized, target, smiles_pairs

def main(args):
    if args.mode == 'train':
        print('=' * 80)
        print('TRAINING MODE: Generating datasets with train/val/test split')
        print('=' * 80)

        dataset_config = {
            'data_file': args.data_file,
            'seed': args.seed,
            'save_path': args.save_path
        }

        make_training_dataset(dataset_config)
        
    elif args.mode == 'inference':
        print('=' * 80)
        print('INFERENCE MODE: Generating inference dataset')
        print('=' * 80)
        
        # train_data_path is required for inference (to load normalization params)
        if args.train_data_path is None:
            raise ValueError(
                "Error: --train_data_path is required for inference mode.\n"
                "This path should point to the training data directory containing normalization parameters.\n"
                "Example: --train_data_path ./processed_data/DES_melting_point_dataset/"
            )
        
        train_data_path = args.train_data_path
        
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(
                f"Train data path not found: {train_data_path}\n"
                f"Please run train mode first or specify correct --train_data_path"
            )
        
        # Use save_path (default is ./inference_output/)
        save_path = args.save_path
        # Ensure path ends with /
        if not save_path.endswith('/'):
            save_path += '/'
        
        inference_config = {
            'data_file': args.data_file,
            'train_data_path': train_data_path,
            'save_path': save_path,
            'output_name': args.output_name
        }
        
        make_inference_dataset(inference_config)

    print('\n' + '=' * 80)
    print('DONE!')
    print('=' * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate UniMol representations for training or inference.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Mode: train (generate train/val/test datasets) or inference (generate inference dataset)')
    
    # Common arguments
    parser.add_argument('--data_file', type=str, default='../raw_data/DES_melting_point_dataset.csv',
                        help='CSV file path')
    parser.add_argument('--save_path', type=str, default='../processed_data/DES_melting_point_dataset/',
                        help='Save path')
    
    # Train mode arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (train mode)')
    
    # Inference mode arguments
    parser.add_argument('--train_data_path', type=str, default=None,
                        help='Path to train mode output directory containing repr mappings and normalization params (inference mode)')
    parser.add_argument('--output_name', type=str, default='inference',
                        help='Output file name (inference mode)')
    
    args = parser.parse_args()
    main(args)
