from argparse import ArgumentParser


def get_dataset_generation_args(parser):
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                        help='Mode: train (generate train/val/test) or inference (generate inference dataset)')
    parser.add_argument('--data_file', type=str, required=True, 
                        help='Path to input CSV file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='./inference_output/',
                        help='Path to save generated datasets (default: ./inference_output/ for inference mode)')
    
    # Inference mode specific
    parser.add_argument('--train_data_path', type=str,
                        help='Path to training data directory (required for inference mode, for normalization)')
    parser.add_argument('--output_name', type=str, default='inference',
                        help='Output filename prefix (inference mode)')


def get_inference_args(parser):
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--data_file', type=str, required=True, 
                        help='Path to inference data (.pt file)')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='Path to save prediction results (.csv file)')
    parser.add_argument('--smiles_file', type=str, 
                        help='Path to SMILES file (optional)')
    parser.add_argument('--norm_file', type=str,
                        help='Path to normalization parameters file (optional)')
    parser.add_argument('--has_target', action='store_true',
                        help='Whether the data has target values for evaluation')
    
    # Model architecture (must match the trained model)
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts')
    parser.add_argument('--norm', type=str, default='LayerNorm', help='Normalization type')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='Activation function')
    
    # MOE architecture (must match the trained model)
    parser.add_argument('--load_balance_alpha', type=float, default=100,
                        help='Load balance loss weight')
    parser.add_argument('--adaptive_weight', type=str, default='true',
                        choices=['true', 'false'], help='Use adaptive weight for load balance loss')
    parser.add_argument('--target_ratio', type=float, default=0.8,
                        help='Target ratio for adaptive weight')


def get_args():
    parser = ArgumentParser(description='DeepDESmelt: DES Melting Point Prediction')
    subparsers = parser.add_subparsers(dest='parser_name', help='Available commands')
    
    # Dataset generation command (preprocessing)
    dataset_parser = subparsers.add_parser('generate_dataset', 
                                           help='Generate datasets from CSV')
    get_dataset_generation_args(dataset_parser)
    
    # Inference command
    inference_parser = subparsers.add_parser('predict', help='Make predictions')
    get_inference_args(inference_parser)
    
    args = parser.parse_args()
    
    if args.parser_name is None:
        parser.print_help()
        exit(1)
    
    return args
