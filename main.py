import sys
import os

# Add DeepDESmelt package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from args import get_args


if __name__ == '__main__':
    args = get_args()
    
    if args.parser_name == 'generate_dataset':
        print('=' * 80)
        print(f'DATASET GENERATION MODE: {args.mode.upper()}')
        print('=' * 80)
        from deepdesmelt.data.preprocessing import main as preprocessing_main
        preprocessing_main(args)
        
    elif args.parser_name == 'predict':
        print('=' * 80)
        print('PREDICTION MODE: Making predictions')
        print('=' * 80)
        from deepdesmelt.inference.predictor import main as inference_main
        inference_main(args)
        
    else:
        print(f'Error: Unknown command "{args.parser_name}"')
        sys.exit(1)
