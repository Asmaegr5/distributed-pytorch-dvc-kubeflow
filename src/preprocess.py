import os
import argparse
import torch
import yaml
import numpy as np

def preprocess(config_path):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    
    root_dir = params['preprocess']['root_dir']
    # Ensure we use the structure torchvision expects
    raw_dir = os.path.join(root_dir, 'MNIST', 'raw')
    processed_dir = os.path.join(root_dir, 'MNIST', 'processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Processing raw data from {raw_dir}...")
    
    # Map of standard usage -> user provided filename
    # User has dots instead of hyphens for extension separators
    files = {
        'train_img': 'train-images.idx3-ubyte',
        'train_lbl': 'train-labels.idx1-ubyte',
        'test_img':  't10k-images.idx3-ubyte',
        'test_lbl':  't10k-labels.idx1-ubyte'
    }

    # Helper to parse MNIST binaries
    # Helper to parse MNIST binaries with GZIP support and robust path checking
    import gzip
    
    def open_maybe_gz(filepath):
        # Try exact match
        if os.path.exists(filepath):
            return open(filepath, 'rb')
        
        # Try appending .gz (common if user downloaded raw files manually)
        gz_path = filepath + '.gz'
        if os.path.exists(gz_path):
            print(f"Found compressed file: {gz_path}")
            return gzip.open(gz_path, 'rb')
            
        raise FileNotFoundError(f"Could not find {filepath} or {gz_path}")

    def parse_images(filepath):
        abs_path = os.path.abspath(filepath)
        print(f"Parsing: {filepath}")
        print(f"  (Absolute: {abs_path})")
        
        with open_maybe_gz(filepath) as f:
            # Magic (4), Count (4), Rows (4), Cols (4) = 16 bytes
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return torch.from_numpy(data.copy()).view(-1, 28, 28)

    def parse_labels(filepath):
        abs_path = os.path.abspath(filepath)
        print(f"Parsing: {filepath}")
        
        with open_maybe_gz(filepath) as f:
             # Magic (4), Count (4) = 8 bytes
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return torch.from_numpy(data.copy()).long()

    # CI Fallback: If manual files don't exist, try standard download
    # This covers GitHub Actions or other environments with internet access
    if not os.path.exists(os.path.join(raw_dir, files['train_img'])):
        print(f"Manual files not found in {raw_dir}. Attempting automatic download (CI Mode)...")
        try:
            from torchvision import datasets, transforms
            # Define transform just to satisfy API, though we rely on internal processing
            transform = transforms.Compose([transforms.ToTensor()])
            datasets.MNIST(root_dir, train=True, download=True, transform=transform)
            datasets.MNIST(root_dir, train=False, download=True, transform=transform)
            
            # Ensure the marker file is created
            dvc_processed_dir = params['preprocess']['processed_dir']
            os.makedirs(dvc_processed_dir, exist_ok=True)
            with open(os.path.join(dvc_processed_dir, 'preprocessing_done.txt'), 'w') as f:
                f.write('done')
                
            print("Automatic download and processing completed.")
            return
        except Exception as e:
            print(f"Automatic download failed: {e}")
            print("Please ensure manual files are present as per README.")
            # Fall through to manual check which will error out properly
    
    # Process Manual Training Data (User's Local Mode with provided files)
    try:
        train_data = parse_images(os.path.join(raw_dir, files['train_img']))
        train_targets = parse_labels(os.path.join(raw_dir, files['train_lbl']))
        
        # Save as training.pt
        with open(os.path.join(processed_dir, 'training.pt'), 'wb') as f:
            torch.save((train_data, train_targets), f)
        print("Saved training.pt")

        # Process Test Data
        test_data = parse_images(os.path.join(raw_dir, files['test_img']))
        test_targets = parse_labels(os.path.join(raw_dir, files['test_lbl']))
        
        # Save as test.pt
        with open(os.path.join(processed_dir, 'test.pt'), 'wb') as f:
            torch.save((test_data, test_targets), f)
        print("Saved test.pt")
        
        # Create marker file for DVC
        dvc_processed_dir = params['preprocess']['processed_dir']
        os.makedirs(dvc_processed_dir, exist_ok=True)
        
        with open(os.path.join(dvc_processed_dir, 'preprocessing_done.txt'), 'w') as f:
            f.write('done')
            
        print("Manual processing and conversion completed successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure files are in {raw_dir} with names: {list(files.values())}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='params.yaml')
    args = parser.parse_args()
    
    preprocess(args.params)
