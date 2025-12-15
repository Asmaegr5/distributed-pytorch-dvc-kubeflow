import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import sys

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import Net

def train(rank, world_size, params):
    print(f"Running on rank {rank}/{world_size}")
    
    # 1. Setup Distributed Environment
    # If running locally without env vars, default to single process (for basic testing)
    setup_distributed = False
    if 'MASTER_ADDR' in os.environ:
        setup_distributed = True
        backend = params['train']['backend']
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # 2. Prepare Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(params['preprocess']['root_dir'], train=True, transform=transform, download=False)
    
    if setup_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None

    train_loader = DataLoader(dataset, batch_size=params['train']['batch_size'], sampler=sampler, shuffle=(sampler is None))

    # 3. Model Setup
    torch.manual_seed(params['train']['seed'])
    model = Net()
    
    # Move to device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
    else:
        device = torch.device("cpu")
    
    if setup_distributed:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = optim.SGD(model.parameters(), lr=params['train']['lr'], momentum=params['train']['momentum'])
    criterion = nn.NLLLoss()

    # 4. Training Loop
    epochs = params['train']['epochs']
    for epoch in range(1, epochs + 1):
        if setup_distributed:
            sampler.set_epoch(epoch)
            
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]: Loss: {loss.item()}")

    # 5. Save Model (only on rank 0)
    if rank == 0:
        save_path = params['train']['save_path']
        # Unwrap DDP model for saving
        state_dict = model.module.state_dict() if setup_distributed else model.state_dict()
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")

    if setup_distributed:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='params.yaml')
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)

    # Check for DDP environment variables passed by Kubeflow or torchrun
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    train(rank, world_size, params)

if __name__ == '__main__':
    main()
