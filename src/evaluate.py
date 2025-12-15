import argparse
import torch
import yaml
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import Net

def evaluate(params_path):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(params['preprocess']['root_dir'], train=False, transform=transform, download=False)
    test_loader = DataLoader(dataset, batch_size=1000)

    # Load Model
    model = Net().to(device)
    model.load_state_dict(torch.load(params['train']['save_path'], map_location=device))
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    # Save metrics (optional for DVC)
    with open("eval_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\nLoss: {test_loss}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='params.yaml')
    args = parser.parse_args()
    evaluate(args.params)
