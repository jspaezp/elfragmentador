import torch
import torch.autograd as autograd

from elfragmentador.cli import train

if __name__ == "__main__":
    with autograd.detect_anomaly():
        train()
