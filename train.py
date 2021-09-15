from elfragmentador.cli import train
import torch
import torch.autograd as autograd

if __name__ == "__main__":
    with autograd.detect_anomaly():
        train()
