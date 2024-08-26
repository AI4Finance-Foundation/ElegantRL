import torch
import torch_geometric.nn as tgnn
from rich import print
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from params import *
from utils import (
    GraphDataset,
    Hamiltonian_MaxCut,
    Hamiltonian_MaxIndSet,
    Hamiltonian_MinVerCover,
    Logger,
)


class PUBOsolver(nn.Module):
    def __init__(self, N) -> None:
        super().__init__()
        self.N = N
        self.layer1 = tgnn.GCNConv(INPUT_DIM(N), HIDDEN_DIM(N))
        self.layer2 = tgnn.GCNConv(HIDDEN_DIM(N), OUTPUT_DIM)

    def __init_embedding(self):
        return torch.randn(self.N, INPUT_DIM(self.N))

    def forward(self, edge_index, edge_attr):
        x = self.__init_embedding()
        x = F.relu(self.layer1(x, edge_index, edge_attr), inplace=True)
        x = torch.sigmoid(self.layer2(x, edge_index, edge_attr))
        return x.flatten()


def RelaxedHamiltonian(hamiltonian, solution):
    return solution @ hamiltonian @ solution


if __name__ == "__main__":
    dataset = DataLoader(
        GraphDataset(transform=Hamiltonian_MaxCut), batch_size=1, shuffle=True
    )
    for name, edge_index, edge_attr, hamiltonian in dataset:
        name = name[0]
        edge_index = edge_index.squeeze(0).to(DEVICE)
        edge_attr = edge_attr.squeeze(0).to(DEVICE)
        hamiltonian = hamiltonian.squeeze(0).to(DEVICE)

        logger = Logger(f"log/{name}.log")
        print("=" * 25, name, "=" * 25)

        solver = PUBOsolver(hamiltonian.shape[0]).to(DEVICE)
        optimizer = torch.optim.Adam(solver.parameters(), lr=LEARNING_RATE)
        criterion = RelaxedHamiltonian

        best_score = torch.inf
        best_solution = None
        # early stop strategy
        cumulative_improvement = 0
        last_significant_improvement = 0
        # unsupervised training
        logger("Training:")
        for epoch in range(NUM_EPOCH):
            output = solver(edge_index, edge_attr)
            loss = criterion(hamiltonian, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if loss < best_score:
                cumulative_improvement += best_score - loss
                best_score = loss
                best_solution = output
            if cumulative_improvement >= TOLERANCE:
                last_significant_improvement = epoch
                cumulative_improvement = 0
            elif epoch - last_significant_improvement >= PATIENCE:
                logger(f"Early stopping with best score {best_score.item():.5f}")
                break

            logger(
                f"[{epoch+1}/{NUM_EPOCH}], loss: {loss.item():.5f}, best score: {best_score.item():.5f}",
                print_=(epoch + 1) % 1000 == 0,
            )

        # evaluation
        logger("Evaluation:")
        for i in range(NUM_EVAL):
            output = solver(edge_index, edge_attr)
            loss = criterion(hamiltonian, output)
            if loss < best_score:
                best_score = loss
                best_solution = output
            logger(
                f"[{i+1}/{NUM_EVAL}], loss: {loss.item():.5f}, best score: {best_score.item():.5f}",
                print_=False,
            )
        # Final result
        logger(f"best score: {best_score.item():.5f} on {name}")
