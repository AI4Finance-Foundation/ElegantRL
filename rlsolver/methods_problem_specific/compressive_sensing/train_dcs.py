import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torchvision
import torchvision.transforms as transforms
from file_utils import *
from copy import deepcopy
from tqdm import tqdm
import sys

class Generator(nn.Module):
    def __init__(self, in_dim=100, out_dim=28**2, mid_dim = 512):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, mid_dim),nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.out_dim),nn.Tanh()
        )

    def forward(self, input):
        return self.net(input)

class Measurement(nn.Module):
    def __init__(self, in_dim=28**2, out_dim=25, mid_dim=500, bs=64, device=torch.device("cuda:0")):
        super(Measurement, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.bs = bs
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.measure = torch.randn(self.bs, self.out_dim, self.in_dim, device=self.device)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.mid_dim), nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.out_dim)
        )

    def forward(self, input):
        #return torch.bmm(self.measure, input.unsqueeze(dim=-1)).reshape(self.bs, -1)
        return self.net(input)

class Step_size(nn.Module):
    def __init__(self, initial_step_size=np.log(0.01)):
        #self.s = nn.Parameter(torch.tensor(0.01))
        # Other necessary setup
        super(Step_size, self).__init__()
        self.s = nn.Parameter(torch.tensor(initial_step_size))

    def forward(self, ):
        # Necessary forward computations
        return 1

def load_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
    return train_loader


def train_dcs(latent_dim=100, batch_size=64, num_training_epoch=100000, lr=1e-4, initial_step_size=0.01, num_grad_iters=4, device=torch.device("cuda:0")):
    
    file_exporter = FileExporter('./image', )
    training_data = load_data(batch_size)

    gen = Generator().to(device)
    measurement = Measurement().to(device)
    step_size = Step_size().to(device) #torch.nn.Parameter(torch.ones(1) * np.log(initial_step_size)).to(device)
    optimizer = torch.optim.Adam(list(gen.parameters()) + list(measurement.parameters()) + list( step_size.parameters()), lr=lr)
    MSELoss = nn.MSELoss()
    pbar = tqdm(range(num_training_epoch))
    n_batch = 0
    z_0 = torch.randn(batch_size, latent_dim, device=device, requires_grad=False)

    for epoch in pbar:
        for i, (images, labels) in enumerate(training_data):
            if (images.shape[0] == 32):
                continue
            original_data = images.reshape(batch_size, -1).to(device)
            original_data = (original_data) * 2 - 1
            z_initial = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
            measurement_original_data = measurement(original_data)
            z = torch.clone(z_initial)
            z = z / z.norm(keepdim=True, dim=-1)

            for itr in range(1, num_grad_iters):
                t = measurement(gen(z))
                l = (t - measurement_original_data).square().sum(dim=-1)
                g = grad(outputs=l, inputs=z,grad_outputs=torch.ones(64, device=device))[0]
                z = z - step_size.s.exp() * g
                z = z / z.norm(dim=-1, keepdim=True)
            
	    z_optimized = z
            generated_data_initial = gen(z_initial)
            generated_data_optimized = gen(z_optimized)
            measurement_generated_data_initial = measurement(generated_data_initial)
            measurement_generated_data_optimized = measurement(generated_data_optimized)
            generated_loss = (measurement_generated_data_optimized- measurement_original_data).square().sum(dim=-1).mean()
            RIP_loss = ((measurement_generated_data_initial - measurement_original_data).reshape(-1, 25).norm(dim=-1)- \
                                (generated_data_initial - original_data).norm(dim=-1)).square() + \
                	((measurement_generated_data_optimized - measurement_original_data).reshape(-1, 25).norm(dim=-1)- \
                                (generated_data_optimized - original_data).norm(dim=-1)).square() + \
                 	((measurement_generated_data_optimized - measurement_generated_data_initial).reshape(-1, 25).norm(dim=-1)- \
                                (generated_data_optimized - generated_data_initial).norm(dim=-1)).square()#.mean()
            loss = generated_loss + RIP_loss.mean() / 3
            
	    if  i % 50 ==  0:
			
                RECON_LOSS = (generated_data_optimized-original_data).norm(dim=-1).mean()
		
		torch.save(gen.state_dict(), "gen.pth")
                torch.save(measurement.state_dict(), "measurement.pth")
                torch.save(step_size.state_dict(), "step_size.pth")
                file_exporter.save((generated_data_optimized.detach().reshape(batch_size, 28, 28, 1).cpu().numpy() + 1) / 2, f'reconstruction{sys.argv[1]}')
                
		desc = f"nbatch: {n_batch} | RIP_LOSS: {(RIP_loss.mean() / 3):.2f} | GEN_LOSS: {generated_loss:.2f} | RECON_LOSS: {RECON_LOSS:.2f} | step_size: {step_size.s.exp().item():.5f} | optim_cost: {(z_optimized - z_initial).square().sum(dim=-1).mean():.2f} | z_mean:{z_initial.norm(dim=-1).mean():.2f}|z_opt.mean: {z_optimized.norm(dim=-1).mean():.2f}"
		pbar.set_description(desc)                
		wandb.log({"rip_loss": (RIP_loss.mean() / 3).item(), f"recons_loss:|x-z|_2": RECON_LOSS.item(), "z_step_size": step_size.s.exp().item(), f"gen_loss": generated_loss.item(), f"opt_loss":  (z_optimized - z_initial).square().sum(dim=-1).mean().item()})
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_batch += 1

if __name__ == "__main__":
    import wandb
    config = {
        'method': 'cs',
	'backend': 'pytorch',
	'dataset': 'MNIST',
        'latent_dim': 100,
    }
    wandb.init(
            project=f'compressive_sensing',
            entity="beamforming",
            sync_tensorboard=True,
            config=config,
            name='compressive_sensing',
            monitor_gym=True,
            save_code=True,
        )

    train_dcs()
