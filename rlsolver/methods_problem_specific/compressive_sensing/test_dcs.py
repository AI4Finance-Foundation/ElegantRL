import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from file_utils import *
from copy import deepcopy
from tqdm import tqdm
import sys
from torch.autograd import Variable, grad
nc=1
# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64
class dcg(nn.Module):
    def __init__(self, ngpu=1):
        super(dcg, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 1, 1, 2, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)




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
class Policy(nn.Module):
    def __init__(self, in_dim=125, out_dim=100, mid_dim = 512):
        super(Policy, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
             nn.Linear(self.in_dim, mid_dim),nn.LeakyReLU(),
             nn.Linear(self.mid_dim, self.mid_dim),nn.LeakyReLU(),
             nn.Linear(self.mid_dim, self.out_dim),nn.Tanh()
         )
    def forward(self, input):
        x = self.net(input)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class Metric(nn.Module):
    def __init__(self, in_dim=28**2, out_dim=25, mid_dim=500, bs=64, device=torch.device("cuda:0")):
        super(Metric, self).__init__()
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
        return torch.bmm(self.measure, input.unsqueeze(dim=-1)).reshape(self.bs, -1)
        #return self.net(input)

class Step_size(nn.Module):
    def __init__(self, initial_step_size=np.log(0.01)):
        #self.s = nn.Parameter(torch.tensor(0.01))
        # Other necessary setup
        super(Step_size, self).__init__()
        self.s = nn.Parameter(torch.tensor(initial_step_size))

    def forward(self, ):
        # Necessary forward computations
        return 1
def load_data(batch_size, if_train=True):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=if_train,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=if_train,
                                               num_workers=0,
                                               pin_memory=True)
    return train_loader


def test_dcs(latent_dim=100, batch_size=64, m_dim=100, num_training_epoch=100000, lr=1e-4, initial_step_size=0.01, num_grad_iters=5, device=torch.device("cuda:0"), if_grad=False):
    file_exporter = FileExporter('./image', )
    training_data = load_data(batch_size, if_train=False)

    #gen = Generator().to(device)
    gen = dcg().to(device)
    measurement = Metric(out_dim=m_dim).to(device)
    step_size = Step_size().to(device) #torch.nn.Parameter(torch.ones(1) * np.log(initial_step_size)).to(device) #* np.log(initial_step_size)
    gen.load_state_dict(torch.load('./netG.pth'))
    #gen.load_state_dict(torch.load('./gen.pth'))
    #measurement.load_state_dict(torch.load('./measurement.pth'))
    #step_size.load_state_dict(torch.load('./step_size.pth'))
    policy = Policy().to(device)
    policy.load_state_dict(torch.load('./policy.pth'))
    MSELoss = nn.MSELoss()
    pbar = tqdm(range(num_training_epoch))
    n_batch = 0
    z_0 = torch.randn(batch_size, latent_dim, device=device, requires_grad=False)
    original_data_test = 1
    with open("train_X.pkl", 'rb') as f:
        import pickle as pkl
        train_dst = pkl.load(f).to(device)
    for epoch in pbar:
        for i, (images, labels) in enumerate(training_data):
            if (images.shape[0] == 32):
                continue
            original_data = images.reshape(batch_size, -1).to(device)
            original_data = (original_data) * 2 - 1
            
            if n_batch == 0:
                original_data_test = original_data
            else:
                original_data = original_data_test
            original_data = train_dst[:batch_size]
            z_initial = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
            #z_initial = z_initial / z_initial.norm(keepdim=True, dim=-1)
            measurement_original_data = measurement(original_data)
            z = torch.clone(z_initial)
            #z = torch.clone(z_initial)
            z = z #/ z.norm(keepdim=True, dim=-1)
            g = torch.randn_like(z, device=device)
            num_grad_iters = 5000
            for itr in range(num_grad_iters):
                t = measurement(gen(z.reshape(batch_size, latent_dim, 1, 1)).reshape(batch_size, -1))
                l = (t - measurement_original_data).norm(dim=-1).square()#.sum(dim=-1)
                g = grad(l, z, torch.ones(64, device=device), retain_graph=True, allow_unused=True)[0]
                if if_grad:
                    #z = z - step_size.s.exp() * g
                    z = z - 0.01 * g
                    #zn = z.norm(dim=-1, keepdim=True)
                    #for j in range(64):
                    #    if zn[j, 0] < 1e-6:
                    #        zn[j] = 1e-6 * th.ones(100)
                    #z = z / zn# max(1e-6,  z.norm(dim=-1, keepdim=True))
                else:
                    z = policy(torch.cat((z, measurement_original_data), dim=-1))
            z_optimized = z
            generated_data_initial = gen(z_initial.reshape(batch_size, latent_dim, 1, 1)).reshape(batch_size, -1)
            generated_data_optimized = gen(z_optimized.reshape(batch_size, latent_dim, 1, 1)).reshape(batch_size, -1)
            measurement_original_data = measurement(original_data)
            measurement_generated_data_initial = measurement(generated_data_initial)
            measurement_generated_data_optimized = measurement(generated_data_optimized)
            generated_loss = (measurement_generated_data_optimized- measurement_original_data).norm(dim=-1).square().mean()
            RIP_loss = ((measurement_generated_data_initial - measurement_original_data).reshape(-1, m_dim).norm(dim=-1)- \
                                (generated_data_initial - original_data).norm(dim=-1)).square() + \
                ((measurement_generated_data_optimized - measurement_original_data).reshape(-1, m_dim).norm(dim=-1)- \
                                (generated_data_optimized - original_data).norm(dim=-1)).square() + \
                 ((measurement_generated_data_optimized - measurement_generated_data_initial).reshape(-1, m_dim).norm(dim=-1)- \
                                (generated_data_optimized - generated_data_initial).norm(dim=-1)).square()#.mean()
            loss = generated_loss + (RIP_loss/ 3.0).mean()
            if  True:
                #torch.save(gen.state_dict(), "gen.pth")
                #torch.save(measurement.state_dict(), "measurement.pth")
                #torch.save(step_size.state_dict(), "step_size.pth")
                RECON_LOSS = (generated_data_optimized-original_data).norm(dim=-1).mean()
                print(n_batch, RECON_LOSS)
                desc = f"nbatch: {n_batch} | RIP_LOSS: {(RIP_loss.mean() / 3):.2f} | GEN_LOSS: {generated_loss:.2f} | RECON_LOSS: {RECON_LOSS:.2f} | step_size: {step_size.s.exp().item():.5f} | optim_cost: {(z_optimized - z_initial).square().sum(dim=-1).mean():.2f} | z_mean:{z_initial.norm(dim=-1).mean():.2f}|z_opt.mean: {z_optimized.norm(dim=-1).mean():.2f}"
                pbar.set_description(desc)
                #wandb.log({"rip_loss": (RIP_loss.mean() / 3).item(), f"recons_loss:|x-z|_2": RECON_LOSS.item(), "z_step_size": step_size.s.exp().item(), f"gen_loss": generated_loss.item(), f"opt_loss":  (z_optimized - z_initial).square().sum(dim=-1).mean().item()})
                file_exporter.save((original_data.detach().reshape(batch_size, 28, 28, 1).cpu().numpy() + 1) / 2, f'origin_{num_grad_iters}')
                file_exporter.save((generated_data_optimized.detach().reshape(batch_size, 28, 28, 1).cpu().numpy() + 1) / 2, f'reconstruction_{num_grad_iters}')
                #file_exporter.save((original_data.detach().reshape(batch_size, 28, 28, 1).cpu().numpy() + 0) / 1, f'origin_{num_grad_iters}')
                #file_exporter.save((generated_data_optimized.detach().reshape(batch_size, 28, 28, 1).cpu().numpy() + 0) / 1, f'reconstruction_{num_grad_iters}')

            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            assert 0
            n_batch += 1
            if n_batch >5:
                assert 0

if __name__ == "__main__":
    import wandb
    config = {
        'method': 'cs',
	'backend': 'pytorch',
	'dataset': 'MNIST',
        'latent_dim': 100,
    }
    '''
    wandb.init(
            project=f'compressive_sensing',
            entity="beamforming",
            sync_tensorboard=True,
            config=config,
            name='compressive_sensing',
            monitor_gym=True,
            save_code=True,
        )
    '''
    test_dcs()

