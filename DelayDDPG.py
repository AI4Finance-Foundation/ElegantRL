import os
import numpy as np
from torchvision.utils import save_image
from time import time as timer
import numpy.random as rd
import torch
import argparse
from yonv_utils import load_cifar10_data
from yonv_utils import whether_remove_history
from GAN_network import Generator, Discriminator

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
parser.add_argument("--mod_dir", type=str, default='WGAN_GP_3', help="directory of model")
parser.add_argument("--image_size", type=int, default=32, help="directory of model")
parser.add_argument("--train_epoch", type=int, default=256, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--show_gap", type=int, default=2 ** 6, help="")
parser.add_argument("--eval_gap", type=int, default=2 ** 8, help="")
parser.add_argument("--n_critic", type=int, default=4, help="number of training steps for discriminator per iter")
parser.add_argument("--lambda_gp", type=int, default=10, help="Gradient Penalty")

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()


# print(opt)


def calculate_gradient_penalty(discriminator, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, 1, 1, device=device, dtype=torch.float32)

    # interpolates = (alpha * real_samples + ((-alpha + 1) * fake_samples)).requires_grad_(True)
    interpolates = alpha * (real_samples - fake_samples) + fake_samples

    d_interpolates = discriminator(interpolates)
    fake = torch.ones(batch_size, 1, device=device,
                      dtype=torch.float32, requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_it(
        train_loader, device,
        generator, discriminator,
        optim_gene, optim_disc
):
    lambda_gp = opt.lambda_gp

    loss_sum_gene = 0
    loss_sum_disc = 0
    generator.train()
    discriminator.train()

    # smoothing label
    batch_size = train_loader.batch_size
    # real_label = torch.empty(batch_size, 1, dtype=torch.float32, device=device).fill_(0.9)
    # fake_label = torch.empty(batch_size, 1, dtype=torch.float32, device=device).fill_(0.1)

    fake_images = None
    i = j = 0
    for real_images in train_loader:
        k = rd.randint(2, batch_size)
        real_images[:k] = real_images[:k].flip(3)

        z = torch.randn((batch_size, opt.latent_dim), device=device, dtype=torch.float32, requires_grad=False)
        z = z.clamp(-1.0, 1.0)

        optim_disc.zero_grad()
        fake_images = generator(z)
        real_disc = discriminator(real_images).mean()
        fake_disc = discriminator(fake_images).mean()
        gradient_penalty = calculate_gradient_penalty(discriminator, real_images, fake_images, device)
        loss_disc = fake_disc - real_disc + lambda_gp * gradient_penalty
        loss_disc.backward()
        optim_disc.step()

        optim_gene.zero_grad()

        i += 1
        if i % opt.n_critic == 0:
            j += 1

            fake_images = generator(z)
            fake_disc = discriminator(fake_images).mean()
            g_loss = -fake_disc
            g_loss.backward()
            optim_gene.step()

            loss_sum_gene += g_loss.item()
        loss_sum_disc += loss_disc.item()

    loss_avg_gene = loss_sum_gene / j
    loss_avg_disc = loss_sum_disc / len(train_loader)
    return loss_avg_gene, loss_avg_disc, fake_images


def run_train(mod_dir, gpu_id,
              image_size,
              train_epoch, batch_size,
              show_gap, eval_gap):
    whether_remove_history(mod_dir, remove=True)
    train_epochs = [max(int(train_epoch * 0.6065 ** i), 1) for i in range(6)]
    batch_sizes = [int(batch_size * 1.6486 ** i) for i in range(6)]

    '''init env'''
    np.random.seed(1943 + int(timer()))
    torch.manual_seed(1943 + rd.randint(0, int(timer())))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # choose GPU:0
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    '''train model'''
    generator = Generator(opt.latent_dim, 64, 3).to(device)
    discriminator = Discriminator(3, 64, 1).to(device)
    optim_gene = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optim_disc = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999))
    # criterion = None

    '''load data'''
    import torch.utils.data as data
    train_images = load_cifar10_data(image_size)[0]
    train_images = train_images.to(device)

    print("Train Loop:")
    start_time = timer()
    eval_time = show_time = 0

    for train_epoch, batch_size in zip(train_epochs, batch_sizes):
        print(end="\n||%d/%d\t\t" % (batch_size, train_epoch))
        for epoch in range(train_epoch):
            batch_size += 1
            train_loader = data.DataLoader(
                train_images, batch_size=batch_size, shuffle=True, drop_last=True, )

            loss_gene, loss_disc, fake_images = train_it(
                train_loader, device,
                generator, discriminator,
                optim_gene, optim_disc
            )

            if timer() - show_time > show_gap:
                show_time = timer()
                print(end='\n  epoch:%4i |loss_G: %8.4f\t loss_D: %8.4f' % (epoch, loss_gene, loss_disc,))
            if timer() - eval_time > eval_gap:
                eval_time = timer()

                save_image(fake_images.data[:25], "%s/%04i%04i.png" % (mod_dir, batch_size, epoch),
                           nrow=5, normalize=True)
                print(end='  |EVAL')

    print("\nTimeUsed:", int(timer() - start_time))


if __name__ == '__main__':
    run_train(opt.mod_dir, opt.gpu_id,
              opt.image_size,
              opt.train_epoch, opt.batch_size,
              opt.show_gap, opt.eval_gap)
