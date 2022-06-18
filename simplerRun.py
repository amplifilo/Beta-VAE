#%%
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from utils import *
from dataset import *
from model import *
#%%

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()
#%%
def get_args():
    class _args(object):
        def __init__(self):
            pass
    '''
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

        parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
        parser.add_argument('--seed', default=1, type=int, help='random seed')
        parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
        parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
        parser.add_argument('--batch_size', default=64, type=int, help='batch size')

        parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
        parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
        parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
        parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
        parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
        parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
        parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
        parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

        parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
        parser.add_argument('--dataset', default='CelebA', type=str, help='dataset name')
        parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
        parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

        parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
        parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
        parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
        parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
        parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

        parser.add_argument('--gather_step', default=1000, type=int, help='numer of iterations after which data is gathered for visdom')
        parser.add_argument('--display_step', default=10000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
        parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')

        parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
        parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')


    '''

    _args.train= default=True
    _args.seed = 1
    _args.cuda = True
    _args.max_iter = 1e6
    _args.batch_size = 128
    _args.z_dim = 10
    _args.beta = 4          #'beta parameter for KL-term in original beta-VAE'
    _args.objective = 'H'   #'beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    _args.model = 'H'       #'model proposed in Higgins et al. or Burgess et al. H/B')
    _args.gamma = 1000      #'gamma parameter for KL-term in understanding beta-VAE')
    _args.C_max = 25        #, type=float, help='capacity parameter(C) of bottleneck channel')
    _args.C_stop_iter=1e5   #, type=float, help='when to stop increasing the capacity')
    _args.lr = 1e-4         #, type=float, help='learning rate')
    _args.beta1 = 0.9       #, type=float, help='Adam optimizer beta1')
    _args.beta2 = 0.999     #, type=float, help='Adam optimizer beta2')
    _args.dset_dir = 'data' #, type=str, help='dataset directory')

    _args.dataset='dsprites'  #, type=str, help='dataset name')
    _args.image_size=64     #, type=int, help='image size. now only (64,64) is supported')
    _args.num_workers=2     #, type=int, help='dataloader num_workers')

    _args.viz_on=False       #, type=str2bool, help='enable visdom visualization')
    _args.viz_name='main'   #, type=str, help='visdom env name')
    _args.viz_port=8097     #, type=str, help='visdom port number')
    _args.save_output=True  #, type=str2bool, help='save traverse images and gif')
    _args.output_dir='outputs'  #, type=str, help='output directory')

    _args.gather_step=1000  #, type=int, help='numer of iterations after which data is gathered for visdom')
    _args.display_step=10000    #, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    _args.save_step=10000   #, type=int, help='number of iterations after which a checkpoint is saved')

    _args.ckpt_dir='checkpoints'    #, type=str, help='checkpoint directory')
    _args.ckpt_name='last'      # , type=str, help='load previous checkpoint. insert checkpoint filename')
    return _args

class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'

        elif args.dataset.lower() == 'mnist':
            print('here')
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        # self.viz_name = args.viz_name
        # self.viz_port = args.viz_port
        # self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        # if self.viz_on:
        #     self.viz = visdom.Visdom(port=self.viz_port)

        # self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        # if not os.path.exists(self.ckpt_dir):
        #     os.makedirs(self.ckpt_dir, exist_ok=True)
        # self.ckpt_name = args.ckpt_name
        # if self.ckpt_name is not None:
        #     self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather = DataGather()

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for x in self.data_loader:
                if len(x) > 1:
                    x = x[0] #remove training labels -> unsupervised
                self.global_iter += 1
                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                # if self.viz_on and self.global_iter%self.gather_step == 0:
                #     self.gather.insert(iter=self.global_iter,
                #                        mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                #                        recon_loss=recon_loss.data, total_kld=total_kld.data,
                #                        dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        self.global_iter, recon_loss.data[0], total_kld.data[0], mean_kld.data[0]))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    pbar.write(var_str)

                    if self.objective == 'B':
                        pbar.write('C:{:.3f}'.format(C.data[0]))

                    # if self.viz_on:
                    #     self.gather.insert(images=x.data)
                    #     self.gather.insert(images=F.sigmoid(x_recon).data)
                    #     self.viz_reconstruction()
                    #     self.viz_lines()
                    #     self.gather.flush()

                    # if self.viz_on or self.save_output:
                    #     self.viz_traverse()

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter%50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.net_mode(train=True)

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()

        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        else:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))

        if self.win_mu is None:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=mus,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))
        else:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_mu,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))

        if self.win_var is None:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        else:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_var,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        self.net_mode(train=True)

    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)



            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            print(len(gifs))
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

# %%