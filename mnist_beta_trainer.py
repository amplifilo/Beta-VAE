#import data
from dataset import return_data
import proplot as pplt
import numpy as np
import simplerRun as sR
from torch.autograd import Variable
import dill

def get_mnist_args():
    class _args(object):
        def __init__(self):
            pass

    _args.train= default=True
    _args.seed = 1
    _args.cuda = True
    _args.max_iter = 1.5e6
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

    _args.dataset='mnist'  #, type=str, help='dataset name')
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--max_iter', metavar='max_iter', type=int, required=False,
                        help='max iterations')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, required=False,
                        help='batch_size')


    # parser.add_argument('--schema', metavar='path', required=True,
    #                     help='path to schema')
    # parser.add_argument('--dem', metavar='path', required=True,
    #                     help='path to dem')
    # args = parser.parse_args()
    # main(workspace=args.workspace, schema=args.schema, dem=args.dem)
    args  = parser.parse_args()

    mnist_args = get_mnist_args()
    if args.max_iter is not None:
        mnist_args.max_iter = args.max_iter
    if args.batch_size is not None:
        mnist_args.batch_size = args.batch_size

    solver = sR.Solver(mnist_args);
    print(solver.max_iter)
    solver.train()

    ofile = open("trained.model", "wb")
    dill.dump(solver, ofile)
    ofile.close()

    print("Training finished. Let's see it if works.")
    solver.net.encoder(next(iter(solver.data_loader))[0])

