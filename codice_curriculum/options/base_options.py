import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=32, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=32, help='then crop to this size')
        self.parser.add_argument('--depthSize', type=int, default=32, help='depth for 3d images')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_128', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='Experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='pix2pix3d', help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.initialized = True

        #self.parser.add_argument('--random_district', action='store_true', help='distretti in ordine randomico')
        self.parser.add_argument('--grouped_district', action='store_true',help='continue training: load the latest model')
        self.parser.add_argument('--parto_da_switch', action='store_true', help='faccio partire training da switch')
        self.parser.add_argument('--switch_paths', type=str, default='./', help='path to the checkpoint folder')



        args, _ = self.parser.parse_known_args() # Analisi preliminare degli argomenti
        # NOT SWITCH
        # grouped
        if args.grouped_district and not args.parto_da_switch:
            self.parser.add_argument('--district', type=list, default=['adrenal_gland','gallbladder','thyroid','bladder','trachea','kidney','spleen','pancreas','stomach','brain','lung','liver','arms','legs'],help='lista di distretti segmentati')
        # NOT grouped
        elif not args.grouped_district and not args.parto_da_switch:
            self.parser.add_argument('--district', type=list, default=['adrenal_gland','thyroid','gallbladder','bladder','kidney','trachea','pancreas','spleen','brain','stomach','lung','liver','arms','legs'],help='lista di distretti segmentati') # i distretti sono ordinati in ordine crescente sulla base della difficoltà = dimensione (numero di pixel)

        # SWITCH
        else:
            #self.parser.add_argument('--district', type=list,default=['adrenal_gland','thyroid','gallbladder','bladder','kidney','trachea','pancreas','spleen','brain','stomach','lung','liver','arms','legs'],help='lista di distretti segmentati')  # SORTED
            #self.parser.add_argument('--district', type=list,default=['adrenal_gland','kidney','pancreas','arms','thyroid','spleen','liver','lung','stomach','brain','trachea','gallbladder','legs','bladder'],help='lista di distretti segmentati')  # RANDOM
            self.parser.add_argument('--district', type=list, default=['adrenal_gland', 'gallbladder', 'thyroid', 'bladder', 'trachea', 'kidney','spleen', 'pancreas', 'stomach', 'brain', 'lung', 'liver', 'arms','legs'], help='lista di distretti segmentati') # SORTED+GROUPED / SORTED+GROUPED+WARMUP

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        file_name = os.path.join(self.opt.switch_paths,'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


import torch
print(torch.version.cuda)  # Should print CUDA version
print(torch.backends.cudnn.enabled)  # Should be True if CUDA is working