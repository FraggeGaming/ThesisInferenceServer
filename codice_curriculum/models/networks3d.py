import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
#    elif classname.find('BatchNorm3d') != -1:
#        m.weight.data.normal_(1.0, 0.02)
#        m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]): # norm='instance' perchè quando questa funzione viene chiamata in pix2pix3d_model, viene specificato norm='instance'
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        #netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        netG = UnetGenerator()
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        #netG = UnetGenerator() # ATTENZIONE: questa è sempre una Unet_128
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init) #
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.ce_loss = nn.CrossEntropyLoss() # CrossEntropyLoss non si aspetta che l'output del modello sia già una probabilità, ma piuttosto i "logits" che rappresentano i punteggi per ogni classe. La funzione trasforma automaticamente i logits in probabilità usando softmax.

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real): # la variabile target_is_real è un indicatore booleano che viene utilizzato per distinguere se stai passando l'output di un generatore (fake) o un vero campione (real)
                                                        # se target_is_real = True --> immagini reali
                                                        # se target_is_real = False --> immagini generate
                                                        # Quando addestri il generatore, le sue immagini generate devono "ingannare" il discriminatore --> passo le immagini in uscita dal generatore (che queindi sono false) e dico al discriminatore che sono vere
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor


    # Versione senza classificazione (usata dal discriminatore)
    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



    # Versione con classificazione (usata dal generatore)
    def compute_with_classification(self, input, target_is_real, classification_output, classification_target):
        # Calcolo della GAN loss
        target_tensor = self.get_target_tensor(input, target_is_real)
        gan_loss = self.loss(input, target_tensor)

        # Assicurati che `classification_target` sia sullo stesso dispositivo di `classification_output`
        classification_target = classification_target.to(classification_output.device)

        # Calcolo della classification loss (Cross-Entropy)
        #print(f'classification_target: {classification_target}')
        #print(f'classification_output: {classification_output}')
        classification_loss = self.ce_loss(classification_output, classification_target)

        # Restituisci sia la GAN loss che la classification loss
        return gan_loss, classification_loss




class UnetGenerator(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # Encoder ATTENZIONE: cambiare track_running_state =False
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.bn2 = nn.InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) # affine=True
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.bn3 = nn.InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.bn4 = nn.InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv5 = nn.Conv3d(512, 512, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.relu5 = nn.ReLU(inplace=True)

        # Decoder
        self.convt1 = nn.ConvTranspose3d(512, 512, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.bn5 = nn.InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.relu6 = nn.ReLU(inplace=True)
        self.convt2 = nn.ConvTranspose3d(1024, 256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.bn6 = nn.InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.relu7 = nn.ReLU(inplace=True)
        self.convt3 = nn.ConvTranspose3d(512, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.bn7 = nn.InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.relu8 = nn.ReLU(inplace=True)
        self.convt4 = nn.ConvTranspose3d(256, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
        self.bn8 = nn.InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.relu9 = nn.ReLU(inplace=True)
        self.convt5 = nn.ConvTranspose3d(128, 1, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.tanh = nn.Tanh()

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 14)  # 14 classi = 14 distretti
        )

    def forward(self, x):
        # Parallelizzazione manuale
        if len(self.gpu_ids) > 1 and isinstance(x, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self._forward, x, self.gpu_ids)
        else:
            return self._forward(x)

    def _forward(self, x):

        x1 = self.conv1(x)
        #print(f"1st conv: {x1.shape}")
        x2 = self.relu1(x1)
        x3 = self.conv2(x2)
        #print(f"2nd conv: {x3.shape}")
        x4 = self.bn2(x3)
        x5 = self.relu2(x4)
        x6 = self.conv3(x5)
        #print(f"3rd conv: {x6.shape}")
        x7 = self.bn3(x6)
        x8 = self.relu3(x7)
        x9 = self.conv4(x8)
        #print(f"4th conv: {x9.shape}")
        x10 = self.bn4(x9)
        x11 = self.relu4(x10)
        x12 = self.conv5(x11)
        #print(f"5th conv: {x12.shape}")

        classification_output = self.classification_head(x12)

        x13 = self.relu5(x12)
        x14 = self.convt1(x13)

        #print(f"1st deconv:{x14.shape}")
        x15 = self.bn5(x14)
        c1=torch.cat((x15,x10),1)

        x16 = self.relu6(c1)
        x17 = self.convt2(x16)  # x8
        #print(f"2nd deconv:{x17.shape}")
        x18 = self.bn6(x17)

        c2=torch.cat((x18,x7),1)
        x19 = self.relu7(c2)

        x20 = self.convt3(x19)  # x4
        #print(f"3rd deconv:{x20.shape}")
        x21 = self.bn7(x20)

        c3=torch.cat((x21,x4),1)
        x22 = self.relu8(c3)
        x23 = self.convt4(x22)  # x2
        #print(f"4th deconv:{x23.shape}")
        x24 = self.bn8(x23)
        c4=torch.cat((x24,x1),1)
        x25 = self.relu9(c4)
        x26 = self.convt5(x25)  # x
        #print(f"5th deconv:{x26.shape}")
        output = self.tanh(x26)

        return output, classification_output




# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



#########################################################################################################################
#########################################################################################################################





class UnetGeneratorREBECCA(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorREBECCA, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


## Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d # Se Norm_layer=InstanceNorm3d --> bias= True
        else:
            use_bias = norm_layer == nn.InstanceNorm3d


        downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2,padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,kernel_size=4, stride=2,padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            print (self.model(x))
            return self.model(x)
        else:
            print(torch.cat([self.model(x), x], 1))
            return torch.cat([self.model(x), x], 1)




#############################
#
# 3D version of UnetGenerator --> L'ho commentata perchè non era usata neanche nel codice di Rebecca
#class UnetGenerator3d(nn.Module):
#    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
#                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]): # TODO
                                                                            # input_nc --> numero di canali di input
                                                                            # output_nc --> numero di canali di output
                                                                            # num_downs --> numero di livelli di downsampling che devono essere effettuati
                                                                            # ngf --> numero di filtri
#        super(UnetGenerator3d, self).__init__()
#        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
#        assert(input_nc == output_nc)

        # construct unet structure
#        unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True) #  Blocco più interno (bottleneck): Il primo blocco di downsampling senza alcun sottoblocco interno. Qui viene effettuato il downsampling finale.

        # blocchi che fanno il downsampling,
#        for i in range(num_downs - 5):
#            unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

#        unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer) # ngf * 4, ngf * 8, ngf * 2, ngf * 4, ngf, ngf * 2: riduce progressivamente il numero di filtri man mano che ci si avvicina all'output.
#        unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
#        unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
#        unet_block = UnetSkipConnectionBlock3d(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer) # convoluzione finale,

#        self.model = unet_block


#    def forward(self, input):
#       if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#          return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#       else:
#          return self.model(input)


#    def forward(self, input):
#        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#            output, classification_output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#        else:
#            output, classification_output = self.model(input)

#        return output, classification_output







if __name__ == '__main__':

    #gpu_ids = [0, 1, 2, 3]
    #tensore_prova = torch.cuda.FloatTensor(1, 1, 32, 32, 32)  # batch_size=128, input_channel=1, depth=32, height=32, width=32
    tensore_prova = torch.FloatTensor(1, 1, 32, 32, 32)
    generator = UnetGenerator()  # input_nc=1, output_nc=1, num_downs=5
    print(generator)
    # Sposta il modello sulla GPU
    #if len(gpu_ids) > 0:
     #   generator = generator.cuda(gpu_ids[0])  # Sposta l'istanza del modello sulla GPU principale
    #generator.apply(weights_init)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(generator)
    output = generator(tensore_prova)
    print(f"Numero totale di parametri nella rete: {num_params}")