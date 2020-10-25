from argparse import ArgumentParser

import torch
from torch import nn

import modules.losses
from datasets.data_utils import unnormalize, scale_tensor
#from models.base_gan import BaseGAN
from modules import get_norm_layer
from modules.pix2pix_modules import UnetGenerator
from modules.swapnet_modules import TextureModule
from util.decode_labels import decode_cloth_labels
from util.draw_rois import draw_rois_on_texture


import os
from util.util import PromptOnce
import optimizers
#from models import BaseModel
import modules.loss
from modules import discriminators
from modules.discriminators import Discriminator
from collections import OrderedDict


class TextureModel():
    """
    Implements training steps of the SwapNet Texture Module.
    """

    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        if is_train:
            # gan mode choice
            parser.add_argument(
                "--gan_mode",
                help="gan regularization to use",
                default="vanilla",
                choices=(
                    "vanilla",
                    "wgan",
                    "wgan-gp",
                    "lsgan",
                    "dragan-gp",
                    "dragan-lp",
                    "mescheder-r1-gp",
                    "mescheder-r2-gp",
                ),
            )
            parser.add_argument(
                "--lambda_gan",
                type=float,
                default=1.0,
                help="weight for adversarial loss",
            )
            parser.add_argument(
                "--lambda_discriminator",
                type=float,
                default=1.0,
                help="weight for discriminator loss",
            )
            parser.add_argument(
                "--lambda_gp",
                help="weight parameter for gradient penalty",
                type=float,
                default=10,
            )
            # discriminator choice
            parser.add_argument(
                "--discriminator",
                default="basic",
                choices=("basic", "pixel", "n_layers"),
                help="what discriminator type to use",
            )
            parser.add_argument(
                "--n_layers_D",
                type=int,
                default=3,
                help="only used if discriminator==n_layers",
            )
            parser.add_argument(
                "--norm",
                type=str,
                default="instance",
                help="instance normalization or batch normalization [instance | batch | none]",
            )
            # optimizer choice
            parser.add_argument(
                "--optimizer_G",
                "--opt_G",
                "--optim_G",
                help="optimizer for generator",
                default="AdamW",
                choices=("AdamW", "AdaBound"),
            )
            parser.add_argument(
                "--lr",
                "--g_lr",
                "--learning_rate",
                type=float,
                # as recommended by "10 Lessons I Learned Training GANs For a Year"
                default=0.0001,
                help="initial learning rate for generator",
            )
            parser.add_argument('--beta1', type=float, default=0.5,
                                help='momentum term of adam')
            parser.add_argument(
                "--optimizer_D",
                "--opt_D",
                "--optim_D",
                help="optimizer for discriminator",
                default="AdamW",
                choices=("AdamW", "AdaBound"),
            )
            parser.add_argument(
                "--d_lr",
                type=float,
                # as recommended by "10 Lessons I Learned Training GANs For a Year"
                default=0.0004,
                help="initial learning rate for Discriminator",
            )
            parser.add_argument(
                "--d_wt_decay",
                "--d_weight_decay",
                dest="d_weight_decay",
                default=0.01,
                type=float,
                help="optimizer L2 weight decay",
            )
            parser.add_argument(
                "--gan_label_mode",
                default="smooth",
                choices=("hard", "smooth"),
                help="whether to use hard (real 1.0 and fake 0.0) or smooth "
                "(real [0.7, 1.1] and fake [0., 0.3]) values for labels",
            )
            parser.add_argument(
                "--netG",
                default="swapnet",
                choices=["swapnet", "unet_128"]
            )
            parser.add_argument(
                "--lambda_l1",
                type=float,
                default=10,
                help="weight for L1 loss in final term",
            )
            parser.add_argument(
                "--lambda_content",
                type=float,
                default=20,
                help="weight for content loss in final term",
            )
            parser.add_argument(
                "--lambda_style",
                type=float,
                default=1e-8,  # experimentally set to be within the same magnitude as l1 and content
                help="weight for content loss in final term",
            )
            # based on the num entries in self.visual_names during training
            parser.set_defaults(display_ncols=5)
        return parser

    def __init__(self, opt):
        #super().__init__(opt)
        self.opt = opt
        self.gpu_id = opt.gpu_id
        self.is_train = opt.is_train
        # get device name: CPU or GPU
        self.device = (
            torch.device(f"cuda:{self.gpu_id}")
            if self.gpu_id is not None
            else torch.device("cpu")
        )
        # save all the checkpoints to save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if self.is_train:
            PromptOnce.makedirs(self.save_dir, not opt.no_confirm)

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizer_names = []
        # self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'


        self.net_generator = self.define_G().to(self.device)
        modules.init_weights(self.net_generator, opt.init_type, opt.init_gain)

        self.model_names = ["generator"]

        if self.is_train:
            # setup discriminator
            self.net_discriminator = discriminators.define_D(
                self.get_D_inchannels(), 64, opt.discriminator, opt.n_layers_D, opt.norm
            ).to(self.device)
            modules.init_weights(self.net_discriminator, opt.init_type, opt.init_gain)

            # load discriminator only at train time
            self.model_names.append("discriminator")

            # setup GAN loss
            use_smooth = True if opt.gan_label_mode == "smooth" else False
            self.criterion_GAN = modules.loss.GANLoss(
                opt.gan_mode, smooth_labels=use_smooth
            ).to(self.device)

            if opt.lambda_discriminator:
                self.loss_names = ["D", "D_real", "D_fake"]
                if any(gp_mode in opt.gan_mode for gp_mode in ["gp", "lp"]):
                    self.loss_names += ["D_gp"]
            self.loss_names += ["G"]
            if opt.lambda_gan:
                self.loss_names += ["G_gan"]

            # Define optimizers
            self.optimizer_G = optimizers.define_optimizer(
                self.net_generator.parameters(), opt, "G"
            )
            self.optimizer_D = optimizers.define_optimizer(
                self.net_discriminator.parameters(), opt, "D"
            )
            self.optimizer_names = ("G", "D")

        # TODO: decode cloth visual
        self.visual_names = [
            "textures_unnormalized",
            "cloths_decoded",
            "fakes",
            "fakes_scaled",
        ]
        if self.is_train:
            self.visual_names.append("targets_unnormalized")
            # Define additional loss for generator
            self.criterion_L1 = nn.L1Loss().to(self.device)
            self.criterion_perceptual = modules.losses.PerceptualLoss(
                use_style=opt.lambda_style != 0).to(self.device)

            for loss in ["l1", "content", "style"]:
                if getattr(opt, "lambda_" + loss) != 0:
                    self.loss_names.append(f"G_{loss}")






    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # if self.is_train:
        #     self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.is_train or opt.continue_train:
            self.load_checkpoint_dir(opt.load_epoch)
        self.print_networks(opt.verbose)
        return self

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net_" + name)
                net.eval()
        return self

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        # for scheduler in self.schedulers:
        #     if self.opt.lr_policy == 'plateau':
        #         scheduler.step(self.metric)
        #     else:
        #         scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_checkpoint(self, epoch):
        """Save all the networks to the disk.
        Or save latest.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f"{epoch}_net_{name}.pth"
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, f"net_{name}")

                if self.gpu_id is not None and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_id)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
        # todo: save optimizers too!
        for name in self.optimizer_names:
            if isinstance(name, str):
                save_filename = f"{epoch}_optim_{name}.pth"
                save_path = os.path.join(self.save_dir, save_filename)
                optim = getattr(self, f"optimizer_{name}")
                torch.save(optim.state_dict(), save_path)

    def load_model_weights(self, model_name, weights_file):
        """ Loads the weights for a single model
        Args:
            model_name: name of the model to load parameters into
            weights_file: path to weights file
        """
        net = getattr(self, f"net_{model_name}")
        print(f"loading the model {model_name} from {weights_file}")
        state_dict = torch.load(weights_file, map_location=self.device)
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        net.load_state_dict(state_dict)
        return self

    def load_checkpoint_dir(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{epoch}_net_{name}.pth"
                load_path = os.path.join(self.save_dir, load_filename)
                self.load_model_weights(name, load_path)

        if self.is_train:
            for name in self.optimizer_names:
                if isinstance(name, str):
                    load_filename = f"{epoch}_optim_{name}.pth"
                    load_path = os.path.join(self.save_dir, load_filename)
                    optim = getattr(self, f"optimizer_{name}")
                    print(f"loading the optimizer {name} from {load_path}")
                    state_dict = torch.load(load_path)
                    if hasattr(state_dict, "_metadata"):
                        del state_dict._metadata
                    optim.load_state_dict(state_dict)
        return self

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net_" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    "[Network %s] Total number of parameters : %.3f M"
                    % (name, num_params / 1e6)
                )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def optimize_parameters(self):
        self.forward()
        # update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    '''
    def backward_D(self):
        # https://github.com/martinarjovsky/WassersteinGAN/blob/f7a01e82007ea408647c451b9e1c8f1932a3db67/main.py#L185
        if self.opt.gan_mode == "wgan":
            # clamp parameters to a cube
            for p in self.net_discriminator.parameters():
                p.data.clamp(-0.01, 0.01)

        # calculate fake
        pred_fake = self.net_discriminator(self.fakes.detach())
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)
        # calculate real
        pred_real = self.net_discriminator(self.targets)
        self.loss_D_real = self.criterion_GAN(pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real) * self.opt.lambda_discriminator

        if any(gp_mode in self.opt.gan_mode for gp_mode in ["gp", "lp"]):
            # calculate gradient penalty
            self.loss_D_gp = modules.loss.gradient_penalty(
                self.net_discriminator, self.targets, self.fakes, self.opt.gan_mode
            )
            self.loss_D += self.opt.lambda_gp * self.loss_D_gp

        self.loss_D.backward()
    '''
































































    def compute_visuals(self):
        self.textures_unnormalized = unnormalize(
            self.textures, *self.opt.texture_norm_stats
        )
        self.textures_unnormalized = draw_rois_on_texture(
            self.rois, self.textures_unnormalized
        )
        self.cloths_decoded = decode_cloth_labels(self.cloths)

        self.fakes_scaled = scale_tensor(self.fakes, scale_each=True)

        if self.is_train:
            self.targets_unnormalized = unnormalize(
                self.targets, *self.opt.texture_norm_stats
            )
        # all batch, only first 3 channels
        # self.DEBUG_random_input = self.net_generator.DEBUG_random_input[:, :3] # take the top 3 layers, to 'sample' the RGB image

    def get_D_inchannels(self):
        return self.opt.texture_channels + self.opt.cloth_channels

    def define_G(self):
        if self.opt.netG == "unet_128":
            norm_layer = get_norm_layer("batch")
            return UnetGenerator(
                self.opt.texture_channels, self.opt.texture_channels, 7, 64, norm_layer=norm_layer, use_dropout=True
            )
        elif self.opt.netG == "swapnet":
            return TextureModule(
                texture_channels=self.opt.texture_channels,
                cloth_channels=self.opt.cloth_channels,
                num_roi=self.opt.body_channels,
                img_size=self.opt.crop_size,
                norm_type=self.opt.norm,
            )
        else:
            raise ValueError("Cannot find implementation for " + self.opt.netG)

    def set_input(self, input):
        self.textures = input["input_textures"].to(self.device)
        self.rois = input["rois"].to(self.device)
        self.cloths = input["cloths"].to(self.device)
        self.targets = input["target_textures"].to(self.device)

        self.image_paths = tuple(zip(input["cloth_paths"], input["texture_paths"]))

    def forward(self):
        if self.opt.netG == "swapnet":
            self.fakes = self.net_generator(self.textures, self.rois, self.cloths)
        elif self.opt.netG.startswith("unet_"):
            self.fakes = self.net_generator(self.textures)

    def backward_D(self):
        """
        Calculates loss and backpropagates for the discriminator
        """
        # https://github.com/martinarjovsky/WassersteinGAN/blob/f7a01e82007ea408647c451b9e1c8f1932a3db67/main.py#L185
        if self.opt.gan_mode == "wgan":
            # clamp parameters to a cube
            for p in self.net_discriminator.parameters():
                p.data.clamp(-0.01, 0.01)

        # calculate fake
        fake_AB = torch.cat((self.cloths, self.fakes), 1)
        pred_fake = self.net_discriminator(fake_AB.detach())
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)
        # calculate real
        real_AB = torch.cat((self.cloths, self.targets), 1)
        pred_real = self.net_discriminator(real_AB)
        self.loss_D_real = self.criterion_GAN(pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)

        if any(gp_mode in self.opt.gan_mode for gp_mode in ["gp", "lp"]):
            # calculate gradient penalty
            self.loss_D_gp = modules.loss.gradient_penalty(
                self.net_discriminator, self.targets, self.fakes, self.opt.gan_mode
            )
            self.loss_D += self.opt.lambda_gp * self.loss_D_gp

        self.loss_D.backward()

    def backward_G(self):
        """
        Backward G for Texture stage.
        Loss composed of GAN loss, L1 loss, and feature loss.
        Returns:

        """
        fake_AB = torch.cat((self.cloths, self.fakes), 1)
        pred_fake = self.net_discriminator(fake_AB)
        self.loss_G_gan = self.criterion_GAN(pred_fake, True) * self.opt.lambda_gan

        self.loss_G_l1 = (
                self.criterion_L1(self.fakes, self.targets) * self.opt.lambda_l1
        )
        self.loss_G_content = self.loss_G_style = 0
        if self.opt.lambda_content != 0 or self.opt.lambda_style != 0:
            self.loss_G_content, self.loss_G_style = self.criterion_perceptual(
                self.fakes, self.targets)
            self.loss_G_content *= self.opt.lambda_content
            self.loss_G_style *= self.opt.lambda_style

        # weighted sum
        self.loss_G = self.loss_G_gan + self.loss_G_l1 + self.loss_G_content + self.loss_G_style
        self.loss_G.backward()
