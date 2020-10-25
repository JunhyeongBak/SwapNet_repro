from argparse import ArgumentParser
import os
import torch
from torch import nn

import modules.loss
from datasets.data_utils import unnormalize, remove_top_dir
#from models import BaseModel
#from models.base_gan import BaseGAN
from modules.swapnet_modules import WarpModule
from util.decode_labels import decode_cloth_labels
from util.util import PromptOnce
import optimizers
import modules.loss
from modules import discriminators
from modules.discriminators import Discriminator
from collections import OrderedDict

class WarpModel():
    """
    Implements training steps of the SwapNet Texture Module.
    """

    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        """
        Adds warp_mode option for generator loss. This is because Khiem found out using
        plain Cross Entropy works just fine. CE mode saves time and space by not having
        to train an additional discriminator network.
        """
        if is_train:
            #####################
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
            #############
            parser.add_argument("--warp_mode", default="gan", choices=("gan", "ce"))
            parser.add_argument(
                "--lambda_ce",
                type=float,
                default=100,
                help="weight for cross entropy loss in final term",
            )
            # based on the num entries in self.visual_names during training
            parser.set_defaults(display_ncols=4)
        # https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        #parser = super(WarpModel, WarpModel).modify_commandline_options(
        #    parser, is_train
        #)
        return parser

    def __init__(self, opt):
        """
        Initialize the WarpModel. Either in GAN mode or plain Cross Entropy mode.
        Args:
            opt:
        """
        # 3 for RGB
        self.body_channels = (
            opt.body_channels if opt.body_representation == "labels" else 3
        )
        # 3 for RGB
        self.cloth_channels = (
            opt.cloth_channels if opt.cloth_representation == "labels" else 3
        )

        #BaseGAN.__init__(self, opt) # or 
        ####super().__init__(opt)







        ######################################
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
        ######################################

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
        #######################################################################    


        # TODO: decode visuals for cloth
        self.visual_names = ["inputs_decoded", "bodys_unnormalized", "fakes_decoded"]

        if self.is_train:
            self.visual_names.append(
                "targets_decoded"
            )  # only show targets during training
            # we use cross entropy loss in both
            self.criterion_CE = nn.CrossEntropyLoss()
            if opt.warp_mode != "gan":
                # remove discriminator related things if no GAN
                self.model_names = ["generator"]
                self.loss_names = "G"
                del self.net_discriminator
                del self.optimizer_D
                self.optimizer_names = ["G"]
            else:
                self.loss_names += ["G_ce"]



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
    #################################################################################



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
        self.inputs_decoded = decode_cloth_labels(self.inputs)
        self.bodys_unnormalized = unnormalize(self.bodys, *self.opt.body_norm_stats)
        self.targets_decoded = decode_cloth_labels(self.targets)
        self.fakes_decoded = decode_cloth_labels(self.fakes)

    def define_G(self):
        """
        The generator is the Warp Module.
        """
        return WarpModule(
            body_channels=self.body_channels, cloth_channels=self.cloth_channels
        )

    def get_D_inchannels(self):
        """
        The Warp stage discriminator is a conditional discriminator. 
        This means we concatenate the generated warped cloth with the body segmentation.
        """
        return self.cloth_channels + self.body_channels

    def set_input(self, input):
        self.bodys = input["bodys"].to(self.device)
        self.inputs = input["input_cloths"].to(self.device)
        self.targets = input["target_cloths"].to(self.device)

        self.image_paths = tuple(zip(input["cloth_paths"], input["body_paths"]))

    def forward(self):
        self.fakes = self.net_generator(self.bodys, self.inputs)

    def backward_D(self):
        """
        Warp stage's custom backward_D implementation passes CONDITIONED input to 
        the discriminator. Concats the bodys with the cloth
        """
        # calculate fake
        conditioned_fake = torch.cat((self.bodys, self.fakes), 1)
        pred_fake = self.net_discriminator(conditioned_fake.detach())
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)
        # calculate real
        conditioned_real = torch.cat((self.bodys, self.targets), 1)
        pred_real = self.net_discriminator(conditioned_real)
        self.loss_D_real = self.criterion_GAN(pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)

        # calculate gradient penalty
        if any(gp_mode in self.opt.gan_mode for gp_mode in ["gp", "lp"]):
            self.loss_D_gp = (
                modules.loss.gradient_penalty(
                    self.net_discriminator,
                    conditioned_real,
                    conditioned_fake,
                    self.opt.gan_mode,
                )
                * self.opt.lambda_gp
            )
            self.loss_D += self.loss_D_gp

        # final loss
        self.loss_D.backward()

    def backward_G(self):
        """
        If GAN mode, loss is weighted sum of cross entropy loss and adversarial GAN
        loss. Else, loss is just cross entropy loss.
        """
        # cross entropy loss needed for both gan mode and ce mode
        loss_ce = (
            self.criterion_CE(self.fakes, torch.argmax(self.targets, dim=1))
            * self.opt.lambda_ce
        )

        # if we're in GAN mode, calculate adversarial loss too
        if self.opt.warp_mode == "gan":
            self.loss_G_ce = loss_ce  # store loss_ce

            # calculate adversarial loss
            conditioned_fake = torch.cat((self.bodys, self.fakes), 1)
            pred_fake = self.net_discriminator(conditioned_fake)
            self.loss_G_gan = self.criterion_GAN(pred_fake, True) * self.opt.lambda_gan

            # total loss is weighted sum
            self.loss_G = self.loss_G_gan + self.loss_G_ce
        else:
            # otherwise our only loss is cross entropy
            self.loss_G = loss_ce

        self.loss_G.backward()

    def optimize_parameters(self):
        """
        Optimize both G and D if in GAN mode, else just G.
        Returns:

        """
        if self.opt.warp_mode == "gan":
            # will optimize both D and G
            self.forward()
            # update D
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            # update G
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            ####super().optimize_parameters()
        else:
            self.forward()
            # optimize G only
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
