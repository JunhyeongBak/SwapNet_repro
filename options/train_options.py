import copy
import sys
import argparse
import json
import os

import torch

import datasets
import models
import optimizers
from util.util import PromptOnce

datasets, models, optimizers  # so auto import doesn't remove above

class TrainOptions():

    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            conflict_handler="resolve",
        )

        parser.add_argument(
            "--config_file",
            help="load arguments from a json file instead of command line",
        )
        parser.add_argument(
            "--name",
            default="my_experiment",
            help="name of the experiment, determines where things are saved",
        )
        parser.add_argument(
            "--comments",
            default="",
            help="additional comments to add to this experiment, saved in args.json",
        )
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument(
            "--display_winsize",
            type=int,
            default=256,
            help="display window size for both visdom and HTML",
        )
        parser.add_argument(
            "--model", help="which model to run", choices=("warp", "texture", "pix2pix")
        )
        parser.add_argument(
            "--checkpoints_dir", default="./checkpoints", help="Where to save models"
        )
        parser.add_argument(
            "--load_epoch",
            default="latest",
            help="epoch to load (use with --continue_train or for inference, 'latest' "
                 "for latest ",
        )
        parser.add_argument(
            "--dataroot",
            required=True,
            help="path to data, should contain 'cloth/', 'body/', 'texture/', "
                 "'rois.csv'",
        )
        parser.add_argument(
            "--dataset", help="dataset class to use, if none then will use model name"
        )
        parser.add_argument(
            "--dataset_mode",
            default="image",
            choices=("image", "video"),
            help="how data is formatted. video mode allows additional source inputs"
                 "from other frames of the video",
        )
        parser.add_argument(
            "--cloth_representation",
            default="labels",  # default according to SwapNet
            choices=("rgb", "labels"),
            help="which representation the cloth segmentations are in. 'labels' means "
                 "a 2D tensor where each value is the cloth label. 'rgb' ",
        )
        parser.add_argument(
            "--body_representation",
            default="rgb",  # default according to SwapNet
            choices=("rgb", "labels"),
            help="which representation the body segmentations are in",
        )
        parser.add_argument(
            "--cloth_channels",
            default=19,
            type=int,
            help="only used if --cloth_representation == 'labels'. cloth segmentation "
                 "number of channels",
        )
        parser.add_argument(
            "--body_channels",
            default=12,
            type=int,
            help="only used if --body_representation == 'labels'. body segmentation "
                 "number of channels. Use 12 for neural body fitting output",
        )
        parser.add_argument(
            "--texture_channels",
            default=3,
            type=int,
            help="RGB textured image number of channels",
        )
        parser.add_argument(
            "--pad", action="store_true", help="add a padding to make image square"
        )
        parser.add_argument(
            "--load_size",
            default=128,
            type=int,
            help="scale images (after padding) to this size",
        )
        parser.add_argument(
            "--crop_size", type=int, default=128, help="then crop to this size"
        )
        parser.add_argument(
            "--crop_bounds",
            help="DO NOT USE WITH --crop_size. crop images to a region: ((xmin, ymin), (xmax, ymax))",
        )
        parser.add_argument(
            "--max_dataset_size", type=int, default=float("inf"), help="cap on data"
        )
        parser.add_argument(
            "--batch_size", type=int, default=8, help="batch size to load data"
        )
        parser.add_argument(
            "--shuffle_data",
            default=True,
            type=bool,
            help="whether to shuffle dataset (default is True)",
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="number of CPU threads for data loading",
        )
        parser.add_argument(
            "--gpu_id", default=0, type=int, help="gpu id to use. -1 for cpu"
        )
        parser.add_argument(
            "--no_confirm", action="store_true", help="do not prompt for confirmations"
        )
        parser.add_argument(
            "--model",
            help="which model to run",
            choices=("warp", "texture", "pix2pix"),
            required=True
        )
        parser.add_argument(
            "--continue_train",
            action="store_true",
            help="continue training from latest checkpoint",
        )
        # visdom and HTML visualization parameters
        parser.add_argument(
            "--display_freq",
            type=int,
            default=400,
            help="frequency of showing training results on screen",
        )
        parser.add_argument(
            "--display_ncols",
            type=int,
            default=4,
            help="if positive, display all images in a single visdom web panel with "
            "certain number of images per row.",
        )
        parser.add_argument(
            "--display_id", type=int, default=1, help="window id of the web display"
        )
        parser.add_argument(
            "--display_server",
            type=str,
            default="http://localhost",
            help="visdom server of the web display",
        )
        parser.add_argument(
            "--display_env",
            type=str,
            default="main",
            help='visdom display environment name (default is "main")',
        )
        parser.add_argument(
            "--display_port",
            type=int,
            default=8097,
            help="visdom port of the web display",
        )
        parser.add_argument(
            "--update_html_freq",
            type=int,
            default=1000,
            help="frequency of saving training results to html",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="frequency of showing training results on console",
        )
        parser.add_argument(
            "--no_html",
            action="store_true",
            help="do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/",
        )
        # Training parameters
        parser.add_argument(
            "--n_epochs", "--num_epochs", default=20, type=int, help="number of epochs to train until"
        )
        parser.add_argument(
            "--start_epoch", type=int, default=0, help="epoch to start training from"
        )
        parser.add_argument(
            "--sample_freq",
            help="how often to sample and save image results from the generator",
        )
        parser.add_argument(
            "--checkpoint_freq",
            default=2,
            type=int,
            help="how often to save checkpoints. negative numbers for middle of epoch",
        )
        parser.add_argument(
            "--latest_checkpoint_freq",
            default=5120,
            type=int,
            help="how often (in iterations) to save latest checkpoint",
        )
        parser.add_argument(
            "--save_by_iter",
            action="store_true",
            help="whether saves model by iteration",
        )
        parser.add_argument(
            "--lr",
            "--learning_rate",
            type=float,
            default=0.01,
            help="initial learning rate",
        )
        parser.add_argument(
            "--wt_decay",
            "--weight_decay",
            dest="weight_decay",
            default=0,
            type=float,
            help="optimizer L2 weight decay",
        )
        # weights init
        parser.add_argument(
            "--init_type",
            default="kaiming",
            choices=("normal", "xavier", "kaiming"),
            help="weights initialization method",
        )
        parser.add_argument(
            "--init_gain", default=0.02, type=float, help="init scaling factor"
        )

        self._parser = parser
        self.is_train = True

    def gather_options(self):
        # Get parser(basic options)
        parser = self._parser
        opt, unparsed = parser.parse_known_args() # 이제 어느 함수에서든 opt를 접근할수 있음
        parser.set_defaults(dataset=opt.model)

        # modify options for each arg that can do so
        modifiers = ["model", "dataset"]
        if self.is_train:
            modifiers.append("optimizer_D")

        for arg in modifiers:
            import_source = eval(arg.split("_")[0] + "s") # 해당 이름으로 모델 불러오기 즉 = ex) models
            name = getattr(opt, arg) # opt.model에 있는 것

            if name is not None:
                options_modifier = import_source.get_options_modifier(name)
                parser = options_modifier(parser, self.is_train)
                opt, unparsed = parser.parse_known_args()

            if arg is "optimizer_D" and opt.optimizer_D != opt.optimizer_G:
                modifiers.append("optimizer_G")

        self._parser = parser
        final_opt = self._parser.parse_args()
        return final_opt

    @staticmethod
    def _validate(opt):
        """
        Validate that options are correct
        :return:
        """
        assert (
                opt.crop_size <= opt.load_size
        ), "Crop size must be less than or equal to load size "

    def parse(self, print_options=True, store_options=True, user_overrides=True):
        opt = self.gather_options()
        opt.is_train = self.is_train

        # perform assertions on arguments
        self._validate(opt)

        if opt.gpu_id > 0:
            torch.cuda.set_device(opt.gpu_id)
            torch.backends.cudnn.benchmark = True

        self.opt = opt

        # Load options from config file if present
        if opt.config_file:
            self.load(opt.config_file, user_overrides)

        if print_options:  # print what we parsed
            self.print()

        root = opt.checkpoints_dir if self.is_train else opt.results_dir
        self.save_file = os.path.join(root, opt.name, "args.json")
        if store_options:  # store options to file
            self.save()
        return opt

    def print(self):
        """
        prints the options nicely
        :return:
        """
        d = vars(self.opt)
        print("=====OPTIONS======")
        for k, v in d.items():
            print(k, ":", v)
        print("==================")

    def save(self):
        """
        Saves to a .json file
        :return:
        """
        d = vars(self.opt)

        PromptOnce.makedirs(os.path.dirname(self.save_file), not self.opt.no_confirm)
        with open(self.save_file, "w") as f:
            f.write(json.dumps(d, indent=4))

    def load(self, json_file, user_overrides):
        load(self.opt, json_file, user_overrides=user_overrides)



def load(opt, json_file, user_overrides=True):
    """

    Args:
        opt: Namespace that will get modified
        json_file:
        user_overrides: whether user command line arguments should override anything being loaded from the config file

    """
    opt = copy.deepcopy(opt)
    with open(json_file, "r") as f:
        args = json.load(f)

    # if the user specifies arguments on the command line, don't override these
    if user_overrides:
        user_args = filter(lambda a: a.startswith("--"), sys.argv[1:])
        user_args = set(
            [a.lstrip("-") for a in user_args]
        )  # get rid of left dashes
        print("Not overriding:", user_args)

    # override default options with values in config file
    for k, v in args.items():
        # only override if not specified on the cmdline
        if not user_overrides or (user_overrides and k not in user_args):
            setattr(opt, k, v)
    # but make sure the config file matches up
    opt.config_file = json_file
    return opt