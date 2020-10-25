from tqdm import tqdm
import time
from options.train_options import TrainOptions
from datasets import create_dataset
from models import create_model

from util.visualizer import Visualizer

#print = tqdm.write

##########----------##########----------##########----------##########----------##########----------

"""
def my_find_dataset_using_name(dataset_name):
    datasetlib = importlib.import_module("datasets." + dataset_name + "_dataset")
    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"

    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    print(dataset)
    exit(0)

    if dataset is None:
        raise NotImplementedError(
            f"In {dataset_filename}.py, there should be a subclass of BaseDataset "
            f"with class name that matches {target_dataset_name} in lowercase."
        )

    return dataset

def my_find_model_using_name(model_name):
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

"""
import torch
import importlib
from datasets.base_dataset import BaseDataset

class MyCappedDataLoader:
    """
    Create a custom datset object and load datas 
    """
    def __init__(self, opt, **ds_kwargs):
        self.opt = opt

        # Create custom dataset
        dataset_name = opt.dataset if opt.dataset else opt.model

        datasetlib = importlib.import_module("datasets." + dataset_name + "_dataset")
        dataset_class = None
        target_dataset_name = dataset_name.replace("_", "") + "dataset"

        for name, cls in datasetlib.__dict__.items():
            if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
                dataset_class = cls

        if dataset_class is None:
            raise NotImplementedError(
                f"In {dataset_filename}.py, there should be a subclass of BaseDataset "
                f"with class name that matches {target_dataset_name} in lowercase."
            )

        self.dataset = dataset_class(opt, **ds_kwargs)

        # Load datas using dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.shuffle_data,
            num_workers=opt.num_workers,
        )

    def __len__(self):
        # Return the number of dataset
        return min(len(self.dataset), self.opt.max_dataset_size) 

    def __iter__(self):
        # Return the batch of dataset
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

##########----------##########----------##########----------##########----------##########----------

#from models.base_model import BaseModel

def my_create_model(opt):
    model_name = opt.model
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance

##########----------##########----------##########----------##########----------##########----------

if __name__ == "__main__":
    # < Get training options >
    opt = TrainOptions().parse(store_options=True)  

    # < Create dataset obj and set data loader given options >
    dataset = MyCappedDataLoader(opt)
    dataset_size = len(dataset) # the number of images in the dataset

    # < Create a model given option >
    model = my_create_model(opt)
    model.setup(opt)

    # Create a visualizer given option
    visualizer = Visualizer(opt)

    ##########----------##########----------##########----------##########----------##########----------

    total_iters = 0 # the total number of training iterations

    for epoch in tqdm(range(opt.start_epoch + 1, opt.n_epochs + 1), desc="Completed Epochs"):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0 # the number of training iterations in current epoch, reset to 0 every epoch

        with tqdm(total=len(dataset), unit="image") as pbar:
            for i, data in enumerate(dataset): # inner loop within one epoch
                iter_start_time = time.time()
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                # Unpack data from dataset and preprocess
                model.set_input(data)
                #print(model)
                
                # Calculate loss functions, get gradients, update network weights
                model.optimize_parameters()

                # Display images on visdom and save images to a HTML file
                if total_iters % opt.display_freq == 0: 
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                losses = model.get_current_losses()
                Visualizer.just_print_losses(epoch, losses, print_func=lambda m: pbar.set_description(m))

                # Print training losses and save logging information to the disk
                if total_iters % opt.print_freq == 0:
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(
                        epoch,
                        epoch_iter,
                        losses,
                        t_comp,
                        t_data,
                        print_func=lambda *args: None,
                    )
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch - 1, float(epoch_iter) / dataset_size, losses)

                # Cache our latest model every <save_latest_freq> iterations
                if (opt.latest_checkpoint_freq and total_iters % opt.latest_checkpoint_freq == 0):
                    print(f"saving the latest model (epoch {epoch:d}, total_iters {total_iters:d}) ")
                    save_prefix = ("iter_%d" % total_iters if opt.save_by_iter else f"latest")
                    model.save_checkpoint(save_prefix)

                iter_data_time = time.time()

                # Weird unpacking to get the batch_size (we can't use opt.batch_size in case total len is not a multiple of batch_size
                pbar.update(len(tuple(data.values())[0]))

        # Cache our model every <save_epoch_freq> epochs
        if opt.checkpoint_freq and epoch % opt.checkpoint_freq == 0: 
            print(f"saving the model at the end of epoch {epoch:d}, iters {total_iters:d}")
            model.save_checkpoint("latest")
            model.save_checkpoint(epoch)

        # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))
        # model.update_learning_rate()                     # update learning rates at the end of every epoch.
