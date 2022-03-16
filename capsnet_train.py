# CapsNet Project
# This class trains the 3D capsule network.
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (4/10/21)
# Updated (1/15/22)

# -------------------------------------------------- Imports --------------------------------------------------

# Project imports:

from data_loader import AdniDataset, make_image_list
from capsnet_model import CapsNet3D
from loss_functions import DiceLoss

# System imports:

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from os.path import join
from shutil import copyfile
from datetime import datetime
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dipy.io.image import save_nifti


# ----------------------------------------------- TrainUNet3D class ------------------------------------------

class TrainCapsNet3D:

    def __init__(self, saved_model_path=None):
        self.start_time = datetime.now()

        ###########################################################
        #                  SET TRAINING PARAMETERS                #
        ###########################################################

        # Set segmentation target:
        self.output_structure = 'right hippocampus'
        # Set FreeSurfer code for segmentation target:
        # to find the code, open any aparc+aseg.mgz in FreeView and change color coding to lookup table
        self.output_code = 53

        # Set the size of the cropped volume:
        # if this is set to 100, the center of the volumed is cropped with the size of 100 x 100 x 100.
        # if this is set to (100, 64, 64), the center of the volume is cropped with size of (100 x 64 x 64).
        # note that 100, 64 and 64 here respectively represent left-right, posterior-anterior,
        # and inferior-superior dimensions, i.e. standard radiology coordinate system ('L','A','S').
        self.crop = (64, 64, 64)
        # Set cropshift:
        # if the target structure is right hippocampus, the crop box may be shifted to right by 20 pixels,
        # anterior by 5 pixels, and inferior by 20 pixels --> cropshift = (-20, 5, -20);
        # note that crop and cropshift here are set here using standard radiology system ('L','A','S'):
        self.cropshift = (-20, 0, -20)

        # Set model:
        self.model = CapsNet3D()
        # Set initial learning rate:
        self.lr_initial = 0.002
        # Set optimizer: default is Adam optimizer:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_initial)
        # Set learning rate scheduler parameters:
        # set the factor by which the learning rate will be reduced:
        self.lr_factor = 0.5
        # number of validation epochs without loss improvement before learning rate is decreased;
        # if patience = 4 --> optimizer decreases learning rate after 5 validation epochs without loss improvement:
        self.lr_patience = 9
        # ignore validation loss changes smaller than this threshold:
        self.lr_loss_threshold = 0.001
        self.lr_threshold_mode = 'abs'
        # don't decrease learning rate lower than this minimum:
        self.lr_min = 0.0001
        # Initiate the learning rate scheduler:
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                              factor=self.lr_factor,
                                              patience=self.lr_patience,
                                              threshold=self.lr_loss_threshold,
                                              threshold_mode=self.lr_threshold_mode,
                                              min_lr=self.lr_min,
                                              verbose=True)

        # Set loss function: options are DiceLoss, DiceBCELoss, and IoULoss:
        self.criterion = DiceLoss(conversion='margin', low=0.1, high=0.9)
        self.criterion_individual_losses = DiceLoss(conversion='threshold', reduction='none')  # for validation

        # .......................................................................................................

        # Set number of training epochs:
        self.n_epochs = 50

        # Number of training cases in each miniepoch:
        '''
        Miniepoch: a unit of training after which validation is done. 
        Since we have lots of training examples here (>3000), it's inefficient if we wait until after each 
        epoch to do validation. So I changed the paradigm to validation after each miniepoch rather than epoch:
        miniepoch 1 --> validate / update learning rate / save stats / save plots / ±save model
        --> minepoch 2 --> validate / update ...
        '''
        self.miniepoch_size_cases = 120
        # Set training batch size:
        self.train_batch_size = 4
        # Set if data augmentation should be done on training data:
        self.train_transforms = False

        # Set validation batch size:
        self.valid_batch_size = 16
        # Set if data augmentation should be done on validation data:
        self.valid_transforms = False

        # Set project root path:
        self.project_root = '/home/arman_avesta/capsnet'
        # Folder that contains datasets csv files:
        self.datasets_folder = 'data/datasets'
        # Folder to save model results:
        self.results_folder = 'data/results/temp'

        # csv file containing list of inputs for training:
        self.train_inputs_csv = 'train_inputs.csv'
        # csv file containing list of outputs for training:
        self.train_outputs_csv = 'train_outputs.csv'
        # csv file containing list of inputs for validation:
        self.valid_inputs_csv = 'valid_inputs.csv'
        # csv file containing list of outputs for validation:
        self.valid_outputs_csv = 'valid_outputs.csv'

        # Determine if backup to S3 should be done:
        self.s3backup = True
        # S3 bucket backup folder for results:
        self.s3_results_folder = 'HIDDEN FOR PUBLIC CODE'

        # .......................................................................................................
        ###################################
        #   DON'T CHANGE THESE, PLEASE!   #
        ###################################

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model from checkpoint if saved_model_path is provided:
        if saved_model_path is not None:
            self.load_model(saved_model_path)

        # Load lists of training and validation inputs and outputs:
        self.train_inputs = make_image_list(join(self.project_root, self.datasets_folder,
                                                 self.train_inputs_csv))
        self.train_outputs = make_image_list(join(self.project_root, self.datasets_folder,
                                                  self.train_outputs_csv))
        self.valid_inputs = make_image_list(join(self.project_root, self.datasets_folder,
                                                 self.valid_inputs_csv))
        self.valid_outputs = make_image_list(join(self.project_root, self.datasets_folder,
                                                  self.valid_outputs_csv))

        # Initialize dataloader for training and validation datasets:
        self.train_dataset = AdniDataset(self.train_inputs, self.train_outputs, maskcode=self.output_code,
                                         crop=self.crop, cropshift=self.cropshift, testmode=False)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

        self.valid_dataset = AdniDataset(self.valid_inputs, self.valid_outputs, maskcode=self.output_code,
                                         crop=self.crop, cropshift=self.cropshift, testmode=False)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.valid_batch_size, shuffle=False)

        # Training epochs:
        self.epoch = 1
        self.epochs = range(1, 1 + self.n_epochs)

        # Training miniepochs:
        self.miniepoch = 1
        self.miniepoch_size_batches = int(np.ceil(self.miniepoch_size_cases / self.train_batch_size))

        # Training iterations (batches):
        self.iterations = trange(1, 1 + self.n_epochs * len(self.train_dataloader),
                                 desc=f'Training '
                                      f'(epoch {self.epoch}, '
                                      f'miniepoch {self.miniepoch}, '
                                      f'LR {self.optimizer.param_groups[0]["lr"]: .4f})')
        self.iterations.update()  # to set the first value of self.iterations.n to 1

        # Training and validation losses:
        self.train_epoch_losses = pd.DataFrame()
        self.train_miniepoch_losses = pd.DataFrame()
        self.valid_losses = pd.DataFrame()

        # Computation times over each training iteration:
        self.train_times = pd.DataFrame()

        # Learning rates over training miniepochs:
        self.lrs = pd.DataFrame({f'm{self.miniepoch}_e{self.epoch}': [self.optimizer.param_groups[0]['lr']]})

        # Best model selection parameters:
        #################################################
        self.best_loss_threshold = self.lr_loss_threshold
        #################################################
        self.best_valid_loss = np.inf
        self.best_train_loss = np.inf
        self.best_lr = None
        self.best_time = None
        self.best_epoch = None
        self.best_miniepoch = None
        self.best_valid_loss_hx = []
        self.best_train_loss_hx = []
        self.best_lr_hx = []
        self.best_miniepoch_hx = []

        # .......................................................................................................

        # Run trainer:
        self.train()

        # Backup the results to S3:
        if self.s3backup:
            self.backup_to_s3()


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def train(self):
        print(f'''

        ###########################################################################

                            >>>   Starting training   <<< 

        Segmentation target:                    {self.output_structure}
        Segmentation target code:               {self.output_code}
        Cropped image size:                     {self.crop}
        Crop shift in (L,A,S) system:           {self.cropshift}

        Total training epochs:                  {self.n_epochs}
        Total miniepochs:                       {len(self.iterations) // self.miniepoch_size_batches}       
        Total iterations:                       {len(self.iterations)}

        Number of training images:              {len(self.train_dataset)}
        Training batch size:                    {self.train_batch_size}
        Batches in each epoch:                  {len(self.train_dataloader)} 

        Miniepochs in each epoch:               {len(self.train_dataloader) // self.miniepoch_size_batches}
        Batches in each miniepoch:              {self.miniepoch_size_batches}                      
        Images in each miniepoch:               {self.miniepoch_size_cases}

        Number of validation images:            {len(self.valid_dataset)}
        Validation batch size:                  {self.valid_batch_size}
        Batches in each validation epoch:       {len(self.valid_dataloader)}
        Validation frequency:                   {self.valid_frequency}
        
        S3 folder:                              {self.s3_results_folder}

        ###########################################################################
        ''')
        self.model = self.model.to(self.device)
        self.model.train()

        for self.epoch in self.epochs:

            for i, data_batch in enumerate(self.train_dataloader):
                t0 = datetime.now()

                inputs, targets = data_batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss_value = loss.item()
                loss.backward()
                self.optimizer.step()

                self.train_epoch_losses.at[i, f'e{self.epoch}'] = loss_value
                self.train_times.at[i, f'e{self.epoch}'] = datetime.now() - t0

                # ...... MINIEPOCH .................................................................................
                # After completion of each miniepoch --> validate, update records of miniepoch training loss and LR,
                # update LR scheduler, and save model if it beats the previous best model:
                if self.iterations.n % self.miniepoch_size_batches == 0:

                    # Validate:
                    self.validate()

                    # Update records of miniepochs training losses:
                    this_miniepoch_losses = pd.DataFrame(
                        {f'm{self.miniepoch}_e{self.epoch}':
                             self.train_epoch_losses
                                 .drop(index='averages', errors='ignore')
                                 .values
                                 .flatten(order='F')}
                    ).dropna().iloc[-self.miniepoch_size_batches:].reset_index(drop=True)

                    self.train_miniepoch_losses = pd.concat([self.train_miniepoch_losses, this_miniepoch_losses],
                                                            axis=1)

                    # Update records of miniepoch learning rates:
                    self.lrs.at[0, f'm{self.miniepoch}_e{self.epoch}'] = self.optimizer.param_groups[0]['lr']

                    # Update learning rate scheduler:
                    valid_loss = self.valid_losses.drop(index='averages', errors='ignore').iloc[:, -1].mean()
                    self.lr_scheduler.step(valid_loss)

                    # Save model if it beats the previous best model:
                    if valid_loss < self.best_valid_loss - self.best_loss_threshold:
                        self.best_valid_loss = valid_loss
                        self.best_train_loss = this_miniepoch_losses.mean().iloc[0]
                        self.best_lr = self.lrs.iloc[0, -1]
                        self.best_time = datetime.now() - self.start_time
                        self.best_epoch = self.epoch
                        self.best_miniepoch = self.miniepoch
                        self.best_valid_loss_hx.append(self.best_valid_loss)
                        self.best_train_loss_hx.append(self.best_train_loss)
                        self.best_lr_hx.append(self.best_lr)
                        self.best_miniepoch_hx.append(self.best_miniepoch)
                        self.save_model()

                    # Save stats and plots:
                    self.save_stats()
                    # Back up results to S3:
                    if self.s3backup:
                        self.backup_to_s3()
                    # Update miniepoch counter:
                    self.miniepoch += 1

                # ....... Update training progress ................................................................
                self.iterations.update()
                next_validation = (self.miniepoch_size_batches - self.iterations.n % self.miniepoch_size_batches
                                   if self.iterations.n % self.miniepoch_size_batches != 0 else 0)
                try:
                    self.iterations.set_description(f'Training '
                                                    f'(epoch {self.epoch}, '
                                                    f'miniepoch {self.miniepoch}, '
                                                    f'next valid {next_validation}, '
                                                    f'LR {self.optimizer.param_groups[0]["lr"]: .4f}, '
                                                    f'train loss {loss_value: .3f}, '
                                                    f'valid loss {valid_loss: .3f})')
                except (IndexError, UnboundLocalError):
                    self.iterations.set_description(f'Training '
                                                    f'(epoch {self.epoch}, '
                                                    f'miniepoch {self.miniepoch}, '
                                                    f'next valid {next_validation}, '
                                                    f'LR {self.optimizer.param_groups[0]["lr"]: .4f}, '
                                                    f'train loss {loss_value: .3f})')


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def validate(self):
        print('>>>   Validating   <<<')
        self.model.eval()

        this_epoch_losses = []

        for i, data_batch in enumerate(self.valid_dataloader):

            inputs, targets = data_batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                losses = self.criterion_individual_losses(outputs, targets)

            this_epoch_losses += list(losses.cpu().numpy())

        self.valid_losses = pd.concat([self.valid_losses,
                                       pd.DataFrame({f'm{self.miniepoch}_e{self.epoch}': this_epoch_losses})],
                                      axis=1)

        self.model.train()


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def save_stats(self):
        """
        This method computes training stats and writes the training stats and hyperparameters csv files:

        Outputs:
            - training_losses.csv (columns: miniepochs, rows: batches)
            - training_losses_epochs.csv (columns: epochs, rows: batches)
            - training_times.csv: training computation times (columns: epochs, rows: batches)
            - learning_rates.csv (columns: miniepochs, one row)
            - validation_losses.csv (columns: validation epochs (after each training miniepoch),
                                    rows: validation examples)

            These files are saved in the directory set by self.rerulst_folder
        """
        # Remove previous summary stats:
        self.train_epoch_losses.drop(index='averages', errors='ignore', inplace=True)
        self.train_miniepoch_losses.drop(index='averages', errors='ignore', inplace=True)
        self.valid_losses.drop(index='averages', errors='ignore', inplace=True)
        self.train_times.drop(index='totals', errors='ignore', inplace=True)

        # Add latest summary stats:
        self.train_epoch_losses.at['averages', :] = self.train_epoch_losses.mean()
        self.train_miniepoch_losses.at['averages', :] = self.train_miniepoch_losses.mean()
        self.valid_losses.at['averages', :] = self.valid_losses.mean()
        self.train_times.at['totals', :] = self.train_times.sum()

        # Sort data rows so that the summary stats will be the last row:
        # self.train_epoch_losses.sort_index(key=lambda xs: [str(x) for x in xs], inplace=True)
        # self.train_miniepoch_losses.sort_index(key=lambda xs: [str(x) for x in xs], inplace=True)
        # self.valid_losses.sort_index(key=lambda xs: [str(x) for x in xs], inplace=True)
        # self.train_times.sort_index(key=lambda xs: [str(x) for x in xs], inplace=True)

        # Save stats:
        os.makedirs(join(self.project_root, self.results_folder), exist_ok=True)

        self.train_epoch_losses.to_csv(join(self.project_root, self.results_folder,
                                            'training_losses_epochs.csv'))
        self.train_miniepoch_losses.to_csv(join(self.project_root, self.results_folder,
                                                'training_losses.csv'))
        self.valid_losses.to_csv(join(self.project_root, self.results_folder,
                                      'validation_losses.csv'))
        self.train_times.to_csv(join(self.project_root, self.results_folder,
                                     'training_times.csv'))
        self.lrs.to_csv(join(self.project_root, self.results_folder,
                             'learning_rates.csv'), index=False)
        print(f'>>>   Saved stats at epoch {self.epoch}, miniepoch {self.miniepoch}   <<<')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def save_model(self):
        os.makedirs(join(self.project_root, self.results_folder), exist_ok=True)
        complete_path = join(self.project_root, self.results_folder, 'saved_model.pth.tar')
        checkpoint = {'state_dict': self.model.state_dict()}
        # checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, complete_path)
        print(f'''
        >>>   SAVED MODEL at epoch {self.epoch}, miniepoch {self.miniepoch}   <<<
        ''')


    def load_model(self, saved_model_path):
        checkpoint = torch.load(saved_model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'>>>   Loaded the model from: {saved_model_path}   <<<')


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def backup_to_s3(self, verbose=False):
        """
        This method backs up the results to S3 bucket.
        It runs in the background and doesn't slow down training.
        """
        ec2_results_folder = join(self.project_root, self.results_folder)

        command = f'aws s3 sync {ec2_results_folder} {self.s3_results_folder}' if verbose \
            else f'aws s3 sync {ec2_results_folder} {self.s3_results_folder} >/dev/null &'

        os.system(command)
        print(f'>>>   S3 backup done at epoch {self.epoch}, miniepoch {self.miniepoch}   <<<')


# ------------------------------------------- Run TrainCapsNet3D Instance -------------------------------------------

if __name__ == '__main__':
    capstrain = TrainCapsNet3D()
