import os
import yaml
import zipfile

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from typing import Union

import torch


def save_yaml(config: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def zip_files(output_filename: str, file_paths: list):
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))


class Tracker:
    """
    Logs training loss and plots them in real-time in a Jupyter notebook.
    """
    def __init__(
            self, 
            n_epochs: int,
            plot_freq: Union[int, None] = None, # plot every plot_freq epochs
            ):
        
        self.training_losses = []

        # evaluation metrics for supervised learning:
        self.losses_traindata = []
        self.accs_traindata = []
        self.losses_testdata = []
        self.accs_testdata = []

        self.plot = plot_freq is not None
        self.epoch = 0
        self.n_epochs = n_epochs
        if self.plot:
            self.plot_freq = plot_freq
            self.plot_results()


    def plot_results(self):
        self.fig, (self.loss_ax, self.acc_ax) = plt.subplots(1, 2, figsize=(16, 4))

        # Loss plot:
        self.training_loss_curve, = self.loss_ax.plot(
            range(1, len(self.training_losses)+1), 
            self.training_losses,
            '-o',
            label = 'train losses',
            )

        self.losses_traindata_curve, = self.loss_ax.plot(
            range(1, len(self.losses_traindata)+1), 
            self.losses_traindata,
            '-o',
            label = 'eval loss on train data',
            )
        
        self.losses_testdata_curve, = self.loss_ax.plot(
            range(1, len(self.losses_testdata)+1), 
            self.losses_testdata,
            '-o',
            label = 'eval loss on test data',
            )
        
        self.loss_ax.set_xlim(0, self.n_epochs+1)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Loss Curve')
        self.loss_ax.grid(linestyle='--', alpha=0.5)
        self.loss_ax.set_xticks(range(1, self.n_epochs+1))
        self.loss_ax.legend()
        
        # Accuracy plot:
        self.accs_traindata_curve, = self.acc_ax.plot(
            range(1, len(self.accs_traindata)+1), 
            self.accs_traindata,
            '-o',
            label = 'accuracy on train data',
            )
        
        self.accs_testdata_curve, = self.acc_ax.plot(
            range(1, len(self.accs_testdata)+1), 
            self.accs_testdata,
            '-o',
            label = 'accuracy on test data',
            )
        
        self.acc_ax.set_xlim(0, self.n_epochs+1)
        self.acc_ax.set_xlabel('Epoch')
        self.acc_ax.set_ylabel('Accuracy')
        self.acc_ax.set_title('Accuracy Curve')
        self.acc_ax.grid(linestyle='--', alpha=0.5)
        self.acc_ax.set_xticks(range(1, self.n_epochs+1))
        self.acc_ax.legend()
    
        # a text display of the latest epoch:
        self.text = self.loss_ax.text(1.01, 1.0, '', transform=self.loss_ax.transAxes, va='top', ha='left')


    def update(
            self, 
            training_loss: float,
            eval_loss_traindata: Union[float, None] = None,
            eval_acc_traindata: Union[float, None] = None,
            eval_loss_testdata: Union[float, None] = None,
            eval_acc_testdata: Union[float, None] = None,
            ):
        
        self.training_losses.append(training_loss)
        if eval_loss_traindata is not None:
            self.losses_traindata.append(eval_loss_traindata)
        if eval_acc_traindata is not None:
            self.accs_traindata.append(eval_acc_traindata)
        if eval_loss_testdata is not None:
            self.losses_testdata.append(eval_loss_testdata)
        if eval_acc_testdata is not None:
            self.accs_testdata.append(eval_acc_testdata)
        self.epoch += 1

        if self.plot and self.epoch % self.plot_freq == 0:

            # loss plot:
            self.training_loss_curve.set_data(range(1, len(self.training_losses)+1), self.training_losses)
            self.losses_traindata_curve.set_data(range(1, len(self.losses_traindata)+1), self.losses_traindata)
            self.losses_testdata_curve.set_data(range(1, len(self.losses_testdata)+1), self.losses_testdata)
            self.loss_ax.relim()
            self.loss_ax.autoscale_view()
            self.loss_ax.set_ylim(bottom=0.0, top=None)

            # accuracy plot:
            self.accs_traindata_curve.set_data(range(1, len(self.accs_traindata)+1), self.accs_traindata)
            self.accs_testdata_curve.set_data(range(1, len(self.accs_testdata)+1), self.accs_testdata)
            self.acc_ax.relim()
            self.acc_ax.autoscale_view()
            self.acc_ax.set_ylim(bottom=0.0, top=1.0)

            text = f'epoch {self.epoch}\n' + 20*'-' + '\n' + f'Train loss: {training_loss:.4f}'
            if eval_loss_traindata is not None:
                text += f'\nEval loss (train): {eval_loss_traindata:.4f}'
            if eval_acc_traindata is not None:
                text += f'\nEval acc (train): {eval_acc_traindata:.4f}'
            if eval_loss_testdata is not None:
                text += f'\nEval loss (test): {eval_loss_testdata:.4f}'
            if eval_acc_testdata is not None:
                text += f'\nEval acc (test): {eval_acc_testdata:.4f}'
            self.text.set_text(text)

            plt.tight_layout()
            self.fig.canvas.draw()
            clear_output(wait=True)
            display(self.fig)

# %% autograder

NT_Xent_test_cases = [3.556567907333374, 3.208645820617676, 3.2400364875793457, 11.805980682373047, 3.335388660430908]

def test_NT_Xent(NT_Xent, seed: int = 0):
    torch.manual_seed(seed)
    temp = 0.05 + 0.95 * torch.rand(1).item()
    loss_fn = NT_Xent(temp)
    N, D = torch.randint(8, 16, (2,))
    z1 = torch.randn(N, D)
    z2 = torch.randn(N, D)
    return loss_fn(z1, z2)

def Test_NT_Xent(NT_Xent):
    feedback = 21*'=' + ' NT-Xent ' + 20*'=' + '\n'
    for i, target in enumerate(NT_Xent_test_cases):
        try:
            torch.testing.assert_close(test_NT_Xent(NT_Xent, seed=i), torch.tensor(target))
            feedback += f'Test {i+1} passed!\n'
        except AssertionError as e:
            feedback += f'Test {i+1} failed!\n'
            feedback += f'{e}\n'
        feedback += 50*'-' + '\n'
    print(feedback)


Barlow_Twins_test_cases = [9.23009967803955, 9.101922988891602, 13.23111343383789, 9.119426727294922, 14.295135498046875]

def test_Barlow_Twins(Barlow_Twins, seed: int = 0):
    torch.manual_seed(seed)
    lambda_ = torch.randn(1).item() * 1e-3
    loss_fn = Barlow_Twins(lambda_)
    N, D = torch.randint(8, 16, (2,))
    z1 = torch.randn(N, D)
    z2 = torch.randn(N, D)
    return loss_fn(z1, z2)

def Test_Barlow_Twins(Barlow_Twins):
    feedback = 18*'=' + ' Barlow Twins ' + 18*'=' + '\n'
    for i, target in enumerate(Barlow_Twins_test_cases):
        try:
            torch.testing.assert_close(test_Barlow_Twins(Barlow_Twins, seed=i), torch.tensor(target))
            feedback += f'Test {i+1} passed!\n'
        except AssertionError as e:
            feedback += f'Test {i+1} failed!\n'
            feedback += f'{e}\n'
        feedback += 50*'-' + '\n'
    print(feedback)
