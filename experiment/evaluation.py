# F-TTFS Experiment evaluation

# Constants
from pathlib import Path
DIRECTORY = current_file_path = Path(__file__).resolve()
PATH_ROOT = str(DIRECTORY.parent.parent) + '/' #F-TTFS/
PATH_DATASET = PATH_ROOT + 'dataset/'
PATH_RESULT_ROOT = PATH_ROOT + 'result/'
PATH_RESULT_MODEL = PATH_RESULT_ROOT + 'model/'
PATH_RESULT_ACCURACY = PATH_RESULT_ROOT + 'accuracy/'
PATH_RESULT_META = PATH_RESULT_ROOT + 'meta/'
PATH_UTILITY = PATH_ROOT + 'utility/'

# Imports
import sys
sys.path.append(PATH_ROOT)
sys.path.append(PATH_DATASET)
sys.path.append(PATH_RESULT_ROOT)

# PyTorch family
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import spikegen
import snntorch.functional as SF

# Utilities
import argparse
import math
import re
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import traceback
from types import SimpleNamespace
from datetime import datetime


def save_record_to_csv(path: str, record: list):

    file_exists = os.path.isfile(path)
    
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = record.keys()
            writer.writerow(header)

        writer.writerow(record.values())

def visualize_confusion_matrix(pilot: bool, all_labels: list, all_predictions: list,
                               num_classes: int, path: str=None, *args):
    
    cm = confusion_matrix(all_labels, all_predictions)
    labels = list(range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)

    ### 제목 설정
    title = ' '.join(str(arg) for arg in args)
    plt.title(title)
    ###
    
    if pilot is True:
        plt.show()
        if(path is not None):
            plt.savefig(path)
    else:
        plt.tight_layout()
        plt.savefig(path)

class defaultTTFS(nn.Module):
    def __init__(self, input_channels=1, output_channels=10, beta=0.95, input_size=28):
        super(defaultTTFS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        # Compute the size after convolution and pooling layers
        size_after_conv1 = input_size  # Since padding=1, stride=1, kernel_size=3
        size_after_pool1 = size_after_conv1 // 2  # After first pooling
        size_after_conv2 = size_after_pool1  # Conv layer maintains size
        size_after_pool2 = size_after_conv2 // 2  # After second pooling
        final_size = size_after_pool2

        # Define fully connected layers with dynamic input size
        self.fc1 = nn.Linear(64 * final_size * final_size, 100)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2 = nn.Linear(100, output_channels)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        
        for step in range(num_steps):
            x_t = x[step]
            cur1 = self.conv1(x_t)
            spk1= self.lif1(cur1)
            spk1 = self.pool(spk1)
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool(spk2)
            spk2_flat = spk2.view(spk2.size(0), -1)
            cur3 = self.fc1(spk2_flat)
            spk3 = self.lif3(cur3)
            cur4 = self.fc2(spk3)
            spk4 = self.lif4(cur4)
            spk_out.append(spk4)
            
        return torch.stack(spk_out)
    
class earlyTTFS(nn.Module):
    def __init__(self, input_channels=1, output_channels=10, beta=0.95, input_size=28):
        super(earlyTTFS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        # Compute the size after convolution and pooling layers
        size_after_conv1 = input_size  # Since padding=1, stride=1, kernel_size=3
        size_after_pool1 = size_after_conv1 // 2  # After first pooling
        size_after_conv2 = size_after_pool1  # Conv layer maintains size
        size_after_pool2 = size_after_conv2 // 2  # After second pooling
        final_size = size_after_pool2

        # Define fully connected layers with dynamic input size
        self.fc1 = nn.Linear(64 * final_size * final_size, 100)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2 = nn.Linear(100, output_channels)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        batch_size = x.size(1)
        not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for step in range(num_steps):
            #x_t = x[step].clone()
            x_t = x[step]
            cur1 = self.conv1(x_t)
            spk1= self.lif1(cur1)
            spk1 = self.pool(spk1)
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool(spk2)
            spk2_flat = spk2.view(spk2.size(0), -1)
            cur3 = self.fc1(spk2_flat)
            spk3 = self.lif3(cur3)
            cur4 = self.fc2(spk3)
            spk4 = self.lif4(cur4)
            spk_out.append(spk4)
            spiked_now = spk4.sum(dim=1) > 0
            newly_spiked = spiked_now & not_spiked
            not_spiked[newly_spiked] = False
            if not not_spiked.any():
                for _ in range(step + 1, num_steps):
                    spk_out.append(torch.zeros_like(spk4))
                break
            
        return torch.stack(spk_out)

class zeroTTFS(nn.Module):
    def __init__(self, input_channels=1, output_channels=10, beta=0.95, input_size=28):
        super(zeroTTFS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        # Compute the size after convolution and pooling layers
        size_after_conv1 = input_size  # Since padding=1, stride=1, kernel_size=3
        size_after_pool1 = size_after_conv1 // 2  # After first pooling
        size_after_conv2 = size_after_pool1  # Conv layer maintains size
        size_after_pool2 = size_after_conv2 // 2  # After second pooling
        final_size = size_after_pool2

        # Define fully connected layers with dynamic input size
        self.fc1 = nn.Linear(64 * final_size * final_size, 100)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2 = nn.Linear(100, output_channels)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        batch_size = x.size(1)
        not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for step in range(num_steps):
            #x_t = x[step].clone()
            x_t = x[step]
            x_t[~not_spiked] = 0.0
            #x_t = x_t * not_spiked.view(-1, 1, 1, 1).float()
            cur1 = self.conv1(x_t)
            spk1= self.lif1(cur1)
            spk1 = self.pool(spk1)
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool(spk2)
            spk2_flat = spk2.view(spk2.size(0), -1)
            cur3 = self.fc1(spk2_flat)
            spk3 = self.lif3(cur3)
            cur4 = self.fc2(spk3)
            spk4 = self.lif4(cur4)
            spk_out.append(spk4)
            #spiked_now = spk4.sum(dim=1) > 0
            #newly_spiked = spiked_now & not_spiked
            newly_spiked = (spk4.sum(dim=1)>0) & not_spiked
            not_spiked[newly_spiked] = False
            
        return torch.stack(spk_out)
    
class earlyzeroTTFS(nn.Module):
    def __init__(self, input_channels=1, output_channels=10, beta=0.95, input_size=28):
        super(earlyzeroTTFS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        # Compute the size after convolution and pooling layers
        size_after_conv1 = input_size  # Since padding=1, stride=1, kernel_size=3
        size_after_pool1 = size_after_conv1 // 2  # After first pooling
        size_after_conv2 = size_after_pool1  # Conv layer maintains size
        size_after_pool2 = size_after_conv2 // 2  # After second pooling
        final_size = size_after_pool2

        # Define fully connected layers with dynamic input size
        self.fc1 = nn.Linear(64 * final_size * final_size, 100)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2 = nn.Linear(100, output_channels)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        batch_size = x.size(1)
        not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for step in range(num_steps):
            #x_t = x[step].clone()
            x_t = x[step]
            x_t[~not_spiked] = 0.0
            #x_t = x_t * not_spiked.view(-1, 1, 1, 1).float()
            cur1 = self.conv1(x_t)
            spk1= self.lif1(cur1)
            spk1 = self.pool(spk1)
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool(spk2)
            spk2_flat = spk2.view(spk2.size(0), -1)
            cur3 = self.fc1(spk2_flat)
            spk3 = self.lif3(cur3)
            cur4 = self.fc2(spk3)
            spk4 = self.lif4(cur4)
            spk_out.append(spk4)
            #spiked_now = spk4.sum(dim=1) > 0
            #newly_spiked = spiked_now & not_spiked
            newly_spiked = (spk4.sum(dim=1)>0) & not_spiked
            not_spiked[newly_spiked] = False
            if not not_spiked.any():
                for _ in range(step + 1, num_steps):
                    spk_out.append(torch.zeros_like(spk4))
                break
            
        return torch.stack(spk_out)
'''
# ALL IN ONE Model (Deprecated)
class CNV_SNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=10, beta=0.95):
        super(CNV_SNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 100)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        
        self.fc2 = nn.Linear(100, output_channels)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, num_steps, early_stopping=False, propagate_zero=False):
        # Initialize hidden states
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()

        # Number of time steps
        num_steps = x.shape[0]

        # Lists to store outputs
        spk_out = []
        #mem_out = []
        if early_stopping and not self.training:
            batch_size = x.size(1)
            not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            
            if propagate_zero:
                for step in range(num_steps):
                    # Current input
                    x_t = x[step]

                    # Zero out inputs for samples that have already spiked
                    x_t = x_t.clone()
                    x_t[~not_spiked] = 0.0

                    # Layer 1
                    cur1 = self.conv1(x_t)
                    spk1= self.lif1(cur1)
                    spk1 = self.pool(spk1)

                    # Layer 2
                    cur2 = self.conv2(spk1)
                    spk2 = self.lif2(cur2)
                    spk2 = self.pool(spk2)

                    # Flatten
                    spk2_flat = spk2.view(spk2.size(0), -1)

                    # Layer 3
                    cur3 = self.fc1(spk2_flat)
                    spk3 = self.lif3(cur3)

                    # Layer 4
                    cur4 = self.fc2(spk3)
                    spk4 = self.lif4(cur4)

                    # Collect spikes
                    spk_out.append(spk4)
                    
                    # Check if any output neuron spiked for each sample
                    spiked_now = spk4.sum(dim=1) > 0  # Boolean tensor of shape [batch_size]
                    # Identify samples that spiked in this time step and haven't spiked before
                    newly_spiked = spiked_now & not_spiked
                    # Update the not_spiked mask
                    not_spiked[newly_spiked] = False

                    # If all samples have spiked, break the loop
                    if not not_spiked.any():
                        for _ in range(step + 1, num_steps):
                            spk_out.append(torch.zeros_like(spk4))
                        break

            else: # Propagate zero is False
                for step in range(num_steps):
                    # Current input
                    x_t = x[step]

                    # Layer 1
                    cur1 = self.conv1(x_t)
                    spk1= self.lif1(cur1)
                    spk1 = self.pool(spk1)

                    # Layer 2
                    cur2 = self.conv2(spk1)
                    spk2 = self.lif2(cur2)
                    spk2 = self.pool(spk2)

                    # Flatten
                    spk2_flat = spk2.view(spk2.size(0), -1)

                    # Layer 3
                    cur3 = self.fc1(spk2_flat)
                    spk3 = self.lif3(cur3)

                    # Layer 4
                    cur4 = self.fc2(spk3)
                    spk4 = self.lif4(cur4)

                    # Collect spikes
                    spk_out.append(spk4)

                    # Update not_spiked and spike_times
                    # spk4 is of shape [batch_size, num_classes]
                    # Check if any output neuron spiked for each sample
                    spiked_now = spk4.sum(dim=1) > 0  # Boolean tensor of shape [batch_size]
                    # Identify samples that spiked in this time step and haven't spiked before
                    newly_spiked = spiked_now & not_spiked
                    # Update the not_spiked mask
                    not_spiked[newly_spiked] = False

                    # If all samples have spiked, break the loop
                    if not not_spiked.any():
                        for _ in range(step + 1, num_steps):
                            spk_out.append(torch.zeros_like(spk4))
                        break

        else: # latency coding's train & No option
            for step in range(num_steps):
                # Current input
                x_t = x[step]

                # Layer 1
                cur1 = self.conv1(x_t)
                spk1= self.lif1(cur1)
                spk1 = self.pool(spk1)

                # Layer 2
                cur2 = self.conv2(spk1)
                spk2 = self.lif2(cur2)
                spk2 = self.pool(spk2)

                # Flatten
                spk2_flat = spk2.view(spk2.size(0), -1)

                # Layer 3
                cur3 = self.fc1(spk2_flat)
                spk3 = self.lif3(cur3)

                # Layer 4
                cur4 = self.fc2(spk3)
                spk4 = self.lif4(cur4)

                # Collect spikes
                spk_out.append(spk4)
            
        return torch.stack(spk_out)
'''

def get_classification_metrics(labels: list, predictions: list, average=None) -> tuple:
    
    return (
        accuracy_score(labels, predictions),
        precision_score(labels, predictions, average=average),
        recall_score(labels, predictions, average=average),
        f1_score(labels, predictions, average=average)
    )


def snn_training_loop(args, paths, model, train_loader, device, scheduler, optimizer, criterion):
    epoch_loss_rec = []
    max_epoch_loss = math.inf
    for epoch in range(args.epoch):
        last_lr = scheduler.get_last_lr()[0]
        model.train() #change model's mode to train
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 옵티마이저 초기화
            optimizer.zero_grad()
            
            # 순전파
            inputs = spikegen.latency(inputs, tau=0.1, num_steps=args.steps)
            outputs = model(inputs)
            
            # 손실 계산
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            
            # 옵티마이저 업데이트
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Save best model
        if( epoch_loss < max_epoch_loss ):
            print(f'Model saved: Epoch [{epoch+1}] [Current] {epoch_loss:.4f} << {max_epoch_loss:.4f} [Max]')
            torch.save(model.state_dict(), paths.model)
            best_model_state = model.state_dict()
            best_epoch_idx = epoch + 1
            max_epoch_loss = epoch_loss
        
        # 학습률 감소 스케줄러에 검증 손실 전달
        if args.lr_scheduler:
            scheduler.step(epoch_loss)

            if scheduler.get_last_lr()[0] < (last_lr / 2):
                last_lr = scheduler.get_last_lr()[0]
                with open(paths.meta, 'a') as file:
                    file.write(f'\nlearning rate changed to {last_lr} at Epoch {epoch + 1}')
                print(f'learning rate changed to {last_lr} at Epoch {epoch + 1}')
            
            # 학습률 확인 및 출력
            print_verbose(args.verbose, f"epoch {epoch + 1}) Loss: {epoch_loss:.4f}, Learning rate: {scheduler.get_last_lr()[0]}")
        
        if( epoch % 10 == 9 ):
            print(f"Epoch {epoch + 1}/{args.epoch}, Loss: {epoch_loss:.4f}")
        
        epoch_loss_rec.append(epoch_loss)

    # Load model parameters at best epoch loss
    if args.verbose:
        print(f'Load model state at best epoch loss [{best_epoch_idx}]')
    model.load_state_dict(best_model_state)

def snn_evaluation(args, paths, model, test_loader, device):
    start_time = time.time()
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 모델에 데이터 전달
            inputs = spikegen.latency(inputs, tau=0.1, num_steps=args.steps)
            output = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            # Compute spike times
            first_spike_time = (output == 1).float() * torch.arange(1, args.steps + 1).view(-1, 1, 1).to(device)
            first_spike_time[first_spike_time == 0] = float('inf')
            spk_times, _ = torch.min(first_spike_time, dim=0)
            spk_times[spk_times == float('inf')] = args.steps + 1  # No spike occurred
            # Predict the class with the earliest spike
            predicted = torch.argmin(spk_times, dim=1)
            all_predictions.extend(predicted.cpu().numpy())

    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_classes=10, path=paths.accuracy)

    accuracy, _, _, _ = get_classification_metrics(all_labels, all_predictions, None)
    print(f'Accuracy: {accuracy:.6f}')

    accuracy_summary_record = {
        'xid': xid,
        'accuracy': accuracy,
        'dataset': args.dataset_type,
        'steps': args.steps,
        'epoch': args.epoch,
        'batch_size': args.batch_size,
        'latency_early_stopping': args.latency_early_stopping,
        'latency_propagate_zero': args.latency_propagate_zero,
        'execution_time': time.time() - start_time
    }
    save_record_to_csv(paths.accuracy_summary_csv, accuracy_summary_record)
    print(f"Execution time: {accuracy_summary_record['execution_time']} seconds")

def snn_train_pipeline(args, paths, device, train_loader, test_loader, channels):
    train_time = time.time()
    # Hyperparameter
    if args.latency_early_stopping and args.latency_propagate_zero:
        model = earlyzeroTTFS(channels.input, channels.output, args.beta, channels.size)
    elif args.latency_propagate_zero:
        model = zeroTTFS(channels.input, channels.output, args.beta, channels.size)
    elif args.latency_early_stopping:
        model = earlyTTFS(channels.input, channels.output, args.beta, channels.size)
    else:
        model = defaultTTFS(channels.input, channels.output, args.beta, channels.size)
    #model = CNV_SNN(channels.input, channels.output)
    model = model.to(device)
    criterion = SF.ce_temporal_loss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    # 학습률 스케줄링
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    print_verbose(args.verbose, f"training is on way")
    # Training loop
    snn_training_loop(args, paths, model, train_loader, device, scheduler, optimizer, criterion)
    print(f"{time.time()-train_time} second at training")

    print_verbose(args.verbose, f"testing is on way")

    # Test loop (evaluation)
    #test_time = time.time()
    snn_evaluation(args, paths, model, test_loader, device)
    #print(f"{time.time()-test_time} second at testing encoding")

    return model

def print_verbose(verbose, sentence):
    if verbose:
        print(sentence)
        
def main(args, paths):
    # Setup hyperparameters
    dtype = torch.float
    device = (
        torch.device(f"cuda:{args.single_gpu}") if (torch.cuda.is_available() and args.single_gpu != -1 and args.single_gpu is not None) else
        torch.device("cpu") if args.single_gpu == -1 else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    
    if ( args.dataset_type == 'mnist' ):
        transform = transforms.Compose([transforms.ToTensor()])
        ref_train_dataset = datasets.MNIST(root=paths.dataset,
                                       train=True,
                                       transform=transform,
                                       download=True)
        ref_test_dataset = datasets.MNIST(root=paths.dataset,
                                       train=False,
                                       transform=transform,
                                       download=True)
        #channels = SimpleNamespace(input=1, output=10)
    elif( args.dataset_type == 'cifar10' ):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            ])
        ref_train_dataset = datasets.CIFAR10(root=paths.dataset,
                                         train=True,
                                         transform=transform,
                                         download=True)
        ref_test_dataset = datasets.CIFAR10(root=paths.dataset,
                                        train=False,
                                        transform=transform,
                                        download=True)
        #channels = SimpleNamespace(input=3, output=10)
    elif( args.dataset_type == 'cifar100' ):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            ])
        ref_train_dataset = datasets.CIFAR100(root=paths.dataset,
                                         train=True,
                                         transform=transform,
                                         download=True)
        ref_test_dataset = datasets.CIFAR100(root=paths.dataset,
                                        train=False,
                                        transform=transform,
                                        download=True)
        #channels = SimpleNamespace(input=3, output=100)
    else:
        raise ValueError('Invalid dataset type')
    example_image, _ = ref_train_dataset[0]
    channels = SimpleNamespace(input=example_image.shape[0], output=len(ref_train_dataset.classes), size=example_image.shape[-1])

    # Modify proportion of the dataset
    # On train dataset
    if args.pretrained is not None:
        print_verbose(args.verbose, "No need any train dataset. Move right into evaluation.")
    else:
        train_loader = DataLoader(dataset=ref_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print_verbose(args.verbose, "Train dataset is ready")
    # On test dataset
    print_verbose(args.verbose, "Test dataset is on way")
    test_loader = DataLoader(dataset=ref_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print_verbose(args.verbose, "Test dataset is ready")

    ## 1. Train SNN
    print_verbose(args.verbose, "1. Train SNN is on way")
    if args.pretrained is not None:
        if args.latency_early_stopping and args.latency_propagate_zero:
            model = earlyzeroTTFS(channels.input, channels.output, args.beta, channels.size)
        elif args.latency_propagate_zero:
            model = zeroTTFS(channels.input, channels.output, args.beta, channels.size)
        elif args.latency_early_stopping:
            model = earlyTTFS(channels.input, channels.output, args.beta, channels.size)
        else:
            model = defaultTTFS(channels.input, channels.output, args.beta, channels.size)
        #model = CNV_SNN(channels.input, channels.output)
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
        model = model.to(device)
        if args.verbose:
            print("###Sanity Check###")
            eval_time = time.time()
            snn_evaluation(args, paths, model, test_loader, device)
            print(f"{time.time()-eval_time} second at test")
    else:
        model = snn_train_pipeline(args, paths, device, train_loader, test_loader, channels)

    print_verbose(args.verbose, "END")

def get_next_xid(path: str) -> int:
    max_id = -1
    pattern = re.compile(r'^(\d+)_')
    for filename in os.listdir(path):
        m = pattern.match(filename)
        if m:
            current_id = int(m.group(1))
            if(current_id > max_id):
                max_id = current_id
    return max_id + 1
    
def ensure_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")

def write_metadata_status(path: str, status: str):
    with open(path, 'a') as file:
        file.write(f'\nstatus: {status}')

        
def get_current_time_str() -> str:
    now = datetime.now()
    formatted_time = now.strftime("%y-%m-%d_%H-%M-%S")
    return formatted_time

def restricted_float(x: float):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} is not in range [0.0, 1.0]")
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument('-d', '--dataset_type', type=str, default='mnist', choices=['mnist','cifar10','cifar100'], help="Type of a dataset to use. (Default: mnist)")
    parser.add_argument('--training_dataset_ratio', type=restricted_float, default=1.0, help="Set a specific ratio of training dataset. (Default: 1.0)") 
    parser.add_argument('--test_dataset_ratio', type=restricted_float, default=1.0, help="Set a specific ratio of test dataset. (Default: 1.0)")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Size of a batch. (Default: 64)")
    parser.add_argument('-e', '--epoch', type=int, default=100, help="Size of an epoch. (Default: 10)")
    parser.add_argument('--beta', type=restricted_float, default=0.95, help="Beta value for Leaky Integrate-and-Fire neuron. (Default: 0.95)")
    parser.add_argument('--lr_scheduler', action='store_true', default=False, help="Applying LR Scheduling method. (Default: False)")
    parser.add_argument('--steps', type=int, default=10, help="Number of steps in SNN spike encoding. Must be greater than 1. (Default: 10)")
    parser.add_argument('--single_gpu', type=int, default=None, help="Enable singleGPU mode with GPU index. Disable to use parallel GPU or CPU(when GPU is unavailable). (Default: None)")
    parser.add_argument('--path_dataset', type=str, default=PATH_DATASET)
    parser.add_argument('--path_result_model', type=str, default=PATH_RESULT_MODEL)
    parser.add_argument('--path_result_accuracy', type=str, default=PATH_RESULT_ACCURACY)
    parser.add_argument('--path_result_meta', type=str, default=PATH_RESULT_META)
    parser.add_argument('--pretrained', type=str, default=None, help="Pretrained latency encoding weights path (Default: None)")
    parser.add_argument('--latency_early_stopping', action='store_true', default=False, help="Apply early stopping at latency model. (Default: False)")
    parser.add_argument('--latency_propagate_zero', action='store_true', default=False, help="Propagate zero at latency model. (Default: False)")
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode. (Default: False)")
    parser.add_argument('--notes', type=str, default=None)

    # Parsing arguments
    args = parser.parse_args()

    # Sanity check; directory existence
    ensure_directory(args.path_dataset)
    ensure_directory(args.path_result_model)
    ensure_directory(args.path_result_accuracy)
    ensure_directory(args.path_result_meta)

    # Write meta data
    current_time = get_current_time_str()
    xid = get_next_xid(args.path_result_meta)
    lines = []
    for key, value in vars(args).copy().items():
        line = f'{key}: {str(value)}'
        lines.append(line)

    paths = SimpleNamespace(
        dataset = args.path_dataset,
        meta = f'{args.path_result_meta}/{xid:03d}_meta_{args.dataset_type}.txt',
        model = f'{args.path_result_model}/{xid:03d}_model_{args.dataset_type}_latency.weights',
        accuracy = f'{args.path_result_accuracy}/{xid:03d}_accuarcy_{args.dataset_type}_latency.png',
        accuracy_csv = args.path_result_accuracy + f'accuracy.csv',
        accuracy_summary_csv = args.path_result_accuracy + f'accuracy_summary.csv',
    )

    # Sanity check: Print meta data
    if args.verbose:
        print(f"## XID: {xid} ##")
        print("## Meta data ##")
        for line in lines:
            print(line)
        print("#####")
    
    with open(paths.meta, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    
    # Execution
    try:
        start_time = time.time()
        main(args, paths)
        write_metadata_status(paths.meta, 'SUCCESS')
        print("SUCCESS")
    except KeyboardInterrupt:
        write_metadata_status(paths.meta, 'HALTED')
        print("HALTED")
    except Exception as e:
        _, _, tb = sys.exc_info()
        trace = traceback.format_tb(tb)
        
        write_metadata_status(paths.meta, f'FAILED({e})')
        with open(paths.meta, 'a') as file:
            file.writelines(trace)
            
        print(f"FAILED({type(e).__name__}: {e})")
        print(''.join(trace))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(paths.meta, 'a') as file:
        file.write(f'\nTotal elapsed time: {elapsed_time:.6f} s')
