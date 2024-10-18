# TTFS-ET Experiment evaluation

# Constants
from pathlib import Path
DIRECTORY = current_file_path = Path(__file__).resolve()
PATH_ROOT = str(DIRECTORY.parent.parent) + '/'
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
from sklearn.metrics import accuracy_score
import time
import traceback
from types import SimpleNamespace
from datetime import datetime

import model.simple_model as simple
import model.SCNN_model as scnn
import model.SMLP_784_400_10_model as smlp78440010
import model.SMLP_784_400_400_10_model as smlp78440040010

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
    
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))
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

def snn_training_loop(args, paths, model, train_loader, device, scheduler, optimizer, criterion, test_loader=None):
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
            inputs = spikegen.latency(inputs, tau=args.tau, num_steps=args.steps)
            outputs, _ = model(inputs)
            
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
        if args.verbose and test_loader is not None:
            snn_evaluation(args, paths, model, test_loader, device)

    # Load model parameters at best epoch loss
    if args.verbose:
        print(f'Load model state at best epoch loss [{best_epoch_idx}]')
    model.load_state_dict(best_model_state)

def snn_evaluation(args, paths, model, test_loader, device, channels=None, weight=None):
    # Usage: Determine model's accuracy (WHILE/AFTER training)

    # After training: model is None, there is weight
    if model is None:
        match args.model:
            case 'simple':
                model = simple.earlyTTFS(channels.input, channels.output, args.beta, channels.size) if args.early_termination else simple.defaultTTFS(channels.input, channels.output, args.beta, channels.size)
            case 'scnn' if channels.input==1 and channels.output==10 and channels.size==28:
                model = scnn.earlyTTFS(args.beta) if args.early_termination else scnn.defaultTTFS(args.beta)
            case 'smlp78440010' if channels.input==1 and channels.output==10 and channels.size==28:
                model = smlp78440010.earlyTTFS(args.beta) if args.early_termination else smlp78440010.defaultTTFS(args.beta)
            case 'smlp78440040010' if channels.input==1 and channels.output==10 and channels.size==28:
                model = smlp78440040010.earlyTTFS(args.beta) if args.early_termination else smlp78440040010.defaultTTFS(args.beta)
            case _:
                raise ValueError('Invalid model type')
        model.load_state_dict(weight)
        model = model.to(device)
    # While training: model is not None. Can be evaluated right away
    
    start_time = time.time()
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = spikegen.latency(inputs, tau=args.tau, num_steps=args.steps)
            spk_out = model(inputs)
            all_labels.extend(labels.cpu().numpy())

            # Compute spike times
            first_spike_time = (spk_out == 1).float() * torch.arange(1, len(spk_out) + 1).view(-1, 1, 1).to(device)   # Get each label's first spike time
            first_spike_time[first_spike_time == 0] = float('inf')

            spk_times, _ = torch.min(first_spike_time, dim=0)           # Get first spiked timimg(step) and label
            spk_times[spk_times == float('inf')] = args.steps + 1       # min value is inf, no spike occurred for all labels
            
            predicted = torch.argmin(spk_times, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
    end_time = time.time()

    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_classes=10, path=paths.accuracy)

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Accuracy: {accuracy:.6f}')

    accuracy_summary_record = {
        'xid': xid,
        'accuracy': accuracy,
        'dataset': args.dataset_type,
        'steps': args.steps,
        'epoch': args.epoch,
        'batch_size': args.batch_size,
        'early_termination': args.early_termination,
        'evaluation_time': end_time - start_time,
        'beta': args.beta,
        'tau': args.tau,
        'model': args.model,
    }
    save_record_to_csv(paths.accuracy_summary_csv, accuracy_summary_record)
    print(f"Execution time: {accuracy_summary_record['evaluation_time']} seconds")

def snn_train_pipeline(args, paths, device, train_loader, test_loader, channels):
    train_time = time.time()
    match args.model:
        case 'simple':
            model = simple.defaultTTFS(channels.input, channels.output, args.beta, channels.size)
        case 'scnn' if channels.input==1 and channels.output==10 and channels.size==28:
            model = scnn.defaultTTFS(args.beta)
        case 'smlp78440010' if channels.input==1 and channels.output==10 and channels.size==28:
            model = smlp78440010.defaultTTFS(args.beta)
        case 'smlp78440040010' if channels.input==1 and channels.output==10 and channels.size==28:
            model = smlp78440040010.defaultTTFS(args.beta)
        case _:
            raise ValueError('Invalid model type')
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
    snn_evaluation(args, paths, model, test_loader, device, channels)

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
    print(f"Device: {device}")
    
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
    elif (args.dataset_type=='fashionmnist'):
        transform = transforms.Compose([transforms.ToTensor()])
        ref_train_dataset = datasets.FashionMNIST(root=paths.dataset,
                                        train=True,
                                        transform=transform,
                                        download=True)
        ref_test_dataset = datasets.FashionMNIST(root=paths.dataset,
                                        train=False,
                                        transform=transform,
                                        download=True)
    else:
        raise ValueError('Invalid dataset type')
    example_image, _ = ref_train_dataset[0]
    channels = SimpleNamespace(input=example_image.shape[0], output=len(ref_train_dataset.classes), size=example_image.shape[-1])

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


    print_verbose(args.verbose, "SNN is on way")
    if args.pretrained is not None:
        print("###Sanity Check###")
        eval_time = time.time()
        snn_evaluation(args, paths, None, test_loader, device, channels, torch.load(args.pretrained, map_location=device))
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
    parser.add_argument('-d', '--dataset_type', type=str, default='mnist', choices=['mnist','fashionmnist'], help="Type of a dataset to use. (Default: mnist)")
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'scnn', 'smlp78440010', 'smlp78440040010'], help="Type of a model to use. (Default: simple)")
    parser.add_argument('--training_dataset_ratio', type=restricted_float, default=1.0, help="Set a specific ratio of training dataset. (Default: 1.0)") 
    parser.add_argument('--test_dataset_ratio', type=restricted_float, default=1.0, help="Set a specific ratio of test dataset. (Default: 1.0)")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Size of a batch. (Default: 64)")
    parser.add_argument('-e', '--epoch', type=int, default=100, help="Size of an epoch. (Default: 10)")
    parser.add_argument('--beta', type=restricted_float, default=0.95, help="Beta value for Leaky Integrate-and-Fire neuron. (Default: 0.95)")
    parser.add_argument('--tau', type=float, default=0.1, help="Tau value for SNN spike encoding. (Default: 0.1)")
    parser.add_argument('--lr_scheduler', action='store_true', default=False, help="Applying LR Scheduling method. (Default: False)")
    parser.add_argument('--steps', type=int, default=10, help="Number of steps in SNN spike encoding. Must be greater than 1. (Default: 10)")
    parser.add_argument('--single_gpu', type=int, default=None, help="Enable singleGPU mode with GPU index. Disable to use parallel GPU or CPU(when GPU is unavailable). (Default: None)")
    parser.add_argument('--path_dataset', type=str, default=PATH_DATASET)
    parser.add_argument('--path_result_model', type=str, default=PATH_RESULT_MODEL)
    parser.add_argument('--path_result_accuracy', type=str, default=PATH_RESULT_ACCURACY)
    parser.add_argument('--path_result_meta', type=str, default=PATH_RESULT_META)
    parser.add_argument('--pretrained', type=str, default=None, help="Pretrained latency encoding weights path (Default: None)")
    parser.add_argument('--early_termination', action='store_true', default=False, help="Apply early stopping at latency model. (Default: False)")
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
