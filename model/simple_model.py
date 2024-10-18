# for simple dataset like MNIST
import torch
import torch.nn as nn
import snntorch as snn

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
            newly_spiked = (spk4.sum(dim=1)>0) & not_spiked
            not_spiked[newly_spiked] = False
            if not not_spiked.any():
                break
            
        return torch.stack(spk_out)