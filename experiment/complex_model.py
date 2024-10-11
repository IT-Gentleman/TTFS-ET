# Complex Model for CIFAR-XX dataset
import torch
import torch.nn as nn
import snntorch as snn

class defaultTTFS(nn.Module):
    def __init__(self, input_channels=3, output_channels=10, beta=0.95, input_size=32):
        super(defaultTTFS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool3 = nn.MaxPool2d(2)
        
        # Compute the size after convolution and pooling layers
        size_after_conv1 = input_size  # Since padding=1, stride=1, kernel_size=3
        size_after_pool2 = size_after_conv1 // 2  # After first pooling
        size_after_pool3 = size_after_pool2 // 2  # After second pooling
        final_size = size_after_pool3

        # Define fully connected layers with dynamic input size
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_channels)
        self.lif5 = snn.Leaky(beta=beta, init_hidden=True)
    
    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()
        self.lif5.init_leaky()
        
        num_steps = x.shape[0]
        spk_out = []
        
        for step in range(num_steps):
            x_t = x[step]
            cur1 = self.conv1(x_t)
            spk1 = self.lif1(cur1)
            
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool2(spk2)
            
            cur3 = self.conv3(spk2)
            spk3 = self.lif3(cur3)
            spk3 = self.pool3(spk3)
            
            spk3_flat = spk3.view(spk3.size(0), -1)
            cur4 = self.fc1(spk3_flat)
            spk4 = self.lif4(cur4)
            spk4 = self.drop1(spk4)
            
            cur5 = self.fc2(spk4)
            spk5 = self.lif5(cur5)
            spk_out.append(spk5)
        
        return torch.stack(spk_out)
    
class earlyTTFS(nn.Module):
    def __init__(self, input_channels=1, output_channels=10, beta=0.95, input_size=28):
        super(earlyTTFS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool3 = nn.MaxPool2d(2)
        
        # Compute the size after convolution and pooling layers
        size_after_conv1 = input_size  # Since padding=1, stride=1, kernel_size=3
        size_after_pool2 = size_after_conv1 // 2  # After first pooling
        size_after_pool3 = size_after_pool2 // 2  # After second pooling
        final_size = size_after_pool3

        # Define fully connected layers with dynamic input size
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_channels)
        self.lif5 = snn.Leaky(beta=beta, init_hidden=True)
    
    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()
        self.lif5.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        batch_size = x.size(1)
        #spiked = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for step in range(num_steps):
            x_t = x[step].clone()
            #x_t = x[step]
            cur1 = self.conv1(x_t)
            spk1 = self.lif1(cur1)
            
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool2(spk2)
            
            cur3 = self.conv3(spk2)
            spk3 = self.lif3(cur3)
            spk3 = self.pool3(spk3)
            
            spk3_flat = spk3.view(spk3.size(0), -1)
            cur4 = self.fc1(spk3_flat)
            spk4 = self.lif4(cur4)
            spk4 = self.drop1(spk4)
            
            cur5 = self.fc2(spk4)
            spk5 = self.lif5(cur5)
            spk_out.append(spk5)
            #spiked_now = spk4.sum(dim=1) > 0
            #newly_spiked = spiked_now & not_spiked
            newly_spiked = (spk5.sum(dim=1)>0) & not_spiked
            not_spiked[newly_spiked] = False
            #spiked = spiked | (spk4.sum(dim=1) > 0)
            #if spiked.all():
            #if step % (num_steps // 5) == 0:
            #    if not not_spiked.any():
            #        for _ in range(step + 1, num_steps):
            #            spk_out.append(torch.zeros_like(spk4))
            #        break
            if not not_spiked.any():
                #print(f"Early stopping at step {step}")
                for _ in range(step + 1, num_steps):
                    spk_out.append(torch.zeros_like(spk5))
                break
            
        return torch.stack(spk_out)

class zeroTTFS(nn.Module):
    def __init__(self, input_channels=1, output_channels=10, beta=0.95, input_size=28):
        super(zeroTTFS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool3 = nn.MaxPool2d(2)
        
        # Compute the size after convolution and pooling layers
        size_after_conv1 = input_size  # Since padding=1, stride=1, kernel_size=3
        size_after_pool2 = size_after_conv1 // 2  # After first pooling
        size_after_pool3 = size_after_pool2 // 2  # After second pooling
        final_size = size_after_pool3

        # Define fully connected layers with dynamic input size
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_channels)
        self.lif5 = snn.Leaky(beta=beta, init_hidden=True)
    
    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()
        self.lif5.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        batch_size = x.size(1)
        #spiked = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for step in range(num_steps):
            x_t = x[step].clone()
            #x_t = x[step]
            #x_t[~not_spiked] = 0.0
            x_t = x_t * not_spiked.view(-1, 1, 1, 1).float()
            #x_t = x_t * (~spiked).view(-1, 1, 1, 1).float()
            #x_t[spiked] = 0.0
            cur1 = self.conv1(x_t)
            spk1 = self.lif1(cur1)
            
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool2(spk2)
            
            cur3 = self.conv3(spk2)
            spk3 = self.lif3(cur3)
            spk3 = self.pool3(spk3)
            
            spk3_flat = spk3.view(spk3.size(0), -1)
            cur4 = self.fc1(spk3_flat)
            spk4 = self.lif4(cur4)
            spk4 = self.drop1(spk4)
            
            cur5 = self.fc2(spk4)
            spk5 = self.lif5(cur5)
            spk_out.append(spk5)
            #spiked_now = spk4.sum(dim=1) > 0
            #newly_spiked = spiked_now & not_spiked
            newly_spiked = (spk5.sum(dim=1)>0) & not_spiked
            not_spiked[newly_spiked] = False
            #spiked = spiked | (spk4.sum(dim=1) > 0)

        return torch.stack(spk_out)
    
class earlyzeroTTFS(nn.Module):
    def __init__(self, input_channels=1, output_channels=10, beta=0.95, input_size=28):
        super(earlyzeroTTFS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool3 = nn.MaxPool2d(2)
        
        # Compute the size after convolution and pooling layers
        size_after_conv1 = input_size  # Since padding=1, stride=1, kernel_size=3
        size_after_pool2 = size_after_conv1 // 2  # After first pooling
        size_after_pool3 = size_after_pool2 // 2  # After second pooling
        final_size = size_after_pool3

        # Define fully connected layers with dynamic input size
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_channels)
        self.lif5 = snn.Leaky(beta=beta, init_hidden=True)
    
    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()
        self.lif5.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        batch_size = x.size(1)
        not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for step in range(num_steps):
            x_t = x[step].clone()
            #x_t = x[step]
            #x_t[~not_spiked] = 0.0
            x_t = x_t * not_spiked.view(-1, 1, 1, 1).float()
            cur1 = self.conv1(x_t)
            spk1 = self.lif1(cur1)
            
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool2(spk2)
            
            cur3 = self.conv3(spk2)
            spk3 = self.lif3(cur3)
            spk3 = self.pool3(spk3)
            
            spk3_flat = spk3.view(spk3.size(0), -1)
            cur4 = self.fc1(spk3_flat)
            spk4 = self.lif4(cur4)
            spk4 = self.drop1(spk4)
            
            cur5 = self.fc2(spk4)
            spk5 = self.lif5(cur5)
            spk_out.append(spk5)
            #spiked_now = spk4.sum(dim=1) > 0
            #newly_spiked = spiked_now & not_spiked
            newly_spiked = (spk5.sum(dim=1)>0) & not_spiked
            not_spiked[newly_spiked] = False
            if not not_spiked.any():
                #print(f"Early stopping at step {step}")
                for _ in range(step + 1, num_steps):
                    spk_out.append(torch.zeros_like(spk5))
                break
            
        return torch.stack(spk_out)