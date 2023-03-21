import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import device, device_name


class MultiLayerCNN(nn.Module):

    def __init__(self, obs_input_size, img_input_width, img_input_height, output_size):
        super().__init__()
        self.OBS_SIZE = obs_input_size
        self.IMG_WIDTH = img_input_width
        self.IMG_HEIGHT = img_input_height
        self.DENSE_OUTPUT = 128
        self.CNN_OUTPUT_SIZE = 243
        self.OUTPUT_SIZE = output_size

        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=27, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=27, out_channels=81, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=81, out_channels=self.CNN_OUTPUT_SIZE, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.cnn_output_width = int((self.IMG_WIDTH - 3 + 1) / 2)
        self.cnn_output_width = int(np.floor((self.cnn_output_width - 3 + 1) / 2))
        self.cnn_output_width = int(np.floor((self.cnn_output_width - 3 + 1) / 2))
        self.cnn_output_height = int((self.IMG_HEIGHT - 3 + 1) / 2)
        self.cnn_output_height = int(np.floor((self.cnn_output_height - 3 + 1) / 2))
        self.cnn_output_height = int(np.floor((self.cnn_output_height - 3 + 1) / 2))

        self.dense_stack = nn.Sequential(
            nn.Linear(in_features=self.OBS_SIZE, out_features=self.DENSE_OUTPUT),
            nn.ReLU(),
            nn.Linear(in_features=self.DENSE_OUTPUT, out_features=self.DENSE_OUTPUT),
            nn.ReLU(),
            nn.Linear(in_features=self.DENSE_OUTPUT, out_features=self.DENSE_OUTPUT),
            nn.ReLU()
        )
        self.combined_stack = nn.Sequential(
            nn.Linear(
                in_features=(self.cnn_output_height * self.cnn_output_width * self.CNN_OUTPUT_SIZE) + self.DENSE_OUTPUT,
                out_features=self.DENSE_OUTPUT
            ),
            nn.ReLU(),
            nn.Linear(in_features=self.DENSE_OUTPUT, out_features=self.OUTPUT_SIZE)
        )

    def forward(self, x):
        obs = x[:,:self.OBS_SIZE]
        img = x[:,self.OBS_SIZE:].reshape(-1, self.IMG_HEIGHT, self.IMG_WIDTH, 3)
        img = img.permute(0, 3, 1, 2)
        img_output = self.cnn_stack(img)
        img_output = torch.flatten(input=img_output, start_dim=1)
        obs_output = self.dense_stack(obs)
        output = torch.cat([img_output, obs_output], 1)
        output = self.combined_stack(output)
        return output

class MultiLayerCNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, obs_input_size, img_input_width, img_input_height, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)
        self.OBS_SIZE = obs_input_size
        self.IMG_WIDTH = img_input_width
        self.IMG_HEIGHT = img_input_height
        self.DENSE_OUTPUT = features_dim
        self.CNN_OUTPUT_SIZE = 81

        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=27, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=27, out_channels=self.CNN_OUTPUT_SIZE, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.cnn_output_width = int((self.IMG_WIDTH - 3 + 1) / 2)
        self.cnn_output_width = int(np.floor((self.cnn_output_width - 3 + 1) / 2))
        self.cnn_output_height = int((self.IMG_HEIGHT - 3 + 1) / 2)
        self.cnn_output_height = int(np.floor((self.cnn_output_height - 3 + 1) / 2))

        self.dense_stack = nn.Sequential(
            nn.Linear(in_features=self.OBS_SIZE, out_features=self.DENSE_OUTPUT),
            nn.ReLU()
        )
        self.combined_stack = nn.Sequential(
            nn.Linear(
                in_features=(self.cnn_output_height * self.cnn_output_width * self.CNN_OUTPUT_SIZE) + self.DENSE_OUTPUT,
                out_features=self.DENSE_OUTPUT
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.to(device)
        obs = x[:,:self.OBS_SIZE]
        img = x[:,self.OBS_SIZE:].reshape(-1, self.IMG_HEIGHT, self.IMG_WIDTH, 3)
        img = img.permute(0, 3, 1, 2)
        img_output = self.cnn_stack(img)
        img_output = torch.flatten(input=img_output, start_dim=1)
        obs_output = self.dense_stack(obs)
        output = torch.cat([img_output, obs_output], 1)
        output = self.combined_stack(output)
        return output

class MultiLayerCNNPolicy(nn.Module):
    def __init__(self, observation_space: gym.Space, obs_input_size, img_input_width, img_input_height, features_dim=128):
        super().__init__()
        self.OBS_SIZE = obs_input_size
        self.IMG_WIDTH = img_input_width
        self.IMG_HEIGHT = img_input_height
        self.DENSE_OUTPUT = features_dim
        self.CNN_OUTPUT_SIZE = 81

        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=27, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=27, out_channels=self.CNN_OUTPUT_SIZE, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.cnn_output_width = int((self.IMG_WIDTH - 3 + 1) / 2)
        self.cnn_output_width = int(np.floor((self.cnn_output_width - 3 + 1) / 2))
        self.cnn_output_height = int((self.IMG_HEIGHT - 3 + 1) / 2)
        self.cnn_output_height = int(np.floor((self.cnn_output_height - 3 + 1) / 2))

        self.dense_stack = nn.Sequential(
            nn.Linear(in_features=self.OBS_SIZE, out_features=self.DENSE_OUTPUT),
            nn.ReLU()
        )
        self.combined_stack = nn.Sequential(
            nn.Linear(
                in_features=(self.cnn_output_height * self.cnn_output_width * self.CNN_OUTPUT_SIZE) + self.DENSE_OUTPUT,
                out_features=self.DENSE_OUTPUT
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.to(device)
        obs = x[:,:self.OBS_SIZE]
        img = x[:,self.OBS_SIZE:].reshape(-1, self.IMG_HEIGHT, self.IMG_WIDTH, 3)
        img = img.permute(0, 3, 1, 2)
        img_output = self.cnn_stack(img)
        img_output = torch.flatten(input=img_output, start_dim=1)
        obs_output = self.dense_stack(obs)
        output = torch.cat([img_output, obs_output], 1)
        output = self.combined_stack(output)
        return output

class NetworkBC(nn.Module):
    """ Network for behavioral cloning. """
    
    def __init__(self, obs_input_size, output_size):
        super().__init__()
        self.OBS_SIZE = obs_input_size
        self.DENSE_OUTPUT = 32
        self.OUTPUT_SIZE = output_size

        self.dense_stack = nn.Sequential(
            nn.Linear(
                    in_features=self.OBS_SIZE,
                    out_features=self.DENSE_OUTPUT
            ),
            nn.ReLU(),
            nn.Linear(
                    in_features=self.DENSE_OUTPUT,
                    out_features=self.OUTPUT_SIZE
            )
        )

    def forward(self, x):
        return self.dense_stack(x)


class NetworkBC2(nn.Module):
    """ Network for behavioral cloning place subtask. """
    
    def __init__(self, obs_input_size, output_size):
        super().__init__()
        self.OBS_SIZE = obs_input_size
        self.DENSE_OUTPUT = 32
        self.OUTPUT_SIZE = output_size

        self.dense_stack = nn.Sequential(
            nn.Linear(
                    in_features=self.OBS_SIZE,
                    out_features=self.DENSE_OUTPUT
            ),
            nn.ReLU(),
            ##nn.Linear(
            ##        in_features=self.DENSE_OUTPUT,
            ##        out_features=self.DENSE_OUTPUT
            ##),
            ##nn.ReLU(),
            nn.Linear(
                    in_features=self.DENSE_OUTPUT,
                    out_features=self.OUTPUT_SIZE
            )
        )

    def forward(self, x):
        linear_out = self.dense_stack(x) # shape is (batches, features)
        if len(linear_out.shape) > 1:
            return torch.cat((linear_out[:,0:3], torch.tanh(linear_out[:,3:])), dim=1) # concatenate over feature dimension
        else:
            return torch.cat((linear_out[0:3], torch.tanh(linear_out[3:])))



def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
