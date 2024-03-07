""" Half-hearted attempt a making a custom CNN.

Improvements are being worked on.
"""

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn, no_grad, as_tensor, Tensor


class CustomCNN(BaseFeaturesExtractor):
    """ Bare-bones attempt at using a custom CNN.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        print("="*80)
        print("== CustomCNN")
        print("="*80)

        print(n_input_channels)
        print(observation_space)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with no_grad():
            n_flatten = self.cnn(
                as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        exit(42)

    def forward(self, observations: Tensor) -> Tensor:
        """ Performs one computational step """
        return self.linear(self.cnn(observations))
