import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=4, encoding_dim=2):  # Now expects 4 features
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Returns encoded features for clustering."""
        return self.encoder(x)

def load_trained_autoencoder(model_path, input_dim=4, encoding_dim=2):
    """Loads a trained autoencoder model."""
    model = Autoencoder(input_dim, encoding_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model
