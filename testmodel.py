import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TrajDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# 1. Simple MLP Model
class MLPTrajectoryPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=320):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# 2. Transformer-based Model
class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=320, nhead=4):
        super().__init__()
        self.input_embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers=3
        )
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.input_embedding(x).unsqueeze(1)
        x = x + self.positional_encoding
        x = self.transformer(x)
        return self.output_layer(x.squeeze(1))

# 3. VAE for handling limited data
class VAETrajectoryPredictor(nn.Module):
    def __init__(self, input_size=6, latent_size=32, hidden_size=128, output_size=320):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def train_vae(model, train_loader, num_epochs, device, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def loss_function(recon_x, x, mu, logvar):
        MSE = nn.MSELoss(reduction='sum')(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, target, mu, logvar)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')

def train_model(model, train_loader, num_epochs, device, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')

# Data augmentation techniques for limited data
def augment_trajectory_data(inputs, outputs, num_augmented=1000):
    augmented_inputs = []
    augmented_outputs = []
    
    for _ in range(num_augmented):
        # Randomly select a sample
        idx = np.random.randint(0, len(inputs))
        input_sample = inputs[idx].copy()
        output_sample = outputs[idx].copy()
        
        # Add small random perturbations
        input_noise = np.random.normal(0, 0.02, size=input_sample.shape)
        output_noise = np.random.normal(0, 0.02, size=output_sample.shape)
        
        augmented_inputs.append(input_sample + input_noise)
        augmented_outputs.append(output_sample + output_noise)
    
    return np.vstack(augmented_inputs), np.vstack(augmented_outputs)

# Example usage
def main():
    # Assuming we have some sample data
    # Replace with actual data loading
    num_samples = 100
    inputs = np.random.randn(num_samples, 6)  # 2 sets of XYZ coordinates
    outputs = np.random.randn(num_samples, 320)  # 80 points * 4 (XYZC)
    
    # Augment data
    aug_inputs, aug_outputs = augment_trajectory_data(inputs, outputs)
    inputs = np.vstack([inputs, aug_inputs])
    outputs = np.vstack([outputs, aug_outputs])
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    # Create data loaders
    train_dataset = TrajDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mlp_model = MLPTrajectoryPredictor().to(device)
    transformer_model = TransformerTrajectoryPredictor().to(device)
    vae_model = VAETrajectoryPredictor().to(device)
    
    # Train models
    print("Training MLP model...")
    train_model(mlp_model, train_loader, num_epochs=50, device=device)
    
    print("\nTraining Transformer model...")
    train_model(transformer_model, train_loader, num_epochs=50, device=device)
    
    print("\nTraining VAE model...")
    train_vae(vae_model, train_loader, num_epochs=50, device=device)

if __name__ == "__main__":
    main()