from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizacja danych MNIST
])

train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
    )
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
    )

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Teraz możesz użyć train_loader i test_loader do treningu i testowania modelu.


class DiffusionModel(nn.Module):
    def __init__(self, input_dim):
        super(DiffusionModel, self).__init__()
        # Prosty model MLP dla demonstracji
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, noise_level):
        # Dodaj szum do danych wejściowych
        noisy_x = x + noise_level * torch.randn_like(x)
        # Przepuść przez model
        return self.model(noisy_x)


def train(model, data_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        loss_history = []
        for _, (data, _) in enumerate(data_loader):
            data = data.view(data.size(0), -1).to(device)
            noise_level = np.random.uniform(0, 1)  # Losowy poziom szumu
            noisy_data = data + noise_level * torch.randn_like(data).to(device)
            optimizer.zero_grad()
            output = model(noisy_data, noise_level)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        loss_history = np.array(loss_history)
        torch.save(model, f"model_{epoch}.pth")

        print(f"Epoch {epoch+1}, Loss {np.mean(loss_history)}")


def generate_image(model, start_noise_level=1.0, steps=10, input_dim=784):
    # Zacznij od całkowicie losowego szumu
    noise = torch.randn(1, input_dim).to(device)
    current_noise_level = start_noise_level

    model.eval()
    with torch.no_grad():
        for step in range(steps):
            # Stopniowo zmniejszaj poziom szumu
            noise_level = current_noise_level * (steps - step - 1) / steps
            # Proś model o 'odszumienie' obrazu
            noise = model(noise, noise_level)

    generated_image = noise.view(28, 28).cpu().numpy()

    return generated_image


model = DiffusionModel(input_dim=784).to(device)
train(model, train_loader, 50)

generated_image = generate_image(model)

# Wyświetl wygenerowany obraz

plt.imshow(generated_image, cmap='gray')
plt.imsave("test.png", generated_image)
plt.show()
