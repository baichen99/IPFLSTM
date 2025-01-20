import torch
import json
from models.pinn import PINNs
from models.pinnsformer import PINNsFormer
from models.lstm_pinnsformer import LSTM_PINNsFormer
from models.ipflstm import IPFLSTM
from utils.data_loader import create_dataloader
from utils.metrics import mean_squared_error
from utils.visualization import plot_loss_curve

# Load configuration
with open('./experiment_config.json', 'r') as f:
    config = json.load(f)

device = config["general"]["device"]
models_to_compare = config["experiment"]["models_to_compare"]

# Initialize data loader
train_loader = create_dataloader(config["data"]["train_data_path"], config["data"]["batch_size"])
test_loader = create_dataloader(config["data"]["test_data_path"], config["data"]["batch_size"], shuffle=False)

# Initialize models
models = {
    "pinn": PINNs(**config["model"]["pinn"]).to(device),
    "pinnsformer": PINNsFormer(**config["model"]["pinnsformer"]).to(device),
    "lstm_pinnsformer": LSTM_PINNsFormer(**config["model"]["lstm_pinnsformer"]).to(device),
    "ipflstm": IPFLSTM(**config["model"]["ipflstm"]).to(device)
}

# Training loop
results = {}
for model_name in models_to_compare:
    model = models[model_name]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_history = []

    for epoch in range(config["training"]["epochs"]):
        model.train()
        for batch in train_loader:
            x, y, t, u, v, T = [data.to(device) for data in batch]
            optimizer.zero_grad()
            output = model(x, y, t)
            loss = mean_squared_error(output, torch.cat((u, v), dim=1))
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

    # Evaluate on test set
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x, y, t, u, v, T = [data.to(device) for data in batch]
            output = model(x, y, t)
            loss = mean_squared_error(output, torch.cat((u, v), dim=1))
            test_losses.append(loss.item())

    results[model_name] = {
        "train_loss": sum(loss_history) / len(loss_history),
        "test_loss": sum(test_losses) / len(test_losses),
        "loss_history": loss_history
    }

    # Plot loss curve
    plot_loss_curve(loss_history, title=f"{model_name} Loss Curve")

# Save results
with open('./results/comparative_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=4)