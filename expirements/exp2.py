import torch
import json
from models.ipflstm import IPFLSTM
from utils.data_loader import create_dataloader
from utils.metrics import mean_squared_error
from utils.visualization import plot_loss_curve

# Load configuration
with open('./experiment_config.json', 'r') as f:
    config = json.load(f)

device = config["general"]["device"]

# Initialize data loader
train_loader = create_dataloader(config["data"]["train_data_path"], config["data"]["batch_size"])

# Define custom gate variants for LSTM
class NoForgetLSTM(IPFLSTM):
    def forward(self, x, y, t):
        # Modify forget gate logic here (disable forget gate)
        return super().forward(x, y, t)

class NoInputLSTM(IPFLSTM):
    def forward(self, x, y, t):
        # Modify input gate logic here (disable input gate)
        return super().forward(x, y, t)

class NoOutputLSTM(IPFLSTM):
    def forward(self, x, y, t):
        # Modify output gate logic here (disable output gate)
        return super().forward(x, y, t)

gate_variants = {
    "ipflstm": IPFLSTM(**config["model"]["ipflstm"]).to(device),
    "no_forget_lstm": NoForgetLSTM(**config["model"]["ipflstm"]).to(device),
    "no_input_lstm": NoInputLSTM(**config["model"]["ipflstm"]).to(device),
    "no_output_lstm": NoOutputLSTM(**config["model"]["ipflstm"]).to(device)
}

# Training and comparison
results = {}
for name, model in gate_variants.items():
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

    results[name] = {"loss_history": loss_history}
    plot_loss_curve(loss_history, title=f"{name} Loss Curve")

# Save results
with open('./results/ablation_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=4)