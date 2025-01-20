import torch
import json
from models.ipflstm import IPFLSTM
from utils.data_loader import create_dataloader
from utils.metrics import mean_squared_error

# Load configuration
with open('./experiment_config.json', 'r') as f:
    config = json.load(f)

device = config["general"]["device"]
time_window_lengths = config["experiment"]["time_window_lengths"]

# Custom GRU-based variant of IPFLSTM
class GRU_IPFLSTM(IPFLSTM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm = nn.GRU(input_size=kwargs["hidden_dim"], hidden_size=kwargs["lstm_dim"], num_layers=kwargs["lstm_layers"], batch_first=True)

# Initialize models
models = {
    "lstm": IPFLSTM(**config["model"]["ipflstm"]).to(device),
    "gru": GRU_IPFLSTM(**config["model"]["ipflstm"]).to(device)
}

# Experiment with varying time windows
results = {}
for time_window in time_window_lengths:
    results[time_window] = {}
    for name, model in models.items():
        # Simulate varying time window by slicing input data
        train_loader = create_dataloader(config["data"]["train_data_path"], config["data"]["batch_size"])
        loss_history = []

        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
        for epoch in range(config["training"]["epochs"]):
            model.train()
            for batch in train_loader:
                x, y, t, u, v, T = [data[:, :time_window].to(device) for data in batch]
                optimizer.zero_grad()
                output = model(x, y, t)
                loss = mean_squared_error(output, torch.cat((u, v), dim=1))
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

        results[time_window][name] = loss_history

# Save results
with open('./results/time_window_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=4)