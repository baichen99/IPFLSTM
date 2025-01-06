# IPFLSTM: IPFLSTM: Enhancing Physics-Informed Neural Networks with LSTM and Informer for Efficient Long-Term Prediction of Dynamic Physical Fields

This repository contains the code and resources for **IPFLSTM**, a novel framework combining data-driven Long Short-Term Memory (LSTM) networks with physics-based constraints to predict dynamic physical quantities in middle-to-low-speed maglev transportation systems.

## Overview
IPFLSTM is designed to address challenges in time-series prediction of critical physical quantities that are difficult to model using traditional methods. The framework:

1. Incorporates domain knowledge through physics-based constraints to guide learning.
2. Utilizes LSTM networks for capturing temporal dependencies in complex systems.
3. Achieves high accuracy in predicting physical quantities such as suspension current, gap, and acceleration.

## Key Features
- **Hybrid Framework**: Combines data-driven methods with physical modeling to improve prediction reliability.
- **Generalizability**: Can be adapted to various dynamic systems beyond maglev transportation.
- **Enhanced Interpretability**: Integrates physical constraints for better alignment with real-world system behaviors.

## Use Cases
The framework is particularly useful for:
- Evaluating the deformation of electromagnets.
- Monitoring the aging of rubber components like primary springs and bearings.
- Identifying anomalous states, such as pole short circuits.
- Predicting time-series behavior of physical quantities for system optimization.

## Installation
To use this repository, clone it and install the required dependencies:

```bash
git clone https://github.com/baichen/IPFLSTM.git
cd IPFLSTM
pip install -r requirements.txt
```

## Repository Structure
```plaintext
IPFLSTM/
├── data/                 # Dataset files
├── models/               # LSTM and physics-based models
├── notebooks/            # Jupyter notebooks for analysis and visualization
├── scripts/              # Training and evaluation scripts
├── results/              # Output results and logs
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

## Usage
### Training the Model
Use the `train.py` script to train the IPFLSTM model on your dataset:

```bash
python scripts/train.py --config configs/config.yaml
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any feature requests, bug fixes, or improvements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to explore, use, and extend IPFLSTM for your dynamic system prediction tasks!
