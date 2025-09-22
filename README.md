# Weather forecasting with Graph Neural Networks (WeatherGNN)

WeatherGNN is a Python project for weather forecasting using Graph Neural Networks (GNNs). This repository contains code for data pre-processing, model definition, training, prediction, and visualization of results. 

## Features

- Data handling and preprocessing for weather datasets
- GNN-based model for spatial-temporal weather prediction
- Prediction utilities for running inference on new data
- Visualization tools for analyzing and plotting results

## Directory Structure

- `main.py` - Assemble graph structure, configure and implement training.
- `model.py` - Model architecture and related neural network code
- `Data.py` - Dataset and data loader definitions
- `predict.py` - Tools for running predictions and generating plots
- `utils.py` - Utility functions for training, evaluation, and plotting
- `requirements.txt` - List of required Python packages
- `preproc.joblib` - Preprocessing pipeline (serialized)
- 'stations.py' - Show station informatio and visualise GNN 
- `plots/` - Output directory for generated plots

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/ippocratiss/Weather_GNN.git
   cd Weather_GNN
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   *Additional dependencies may be required for some features (e.g., PyTorch for GNNs).*

## Usage
- **Get list of stations:**
  ```sh
  python stations.py
  ```

- **Training the model:**
  ```sh
  python main.py
  ```
  This will create:
  - `best_model.pt` (best model weights)
  - `preproc.joblib` (per-station scalers & metadata)

- **Running predictions:**
  ```sh
  python predict.py
  ```
  This will use the saved scalers and model to generate plots.
## Configuring Hyperparameters

To tune hyperparameters, edit the values directly in `main.py`. Key parameters include:

- `features` (list): Weather features to use, e.g., `['temp', 'pres']`
- `lookback`: Number of hours of history to use (affects input window)
- `pred_window`: Number of hours ahead to predict
- `trainFrac`: Fraction of data used for training (rest is validation)
- `start`, `end`: Date range for data
- `epochs`: Number of training epochs (e.g., `epochs = 300`)
- `batch_size`: Set in the DataLoader, e.g., `batch_size=10`
- `learning rate`: Set in the optimizer, e.g., `optimizer = torch.optim.AdamW(..., lr=2*10**-4, ...)`
- `nearest`: Number of nearest neighbors for graph construction (in `Connect_Graph`)

**Notes:**  
- The same preprocessing and feature configuration is loaded in `predict.py` via `preproc.joblib`. If you change the features, lookback, or pred_window, retrain the model and regenerate `preproc.joblib`.
- full list of features and details on api can be found at https://dev.meteostat.net/python/
- All hyperparameters listed above are found in main.py. 
## Plotting and Saving Paths

- **Model Checkpoints:**  
  The best model is saved as `best_model.pt` in the repository root whenever validation loss improves during training (`main.py`).

- **Preprocessing:**  
  The preprocessing metadata and scalers are saved as `preproc.joblib` in the root.

- **Training Plots:**  
  At the end of training, the function `Epoch_Plots` (from `utils.py`) is called, which displays the following plots:
    - Training Loss
    - Validation Loss
    - R² Score
    - Learning Rate
  These plots are displayed interactively using `matplotlib.pyplot.show()` and are **not** saved to disk by default.
  To save plots, you can modify `Epoch_Plots` in `utils.py` to call `plt.savefig('path/to/plot.png')` before `plt.show()`.

- **Prediction and Residual Plots:**  
  When running `predict.py`, plots are generated for the forecast and residuals for each station you specify. These also use `plt.show()` for interactive viewing and are not saved by default. You can add `plt.savefig('path/to/plot.png')` in `predict.py` after generating each plot to save figures to disk. 
  
**Notes:**
- To get predictions for a specific station you can add/remove names given in the `station_names` list located in predicit.py. 
- To make predictions for all stations uncomment the `all_stations` variable and add this into the following for loop. 
## Requirements

See `requirements.txt` for a complete list. Typical requirements include:
- Python 3.7+
- numpy
- pandas
- torch (PyTorch ≥2.0)
- torch-geometric (version matching your torch install)
- scikit-learn
- matplotlib
- meteostat
- geopy
- joblib

## License

This repository is a private fork. Refer to the upstream [B-Ruggles/Weather_GNN](https://github.com/B-Ruggles/Weather_GNN) for original licensing.

## Acknowledgements

Adapted from [B-Ruggles/Weather_GNN](https://github.com/B-Ruggles/Weather_GNN).

