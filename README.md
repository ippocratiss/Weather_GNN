# UK Weather Prediction (1D Convolution + GNN)

Forecast UK station features (e.g., temperature, pressure) for the next _N_ hours using per-station temporal Conv1d and a k-NN station graph with GATv2.

---

## 1) Install
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2) Train

python main.py

Creates:

    best_model.pt

    preproc.joblib (per-station scalers + metadata)

3) Predict

python predict.py

Uses the saved scalers and model to generate plots.
Configure (edit in main.py)

    features (e.g., ['temp', 'pres'])

    lookback (hours history) → time_step = lookback + 1

    pred_window (hours ahead)

    trainFrac (train split)

    date range (start, end)

    graph size: Connect_Graph(..., nearest=5)

    Keep the same values in predict.py. It loads preproc.joblib and does not refit scalers.

Project layout

main.py       # training pipeline
predict.py    # inference + plots
model.py      # model (Conv1d branches + GATv2)
data.py       # data fetch, scaling, graph build
utils.py      # train/eval, metrics, plotting
requirements.txt

Requirements

    Python 3.9–3.11

    PyTorch ≥ 2.0 (if you use padding='same', need ≥ 1.10)

    PyTorch Geometric (matching your PyTorch/CUDA)

    meteostat, geopy, scikit-learn, numpy, pandas, matplotlib, joblib

(Install via pip install -r requirements.txt.)
Notes

    No leakage: scalers are fit on the training slice and saved to preproc.joblib; inference only transforms.

    Model output shape: (nodes, pred_window, n_features).

You said:
why do you put the 3) predict like headers out of the boxes can you write one i can copy
ChatGPT said:

Gotcha — here’s a clean, minimal README you can copy-paste as is. The headers are outside the code blocks; only commands are inside.

# UK Weather Prediction (1D Convolution + GATv2)

Forecast UK station features (e.g., temperature, pressure) for the next _N_ hours using per-station Conv1d and a k-NN station graph with GATv2.

## 1) Install
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2) Train

python main.py

3) Predict

python predict.py

Project layout

main.py       # training pipeline
predict.py    # inference + plots
model.py      # model (Conv1d branches + GATv2)
data.py       # data fetch, scaling, graph build
utils.py      # train/eval, metrics, plotting
requirements.txt
