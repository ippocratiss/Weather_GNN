# UK Weather Prediction (0D Convolution + GNN)

Forecast UK temperature and pressure.
The model consists of 1d convolution layer that feeds in to graph neural network. 
The model collects data from the meteostat api, the number of stations availabl
is dependent on consistancy of reporting from each station. 

---

1) Install
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2) Train
```bash
python main.py
```
Creates:

    best_model.pt

    preproc.joblib (per-station scalers + metadata)

3) Predict
```bash 
python predict.py
```
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
