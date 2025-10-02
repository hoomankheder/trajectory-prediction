# Neural Network Integration with Domain Knowledge for High-Precision Trajectory Prediction

Developed the NNDK framework using a TCN for indoor human trajectory prediction from capacitive sensors. Achieved ~4.1 cm RMSE in-domain via kinematic features and Bayesian optimization; highlighted domain shift as key limitation (~13.2 cm RMSE on new data). Model optimized for efficiency (7,218 parameters) and prepared for INT8 deployment on ARM Cortex-M7-NN.

## Overview
- **Problem:** High-precision indoor trajectory prediction from noisy capacitive sensor data, focusing on time-series forecasting ((x, y)-to-(x, y)) using historical ground-truth.
- **Solution:** NNDK framework with modular preprocessing (outlier handling, kinematic features like velocity/acceleration), TCN architecture, and Bayesian hyperparameter tuning.
- **Key Results:** 
  - In-domain (CapEXP2): Average RMSE 4.1 cm ± 0.75 cm (6-fold permutation CV).
  - Generalization (CapEXP1): RMSE 13.2 cm ± 3.77 cm, identifying domain shift.
  - Efficiency: Lightweight TCN for edge deployment.
- **Tech Stack:** Python, TensorFlow/Keras, Scikit-learn (for scaling), NumPy/Pandas (data handling), Matplotlib (visualizations).

## Methodology Highlights
- **Data Collection:** 3m x 3m room with 4 capacitive sensors (3 Hz sampling) and ultrasound ground-truth.
- **Preprocessing:** Spike handling (threshold 0.5m), kinematic features (vx/vy/ax/ay), Min-Max normalization (shared for pairs like X/Y), windowing (seq_len=15).
- **Model:** TCN with dilated/causal convolutions, ReLU, dropout, batch norm. Optimized params: 16 filters, kernel=3, 3 hidden layers, dense=128, dropout=0.1, LR=0.01.
- **Training:** Adam optimizer, MSE loss, EarlyStopping/ReduceLROnPlateau callbacks. 6-fold permutation CV for stability.
- **Evaluation:** MSE/MAE/RMSE, visualizations (loss curves, Euclidean errors, true vs. pred plots)

## How to Run
1. Clone the repo: `git clone https://github.com/hoomankheder/trajectory-prediction.git`
2. Install dependencies: `pip install -r requirements.txt` (TensorFlow, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib).
3. Place your data in /data/ (use preprocessed-CapEXP2.csv as main; format: columns like 'cap1', 'cap2', 'cap3', 'cap4', 'X', 'Y').
4. Run the pipeline: `python code/run_pipeline.py` (trains/evaluates with 6-fold CV, saves results/plots).
5. For generalization test: Update GENERALIZATION_CSV_PATH in run_pipeline.py to your CapEXP1.csv.
6. Visualize: Check /results/plots/ for loss curves, error plots.

## Files
- **code/run_pipeline.py:** Main script for data loading, preprocessing, training, 6-fold CV, generalization test, and plotting.
- **code/preprocessing.py:** TrajectoryPreprocessor class (spike handling, kinematics, scaling, windowing).
- **code/nndk_core.py:** TCN model and training functions.
- **data/preprocessed-CapEXP2_sample.csv:** Placeholder CSV (columns: cap1-cap4, X, Y; add your real data).
- **results/**: Auto-generated (e.g., model_fold_1.keras, cv_summary.csv, plots like Training_vs_Validation_Loss_Fold1.png).
