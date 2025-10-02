import os
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import KFold
from preprocessing import TrajectoryPreprocessor
from nndk_core import train_model, load_trained_model

# --- GLOBAL CONFIGURATION ---
CSV_PATH      = r"../../datasets/preprocessed-CapEXP2.csv"
GENERALIZATION_CSV_PATH = r"../../datasets/preprocessed-CapEXP1.csv"
SEQ_LEN       = 15
TARGET_COLS   = ['X', 'Y']
EPOCHS        = 50
BATCH_SIZE    = 8
# --- Hyperparameters from tuning ---
LR            = 0.01
NUM_FILTERS   = 16
HIDDEN_LAYERS = 3
K_SIZE        = 3
DENSE_UNITS   = 128
DROPOUT_RATE  = 0.1
# -----------------------------------
RESULTS_DIR = 'results_final'
os.makedirs(RESULTS_DIR, exist_ok=True)
FEATURE_SCALER_DIR = RESULTS_DIR
TARGET_SCALER_PATH = os.path.join(RESULTS_DIR, "target_scaler.pkl")

# --- PLOTTING FUNCTION ---
def generate_and_save_plots(history, X_data, Y_true_raw, model, split_name, plot_output_dir, scaler_y):
    os.makedirs(plot_output_dir, exist_ok=True)
    print(f"Generating plots for {split_name} in {plot_output_dir}...")
    Y_pred_scaled = model.predict(X_data)
    Y_pred_orig = scaler_y.inverse_transform(Y_pred_scaled)
    if history:
        plt.figure(figsize=(10, 6))
        
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plot_start_epoch = 10
        if len(train_loss) >= plot_start_epoch:
            epochs_range = range(plot_start_epoch, len(train_loss) + 1)
            plt.plot(epochs_range, train_loss[plot_start_epoch-1:], label='Training Loss (MSE)')
            plt.plot(epochs_range, val_loss[plot_start_epoch-1:], label='Validation Loss (MSE)')
            plt.title(f'Training vs. Validation Loss ({split_name}) (from Epoch {plot_start_epoch})')
        else: 
            epochs_range = range(1, len(train_loss) + 1)
            plt.plot(epochs_range, train_loss, label='Training Loss (MSE)')
            plt.plot(epochs_range, val_loss, label='Validation Loss (MSE)')
            plt.title(f'Training vs. Validation Loss ({split_name})')

        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_output_dir, f'Loss_vs_Epoch_{split_name}.png'))
        plt.close()
        
    error_x = Y_pred_orig[:, 0] - Y_true_raw[:, 0]
    error_y = Y_pred_orig[:, 1] - Y_true_raw[:, 1]
    euclidean_dist = np.sqrt(error_x**2 + error_y**2)

    plt.figure(figsize=(12, 6))
    plt.plot(euclidean_dist, marker='o', linestyle='-', markersize=2, alpha=0.7)
    plt.title(f'Euclidean Distance Error vs. Time ({split_name})')
    plt.xlabel('Sample Index'); plt.ylabel('Euclidean Distance (m)'); plt.grid(True)
    plt.savefig(os.path.join(plot_output_dir, f'Euclidean_Distance_{split_name}.png'))
    plt.close()

    # Plot 2: True & Predicted Trajectory with Signed Error
    error_x_signed = Y_pred_orig[:, 0] - Y_true_raw[:, 0]
    error_y_signed = Y_pred_orig[:, 1] - Y_true_raw[:, 1]
    plt.figure(figsize=(16, 9))

    # Subplot for X
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(np.arange(len(Y_true_raw)), Y_true_raw[:, 0], label='True X', alpha=0.8, color='tab:blue')
    ax1.plot(np.arange(len(Y_true_raw)), Y_pred_orig[:, 0], label='Predicted X', linestyle='--', alpha=0.8, color='tab:orange')
    ax1.set_ylabel('X Position (m)', color='tab:blue')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(np.arange(len(Y_true_raw)), error_x_signed, label='Error X (Pred - True)', alpha=0.6, color='tab:red', linestyle='-.')
    ax1_twin.set_ylabel('Error X (m)', color='tab:red')
    ax1_twin.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax1_twin.legend(loc='upper right')
    ax1.set_title(f'True vs. Predicted X Position & Error Over Time ({split_name})')

    # Subplot for Y
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(np.arange(len(Y_true_raw)), Y_true_raw[:, 1], label='True Y', alpha=0.8, color='tab:blue')
    ax2.plot(np.arange(len(Y_true_raw)), Y_pred_orig[:, 1], label='Predicted Y', linestyle='--', alpha=0.8, color='tab:orange')
    ax2.set_xlabel('Sample Index (Time)')
    ax2.set_ylabel('Y Position (m)', color='tab:blue')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(np.arange(len(Y_true_raw)), error_y_signed, label='Error Y (Pred - True)', alpha=0.6, color='tab:red', linestyle='-.')
    ax2_twin.set_ylabel('Error Y (m)', color='tab:red')
    ax2_twin.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax2_twin.legend(loc='upper right')
    ax2.set_title(f'True vs. Predicted Y Position & Error Over Time ({split_name})')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f'Combined_XY_Error_vs_Time_{split_name}.png'))
    plt.close()

# --- GENERALIZATION TEST FUNCTION ---
def run_generalization_test(new_csv_path: str, trained_model_path: str, target_scaler_path: str, feature_scaler_dir: str, experiment_name: str):
    print(f"\n--- Starting Generalization Test on: {os.path.basename(new_csv_path)} ---")
    preprocessor = TrajectoryPreprocessor(seq_len=SEQ_LEN, target_cols=TARGET_COLS)
    df_gen = pd.read_csv(new_csv_path)
    df_gen = preprocessor._handle_spikes(df_gen)
    df_gen = preprocessor._add_kinematic_features(df_gen)
    
    features_normalized_gen = preprocessor.load_scalers_and_transform(df_gen, feature_scaler_dir)
    targets_raw_gen = df_gen[TARGET_COLS].to_numpy(dtype=np.float32)
    X_gen, Y_gen_raw = preprocessor._make_windows(features_normalized_gen, targets_raw_gen)

    model_for_test = load_trained_model(trained_model_path)
    scaler_y_loaded = joblib.load(target_scaler_path)
    
    Y_gen_pred_scaled = model_for_test.predict(X_gen)
    Y_gen_pred_orig = scaler_y_loaded.inverse_transform(Y_gen_pred_scaled)

    gen_mse = mean_squared_error(Y_gen_raw, Y_gen_pred_orig)
    gen_mae = mean_absolute_error(Y_gen_raw, Y_gen_pred_orig)
    gen_rmse = np.sqrt(gen_mse)
    plot_output_dir = os.path.join(RESULTS_DIR, "plots", "generalization_tests")
    generate_and_save_plots(None, X_gen, Y_gen_raw, model_for_test, f"GenTest_on_model_{experiment_name}", plot_output_dir, scaler_y_loaded)   
    return {'MSE': gen_mse, 'RMSE': gen_rmse, 'MAE': gen_mae}


if __name__ == '__main__':
    # 1. DATA LOADING AND PREPARATION
    print("--- Loading and Preparing Full Dataset ---")
    preprocessor = TrajectoryPreprocessor(seq_len=SEQ_LEN, target_cols=TARGET_COLS)
    df_full = pd.read_csv(CSV_PATH)
    df_full_processed = preprocessor._handle_spikes(df_full)
    df_full_processed = preprocessor._add_kinematic_features(df_full_processed)
    features_normalized_full = preprocessor.fit_and_transform_training_data(df_full_processed, FEATURE_SCALER_DIR)

    scaler_y = MinMaxScaler()
    targets_raw_full = df_full_processed[TARGET_COLS].to_numpy(dtype=np.float32)
    targets_scaled_full = scaler_y.fit_transform(targets_raw_full)
    joblib.dump(scaler_y, TARGET_SCALER_PATH)

    X_full, Y_full_scaled = preprocessor._make_windows(features_normalized_full, targets_scaled_full)
    _, Y_full_raw = preprocessor._make_windows(features_normalized_full, targets_raw_full)

 # 2. SETUP FOR CUSTOM 6-FOLD PERMUTATION CROSS VALIDAITOM
    n_samples = len(X_full)
    cv_results = []
    gen_results = []
    fold_definitions = {
        1: {'name': 'Train(0-60)_Val(60-80)',   'train': (0.0, 0.6), 'val': (0.6, 0.8)},
        2: {'name': 'Train(0-60)_Val(80-100)',  'train': (0.0, 0.6), 'val': (0.8, 1.0)},
        3: {'name': 'Val(0-20)_Train(20-80)',   'train': (0.2, 0.8), 'val': (0.0, 0.2)},
        4: {'name': 'Val(0-20)_Train(40-100)',  'train': (0.4, 1.0), 'val': (0.0, 0.2)},
        5: {'name': 'Train(20-80)_Val(80-100)', 'train': (0.2, 0.8), 'val': (0.8, 1.0)},
        6: {'name': 'Train(40-100)_Val(20-40)', 'train': (0.4, 1.0), 'val': (0.2, 0.4)},
    }

    print(f"\n--- Starting Custom 6-Fold Permutation Cross-Validation ---")
    for fold_num, fold_info in fold_definitions.items():
        print(f"\n--- Processing Fold {fold_num}/6: {fold_info['name']} ---")
        
        tr_start_idx = int(fold_info['train'][0] * n_samples)
        tr_end_idx = int(fold_info['train'][1] * n_samples)
        val_start_idx = int(fold_info['val'][0] * n_samples)
        val_end_idx = int(fold_info['val'][1] * n_samples)
        X_train_fold, Y_train_fold = X_full[tr_start_idx:tr_end_idx], Y_full_scaled[tr_start_idx:tr_end_idx]
        X_val_fold, Y_val_fold = X_full[val_start_idx:val_end_idx], Y_full_scaled[val_start_idx:val_end_idx]
        Y_val_raw_fold = Y_full_raw[val_start_idx:val_end_idx] 

        model_path = os.path.join(RESULTS_DIR, f"model_fold_{fold_num}.keras")
        history, val_metrics_list = train_model(
            X_train_fold, Y_train_fold, [],
            X_val_fold, Y_val_fold, [],
            X_val_fold, Y_val_fold, [],
            save_path=model_path,
            epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
            hidden=HIDDEN_LAYERS, num_filters=NUM_FILTERS,
            k_size=K_SIZE, dense_units=DENSE_UNITS,
            dropout_rate=DROPOUT_RATE
        )
        cv_results.append({
            'val_MSE': val_metrics_list[0], 'val_MAE': val_metrics_list[1], 'val_RMSE': val_metrics_list[2]
        })
        
        # Generate plots for this fold's validation set
        model = load_trained_model(model_path)
        plot_output_dir = os.path.join(RESULTS_DIR, "plots", f"fold_{fold_num}")
        generate_and_save_plots(history, X_val_fold, Y_val_raw_fold, model, f"Fold_{fold_num}_{fold_info['name']}", plot_output_dir, scaler_y)
        if os.path.exists(GENERALIZATION_CSV_PATH):
            gen_metrics = run_generalization_test(
                new_csv_path=GENERALIZATION_CSV_PATH, trained_model_path=model_path,
                target_scaler_path=TARGET_SCALER_PATH, feature_scaler_dir=FEATURE_SCALER_DIR,
                experiment_name=fold_info['name']
            )
            gen_results.append(gen_metrics)

    # 4. CREATE AND PRINT THE FINAL SUMMARY TABLES
    print("\n\n--- Cross-Validation Performance Summary ---")
    cv_df = pd.DataFrame(cv_results)
    cv_summary = pd.DataFrame({
        "Mean": cv_df.mean(),
        "STD": cv_df.std()
    }).T
    cv_summary.index.name = "Statistic"
    print("The table below shows the average performance across the 6 validation permutations.")
    print(cv_summary.to_string())
    cv_summary.to_csv(os.path.join(RESULTS_DIR, "permutation_cv_summary.csv"))

    if gen_results:
        print("\n\n--- Generalization Performance Summary ---")
        gen_df = pd.DataFrame(gen_results)
        gen_summary = pd.DataFrame({
            "Mean": gen_df.mean(),
            "STD": gen_df.std()
        }).T
        gen_summary.index.name = "Statistic"
        print("The table below shows the average generalization performance across all 6 models.")
        print(gen_summary.to_string())
        gen_summary.to_csv(os.path.join(RESULTS_DIR, "generalization_summary.csv"))
