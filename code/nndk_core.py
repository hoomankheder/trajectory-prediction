import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tcn_simple import TCN_model      
from tcn_model  import model_TCN_simple 
from utils import compute_test

def train_model(
    X_train, Y_train, Y_train_seq,
    X_val,   Y_val,   Y_val_seq,
    X_test,  Y_test,  Y_test_seq,
    save_path: str = "NNdk_TCN_model.keras",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden: int = 6,
    num_filters: int = 64,
    k_size: int = 5,
    dense_units: int = 128,
    dropout_rate: float = 0.2 
):
    """
    Train a TCN model (NNdk) to predict the next (X,Y) from a history of past positions.

    Input:
      - X_train, X_val, X_test: shape (n_windows, seq_len, 2) past (X,Y)
      - Y_train, Y_val, Y_test: shape (n_windows, 2) next-step targets
      - Y_*_seq: shape (n_windows, seq_len, 2) full history windows

    Returns:
      - history: Keras History object
      - test_metrics: [test_loss (MSE), test_mae]
    """
    seq_len = X_train.shape[1]
    feature_dim = X_train.shape[2]
    target_dim = Y_train.shape[1]

    print(f"Building TCN with input ({seq_len},{feature_dim}) -> output {target_dim}")
    model = model_TCN_simple(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden=hidden,
        num_filters=num_filters,
        k_size=k_size,
        dense_units=dense_units,
        output_dim=target_dim,
        dropout_rate=dropout_rate
    )
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    model.summary()
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min') 
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, mode='min') 

    print(f"Starting training: epochs={epochs}, batch_size={batch_size}")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlp]
    )
    print("Evaluating on test data...")
    test_metrics = model.evaluate(X_test, Y_test, verbose=0)

    print(f"Test MSE: {test_metrics[0]:.4f}, Test MAE: {test_metrics[1]:.4f}, Test RMSE: {test_metrics[2]:.4f}")
    if save_path:
        save_dir = os.path.dirname(save_path) 
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving model to {save_path}...")
        model.save(save_path)
    compute_test(
            model,
            "NNdk_TCN",
            X_train=X_train, Y_train=Y_train,
            X_val=X_val,     Y_val=Y_val,
            X_test=X_test,   Y_test=Y_test,
            path=save_dir,
            trnable_params=model.count_params(),
            nb_filters=num_filters,
            kernel_size=k_size,
            nb_stacks=hidden,
            dense=dense_units,
            hidden=hidden
    )
    return history, test_metrics

def load_trained_model(model_path: str = "NNdk_TCN_model.keras"):
    print(f"Loading model from {model_path}...")
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import custom_object_scope
    from tcn_simple import TCN_model 
    with custom_object_scope({
        'TCN_model': TCN_model,
    }):
        model = load_model(model_path, compile=False)
    model.compile(loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    print("Model loaded.")
    return model
