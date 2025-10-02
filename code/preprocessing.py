import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class TrajectoryPreprocessor:
    def __init__(self, seq_len: int, target_cols: list):
        """Initializes the preprocessor."""
        self.seq_len = seq_len
        self.target_cols = target_cols
        self.pos_cols = ['X', 'Y']
        self.vel_cols = ['vx', 'vy']
        self.accel_cols = ['ax', 'ay']
        self.other_cols = ['cap1', 'cap2', 'cap3', 'cap4']
        self.feature_cols = self.other_cols + self.pos_cols + self.vel_cols + self.accel_cols
        self.scaler_other = MinMaxScaler()
        self.kinematic_params = {}

    def _add_kinematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates velocity and acceleration from X, Y columns."""
        df['vx'] = df['X'].diff().fillna(0)
        df['vy'] = df['Y'].diff().fillna(0)
        df['ax'] = df['vx'].diff().fillna(0)
        df['ay'] = df['vy'].diff().fillna(0)
        return df

    def _handle_spikes(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Identifies and corrects sharp spikes in positional data."""
        print(f"Applying threshold-based spike handling with threshold: {threshold}m")
        for col in ['X', 'Y']:
            diffs = df[col].diff().abs()
            spike_indices = diffs[diffs > threshold].index
            for idx in spike_indices:
                if idx > 0:
                    df.loc[idx, col] = df.loc[idx - 1, col]
        return df

    def fit_and_transform_training_data(self, df: pd.DataFrame, scaler_dir: str) -> np.ndarray:
        """
        Fits all scalers on the training data, saves them, and transforms the data.
        """
        print("Fitting shared Min-Max scalers on training data...")
        os.makedirs(scaler_dir, exist_ok=True)
        self.scaler_other.fit(df[self.other_cols])
        joblib.dump(self.scaler_other, os.path.join(scaler_dir, "other_features_scaler.pkl"))
        for key, cols in [('pos', self.pos_cols), ('vel', self.vel_cols), ('accel', self.accel_cols)]:
            group_data = df[cols].values
            self.kinematic_params[key] = {
                'min': group_data.min(),
                'max': group_data.max()
            }
        
        joblib.dump(self.kinematic_params, os.path.join(scaler_dir, "kinematic_params.pkl"))
        print(f"Scalers fitted and saved to {scaler_dir}")

        return self.transform_features(df)

    def load_scalers_and_transform(self, df: pd.DataFrame, scaler_dir: str) -> np.ndarray:
        """
        Loads pre-fitted scalers from disk and uses them to transform new data.
        """
        print(f"Loading scalers from {scaler_dir} and transforming data...")
        
        self.scaler_other = joblib.load(os.path.join(scaler_dir, "other_features_scaler.pkl"))
        self.kinematic_params = joblib.load(os.path.join(scaler_dir, "kinematic_params.pkl"))
        
        return self.transform_features(df)

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Helper function to apply all transformations."""
        scaled_other = self.scaler_other.transform(df[self.other_cols])
        
        scaled_pos = (df[self.pos_cols] - self.kinematic_params['pos']['min']) / \
                     (self.kinematic_params['pos']['max'] - self.kinematic_params['pos']['min'])
        
        scaled_vel = (df[self.vel_cols] - self.kinematic_params['vel']['min']) / \
                     (self.kinematic_params['vel']['max'] - self.kinematic_params['vel']['min'])
                     
        scaled_accel = (df[self.accel_cols] - self.kinematic_params['accel']['min']) / \
                       (self.kinematic_params['accel']['max'] - self.kinematic_params['accel']['min'])

        return np.concatenate([scaled_other, scaled_pos, scaled_vel, scaled_accel], axis=1)

    def _make_windows(self, feats: np.ndarray, targs: np.ndarray):
        """Creates overlapping windows from the time-series data."""
        N = len(feats)
        X, Y = [], []
        for i in range(N - self.seq_len):
            X.append(feats[i:i + self.seq_len])
            Y.append(targs[i + self.seq_len])
        return np.array(X), np.array(Y)

    def _make_target_windows(self, targs: np.ndarray):
        """Creates historical target windows."""
        N = len(targs)
        seqs = []
        for i in range(N - self.seq_len + 1):
            seqs.append(targs[i:i + self.seq_len])
        return np.array(seqs)
