import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

class AutoencoderGMMDetector(BaseEstimator, TransformerMixin):
    """
    Generic autoencoder + GMM anomaly detector for tabular data.
    """
    def __init__(self, code_dim=2, epochs=25, batch_size=256, n_components=2,
                 learning_rate=0.001, anomaly_percentile=5):
        self.code_dim = code_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.anomaly_percentile = anomaly_percentile
        
        self.autoencoder = None
        self.gmm = None
        self.threshold = None
        self._history = None
    
    def build_model(self, input_shape):
        inp = Input(shape=(input_shape,))
        x = Dense(64, activation='relu')(inp) 
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        code = Dense(self.code_dim, activation='relu')(x)
        x = Dense(16, activation='relu')(code)
        x = Dense(32, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(input_shape, activation='linear')(x)
        return Model(inp, out)
    
    def fit(self, X, y=None):
        input_shape = X.shape[1]
        self.autoencoder = self.build_model(input_shape)
        self.autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

        self._history = self.autoencoder.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.25,
            callbacks=[earlystop],
            shuffle=True,
            verbose=1
        )

        # Compute reconstruction errors
        recon = self.autoencoder.predict(X, batch_size=self.batch_size, verbose=0)
        errors = np.mean(np.abs(recon - X), axis=1).reshape(-1,1)

        # Fit GMM
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        self.gmm.fit(errors)

        # Threshold = bottom percentile of log-likelihood
        log_likelihood = self.gmm.score_samples(errors)
        self.threshold = np.percentile(log_likelihood, self.anomaly_percentile)
        print(f"Anomaly log-likelihood threshold set at {self.threshold:.6f} "
              f"(bottom {self.anomaly_percentile} percentile)")

        return self

    @property
    def history(self):
        return self._history

    def score_samples(self, X):
        recon = self.autoencoder.predict(X, batch_size=self.batch_size, verbose=0)
        errors = np.mean(np.abs(recon - X), axis=1).reshape(-1,1)
        return self.gmm.score_samples(errors)

    def predict(self, X):
        scores = self.score_samples(X)
        return (scores < self.threshold).astype(int)  # 1 = anomaly/fraud

    def plot_training_loss(self):
        if self._history:
            plt.figure(figsize=(10,6))
            plt.plot(self._history.history['loss'], label='Training Loss')
            plt.plot(self._history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title('Autoencoder Training Curve')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Model not trained yet.")

def evaluate(y_true, y_pred):
    """
    Generic evaluation function for anomaly detection results.
    """
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Correctly Predicted Samples: {(y_pred == y_true).sum()} / {len(y_true)}")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Non-Fraud', 'Fraud']).plot(cmap='Blues')
    plt.show()
