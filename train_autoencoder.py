import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from anomaly_autoencoder_gmm_detector import AutoencoderGMMDetector, evaluate


# Preprocessing
cat_cols = ['transaction_type', 'repeat_transaction', 'transaction_mode','risk_score']

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ]
)


# Load Data
df = pd.read_csv('your_transactions.csv')  # replace with your file
X = df.drop(columns=['fraud'])
y = df['fraud']


# Train-test split
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Only genuine transactions for training autoencoder
X_train_genuine = X_train_full[y_train_full == 0]


# Pipeline

autoencoder_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('autoencoder', AutoencoderGMMDetector(
        code_dim=2, epochs=25, batch_size=64, n_components=2,
        learning_rate=0.0005, anomaly_percentile=10
    ))
])


# Train
autoencoder_pipeline.fit(X_train_genuine)

# Plot training loss
autoencoder_pipeline.named_steps['autoencoder'].plot_training_loss()


# Predictions
X_test_full['anomaly_flag'] = autoencoder_pipeline.predict(X_test_full)
X_test_full['anomaly_score'] = autoencoder_pipeline.score_samples(X_test_full)


# Evaluation
evaluate(y_test_full, X_test_full['anomaly_flag'])
