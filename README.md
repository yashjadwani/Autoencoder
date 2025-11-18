## **_Detecting anomalies like Sherlock Holmes, but with tensors and neurons._**
 
**Autoencoder**: A neural network that learns to compress input into a low-dimensional representation and then reconstruct it; commonly used for anomaly detection.

## Purpose
- Detect anomalous transactions (possible fraud) by learning normal patterns from historical genuine transaction data.
- Operates in an unsupervised fashion, no labelled fraud data required for training.

## Architecture
- Fully-connected feedforward network (dense layers).
- **Encoder**: compresses input features through progressively smaller layers: 64 → 32 → 16 → latent code (`code_dim`).
- **Latent code**: captures the most important patterns/features in a low-dimensional space.
- **Decoder**: symmetric expansion to reconstruct the original input: 16 → 32 → 64 → output.
- **Output Layer**: reconstructs all original features to compute reconstruction error.

## Training Details
- **Loss Function**: Mean Squared Error (MSE) between input and reconstructed output.
- **Optimiser**: Adam (adaptive learning rate, efficient for sparse data).
- **Early Stopping**: stops training if validation loss doesn’t improve for a set number of epochs, preventing overfitting.
- **Batch Size & Epochs**: adjustable hyperparameters (`batch_size` and `epochs`) for controlling convergence and memory usage.

## Anomaly Detection
- **Reconstruction Error**: computes the absolute difference between input and output for each transaction.
- **GMM Scoring**: Gaussian Mixture Model models the distribution of reconstruction errors to estimate likelihood.
- **Thresholding**: transactions with log-likelihood below a specified percentile (`anomaly_percentile`) are flagged as anomalies (fraud).
- Allows fine-tuning of the recall/precision tradeoff via the anomaly percentile.

## Hyperparameter Tuning Guidelines
- **code_dim**: Larger values capture more detail but may overfit small datasets. Start small (2–5) and increase if the reconstruction error is too high.
- **epochs**: More epochs may improve learning, but monitor for overfitting using validation loss.
- **batch_size**: Smaller batches can lead to more stable learning; larger batches train faster.
- **learning_rate**: Adjust based on dataset size. Typical range for Adam: 0.0001–0.001.
- **n_components (GMM)**: Start with 2–5 components. Increase if the reconstruction error distribution is multimodal.
- **anomaly_percentile**: Set according to your tolerance for false positives/negatives (e.g., 5–10% for rare fraud).
- Users should experiment with these values on a validation set that reflects their own transaction data to optimise recall and precision.

## Advantages
- Learns **non-linear relationships** in the data.
- Can handle **high-dimensional transaction data** efficiently.
- Adaptive anomaly detection that accounts for variations in normal behaviour.
- Suitable for **streaming or batch inference** once trained.

## Outputs
- **Anomaly Flag**: 1 if a transaction is likely fraudulent, 0 otherwise.
- **Anomaly Score**: log-likelihood or reconstruction error for further risk assessment.
- **Training Curves**: loss over epochs for monitoring convergence and detecting overfitting.
