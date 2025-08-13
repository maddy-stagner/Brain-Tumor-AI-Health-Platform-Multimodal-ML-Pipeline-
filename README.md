This project implements an AI-powered health platform for brain tumor detection, using  data to simulate a real-world healthcare AI pipeline. It integrates image-based features (e.g., MRI scans) and tabular clinical data (e.g., patient symptoms, vitals) to build a fused machine learning model.

It showcases:

Multimodal data fusion (images + tabular)

CNN-based feature extraction

XGBoost classification

Stratified K-Fold validation

Explainability (SHAP)

Fairness & anomaly detection

This architecture is ideal for research, academic demo projects, and as a scalable base for deploying with real healthcare datasets (e.g., BraTS, TCGA, EHR systems).

AI Model Pipeline
1. CNN for Image Feature Extraction
Functional API model with Conv → MaxPooling → Dense (penultimate feature layer)

Dropout and L2 regularization to avoid overfitting

2. Tabular Clinical Data
10 simulated clinical features

Class-conditional Gaussian distribution

3. Feature Fusion & Classification
Fuses CNN features + tabular data

Uses XGBoost with early stopping and regularization

4. Evaluation
Stratified 5-Fold CV

Classification Report (Accuracy, Precision, Recall, F1)

SHAP explanations (per-class + summary)

5. Additional Insights
Anomaly Detection: Isolation Forest on tabular data

Fairness Check: Accuracy by synthetic age groups (young, middle, old)

