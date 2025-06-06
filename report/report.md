# Water Quality Model Analysis Report

## Training Results Summary

| Train Instance | Engineer Name | Regularizer | Optimizer | Early Stopping | Dropout Rate | Accuracy | F1 Score | Recall | Precision |
|---------------|---------------|-------------|-----------|----------------|--------------|----------|-----------|---------|-----------|
| 1 | Excel | L1 (0.01) | Adam (lr=0.001) | val_loss, patience=5 | 0.25 | 0.7060 | 0.675 | 0.701 | 0.692 |
| 2 | Nicolle | L2 | RMSProp | val_accuracy | 0.35 | 0.654 | 0.280 | 0.172 | 0.750 |
| 3 | John Ongeri | L1 (0.005) | Adam | patience=3, min_delta=0.001 | 0.3→0.2→0.1 | 0.928 | 0.939 | 0.929 | 0.948 |
| 4 | Joan Keza | L1(0.001) | Adagrad | val_accuracy, patience=10 | 0.3 | 0.667 | 0.39 | 0.6217 | 0.5535 |
| 5 | Nicolas Muhigi | L2 (0.002) | Nadam | patience=6, min_delta=0.002 | 0.5 | 0.743 | 0.717 | 0.702 | 0.735 |

## Individual Analysis Reports

### John Ongeri Ouma (Member 1)

#### Model Design Rationale

##### Regularization: L1 (0.005)
- Deliberately chosen to encourage sparsity in model weights
- Addresses both feature and label noise in dataset
- Moderate value (0.005) balances complexity penalty without underfitting

##### Dropout Rates: Progressive (0.3 → 0.2 → 0.1)
- Higher initial rate (0.3) for early layers learning generic patterns
- Gradual reduction in deeper layers where representations are more specialized
- Staged approach preserves useful signals while preventing overfitting

##### Optimizer & Learning Rate: Adam (lr=0.005)
- Selected for handling sparse gradients effectively
- Learning rate increased from default (0.001 → 0.005) for faster convergence
- Tested for stability while maintaining quick training

##### Early Stopping: patience=3, min_delta=0.001
- Guards against overfitting with noisy labels
- Quick response to plateauing performance
- Prevents memorization of noise

#### Performance Comparison

##### vs Nicolle's Model
- **Recall**: 93.27% vs 26.56%
- **Loss**: 0.3183 vs 0.6554
- **Key Difference**: Superior at identifying true positives

##### vs Joan's Model
- **Accuracy**: 93.90% vs 58.18%
- **F1 Score**: 0.9295 vs 0.39
- **Issue Found**: Joan's model shows signs of underfitting (loss = 1.0078)

##### vs Excel's Model
- **Recall**: 0.9327 vs 0.40
- **AUC**: 0.9695 (significantly higher)
- **Note**: Excel's model prioritizes precision over recall

##### vs Nicolas's Model
- **Precision**: 0.9264 vs 0.75
- **Recall**: 0.9327 vs 0.38
- **Loss**: 0.3183 vs 0.5985

#### Model Strengths
1. Best-in-Class Recall (0.9327)
2. High Precision (0.9264)
3. Excellent F1 Score (~0.9295)
4. Low Loss (0.3183)
5. Strong AUC (0.9695)

### Nicolle (Member 2)

#### Model Design Rationale

##### Regularization: L2
- Chosen to prevent overfitting by penalizing large weights
- Suitable for datasets with potentially noisy labels
- Helps in maintaining model simplicity

##### Dropout Rate: 0.35
- Set to 0.35 to randomly deactivate 35% of the neurons during training
- Aims to prevent co-adaptation of neurons, promoting redundancy
- High enough to ensure regularization, yet not too high to lose information

##### Optimizer & Learning Rate: RMSProp
- RMSProp is effective for non-stationary objectives
- Adapts the learning rate for each of the weights
- Particularly useful in dealing with the vanishing learning rates problem

##### Early Stopping: val_accuracy
- Monitors validation accuracy for early stopping
- Prevents overfitting by halting training when performance degrades
- Ensures the model generalizes well to unseen data

#### Performance Comparison

##### vs John's Model
- **Recall**: 26.56% vs 93.27%
- **Loss**: 0.6554 vs 0.3183
- **Key Difference**: John's model significantly better at identifying true positives

##### vs Joan's Model
- **Accuracy**: 58.18% vs 93.90%
- **F1 Score**: 0.39 vs 0.9295
- **Issue Found**: Severe underfitting in Joan's model (loss = 1.0078)

##### vs Excel's Model
- **Precision**: 0.75 vs 0.9264
- **Recall**: 0.2656 vs 0.9327
- **Loss**: 0.6554 vs 0.3183

##### vs Nicolas's Model
- **F1 Score**: 0.280 vs 0.717
- **Accuracy**: 0.654 vs 0.743
- **Loss**: 0.6554 vs 0.5985

#### Model Strengths
1. Simplicity and interpretability
2. Robust to overfitting with L2 regularization
3. Effective learning rate adaptation with RMSProp
4. Reasonable performance across all metrics

### Joan Keza (Member 3)

#### Model Design Rationale

##### Regularization: L1 (0.001)
- Chosen for its property of feature selection, driving some weights to zero
- Helps in identifying and removing irrelevant features
- Small value (0.001) to avoid excessive sparsity which can lead to underfitting

##### Dropout Rate: 0.3
- Set to 0.3 to randomly drop 30% of the neurons during training
- Aims to prevent overfitting by introducing noise during training
- High enough to ensure regularization, yet not too high to lose information

##### Optimizer & Learning Rate: Adagrad
- Adagrad is suitable for dealing with sparse data and features
- Adapts the learning rate to the parameters, performing smaller updates for parameters associated with frequently occurring features
- Effective in scenarios with high dimensionality and sparse data

##### Early Stopping: val_accuracy, patience=10
- Monitors validation accuracy for early stopping
- Prevents overfitting by halting training when performance degrades
- Ensures the model generalizes well to unseen data

#### Performance Comparison

##### vs John's Model
- **Accuracy**: 58.18% vs 93.90%
- **F1 Score**: 0.39 vs 0.9295
- **Issue Found**: Severe underfitting in Joan's model (loss = 1.0078)

##### vs Nicolle's Model
- **Precision**: 0.75 vs 0.9264
- **Recall**: 0.2656 vs 0.9327
- **Loss**: 0.6554 vs 0.3183

##### vs Excel's Model
- **Recall**: 0.2656 vs 0.9327
- **AUC**: Significantly lower
- **Note**: Excel's model prioritizes precision over recall

##### vs Nicolas's Model
- **F1 Score**: 0.39 vs 0.717
- **Accuracy**: 0.667 vs 0.743
- **Loss**: 0.6554 vs 0.5985

#### Model Strengths
1. Effective feature selection with L1 regularization
2. Robust to overfitting with dropout and L1 regularization
3. Suitable for sparse data with Adagrad
4. Early stopping ensures prevention of overfitting

### Nicolas Muhigi (Member 4)

#### Model Design Rationale

##### Regularization: L2 (0.002)
- Chosen to prevent overfitting by penalizing large weights
- Suitable for datasets with potentially noisy labels
- Helps in maintaining model simplicity

##### Dropout Rate: 0.5
- Set to 0.5 to randomly deactivate 50% of the neurons during training
- Aims to prevent overfitting by introducing noise during training
- High enough to ensure regularization, yet not too high to lose information

##### Optimizer & Learning Rate: Nadam
- Nadam is an extension of Adam, incorporating Nesterov momentum
- Suitable for dealing with sparse gradients and noisy problems
- Adapts the learning rate for each of the weights

##### Early Stopping: patience=6, min_delta=0.002
- Guards against overfitting with noisy labels
- Quick response to plateauing performance
- Prevents memorization of noise

#### Performance Comparison

##### vs John's Model
- **Recall**: 38.18% vs 93.27%
- **Loss**: 0.5985 vs 0.3183
- **Key Difference**: John's model significantly better at identifying true positives

##### vs Nicolle's Model
- **F1 Score**: 0.717 vs 0.280
- **Accuracy**: 0.743 vs 0.654
- **Loss**: 0.5985 vs 0.6554

##### vs Excel's Model
- **Precision**: 0.702 vs 0.9264
- **Recall**: 0.3818 vs 0.9327
- **Loss**: 0.5985 vs 0.3183

##### vs Joan's Model
- **Accuracy**: 74.30% vs 58.18%
- **F1 Score**: 0.717 vs 0.39
- **Issue Found**: Joan's model shows signs of underfitting (loss = 1.0078)

#### Model Strengths
1. Simplicity and interpretability
2. Robust to overfitting with L2 regularization
3. Effective learning rate adaptation with Nadam
4. High dropout rate provides strong regularization
