# 📊 Common Machine Learning Loss Functions & Metrics

This file summarizes several popular **loss functions** and **evaluation metrics**, including what they measure and when to use them.

---

## 1. Binary Cross-Entropy (BCE)
**Type:** Classification  
**Use for:** Binary classification (e.g., logistic regression, neural networks)

**Formula:**
```

BCE = - (1/N) * Σ [ y * log(y_pred) + (1 - y) * log(1 - y_pred) ]

```

**Description:**  
Measures the difference between predicted probabilities and true binary labels.  
Penalizes confident incorrect predictions heavily.

---

## 2. Hinge Loss
**Type:** Classification  
**Use for:** Margin-based classifiers (e.g., SVM)

**Formula:**
```

Hinge = (1/N) * Σ max(0, 1 - y * y_pred)

```

**Description:**  
Used when labels ∈ {−1, +1}. Enforces a margin — loss is zero for confidently correct predictions.

---

## 3. Huber Loss
**Type:** Regression  
**Use for:** Robust regression that balances MSE and MAE

**Formula:**
```

If |y - y_pred| <= δ:
Loss = 0.5 * (y - y_pred)^2
Else:
Loss = δ * (|y - y_pred| - 0.5 * δ)

```

**Description:**  
Quadratic for small errors, linear for large errors — reduces sensitivity to outliers.

---

## 4. Mean Absolute Error (MAE)
**Type:** Regression  
**Use for:** Average magnitude of prediction errors

**Formula:**
```

MAE = (1/N) * Σ |y - y_pred|

```

**Description:**  
Represents the average absolute difference between predicted and actual values.  
Less sensitive to outliers than MSE.

---

## 5. Mean Absolute Percentage Error (MAPE)
**Type:** Regression  
**Use for:** Error expressed as a percentage of actual values

**Formula:**
```

MAPE = (100/N) * Σ |(y - y_pred) / y|

```

**Description:**  
Shows prediction error as a percentage.  
Be cautious if y contains zeros — division instability.

---

## 6. Mean Squared Error (MSE)
**Type:** Regression  
**Use for:** Penalizing large errors strongly

**Formula:**
```

MSE = (1/N) * Σ (y - y_pred)^2

```

**Description:**  
Squares the error — large deviations are penalized more.  
Smooth and widely used in model optimization.

---

## 7. Root Mean Squared Error (RMSE)
**Type:** Regression  
**Use for:** Error in same units as target variable

**Formula:**
```

RMSE = sqrt( (1/N) * Σ (y - y_pred)^2 )

```

**Description:**  
Square root of MSE — interpretable in the same scale as the target variable.  
More sensitive to large errors.

---

## 8. Coefficient of Determination (R²)
**Type:** Regression metric  
**Use for:** Measuring model fit quality

**Formula:**
```

R2 = 1 - [ Σ (y - y_pred)^2 / Σ (y - mean(y))^2 ]

```

**Description:**  
Indicates how much variance in y is explained by the model.  
R² = 1 → perfect fit, 0 → no better than mean prediction, < 0 → worse than mean.

---

## 9. Symmetric Mean Absolute Percentage Error (SMAPE)
**Type:** Regression  
**Use for:** Balanced relative error (scale-independent)

**Formula:**
```

SMAPE = (100/N) * Σ [ 2 * |y - y_pred| / (|y| + |y_pred|) ]

```

**Description:**  
Prevents asymmetry seen in MAPE and keeps percentage error bounded (0–200%).

---

## 🔖 Summary Table

| Metric / Loss | Type | Outlier Robustness | Common Use |
|----------------|------|--------------------|-------------|
| **BCE** | Classification | No | Binary probability models |
| **Hinge** | Classification | No | SVMs, margin-based models |
| **Huber** | Regression | Yes | Smooth, hybrid loss |
| **MAE** | Regression | Yes | Simple average error |
| **MAPE** | Regression | ⚠️ | Relative percentage error |
| **MSE** | Regression | No | Standard regression loss |
| **RMSE** | Regression | No | Human-interpretable scale |
| **R²** | Regression | — | Model fit quality |
| **SMAPE** | Regression | Yes | Scale-invariant percentage error |

---

### 🧩 Notes
- **MSE/RMSE:** penalize large deviations heavily.  
- **MAE/Huber/SMAPE:** more robust when data has outliers.  
- **BCE/Hinge:** for classification models.  
- **R²:** a statistical metric, not a loss.

