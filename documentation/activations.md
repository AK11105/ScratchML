# Activation Functions and Rationale

## **1. Sigmoid**

* **Purpose:** Maps input values to a range between 0 and 1.
* **Why it was made:** Useful for modeling probabilities and for early neural networks.
* **Limitations:** Saturates for large positive/negative inputs, leading to vanishing gradients.

---

## **2. TanH**

* **Purpose:** Maps input values to a range between -1 and 1.
* **Why it was made:** Centered around zero, which often leads to faster convergence than sigmoid.
* **Limitations:** Still suffers from vanishing gradients at large input magnitudes.

---

## **3. ReLU (Rectified Linear Unit)**

* **Purpose:** Outputs zero for negative inputs, linear for positive inputs.
* **Why it was made:** Computationally efficient, avoids vanishing gradient for positive inputs.
* **Limitations:** “Dead neurons” problem—neurons that never activate for negative inputs.

---

## **4. LeakyReLU**

* **Purpose:** Like ReLU but allows a small, non-zero gradient for negative inputs.
* **Why it was made:** To solve ReLU’s dead neuron problem while keeping simplicity.

---

## **5. ELU (Exponential Linear Unit)**

* **Purpose:** Smoothly saturates for negative inputs, linear for positive.
* **Why it was made:** Improves learning speed and reduces bias shift compared to ReLU.
* **Extra:** Helps networks converge faster and handle negative inputs better.

---

## **6. GeLU (Gaussian Error Linear Unit)**

* **Purpose:** Combines ReLU-style activation with a smooth, probabilistic gating.
* **Why it was made:** Improves performance in transformer-based models and deep networks by giving smoother gradient transitions.

---

## **7. SeLU (Scaled ELU)**

* **Purpose:** Self-normalizing activation function.
* **Why it was made:** Maintains zero mean and unit variance through layers, stabilizing deep networks without explicit batch normalization.

---

## **8. ThresholdedReLU**

* **Purpose:** ReLU variant that only activates if input exceeds a certain threshold.
* **Why it was made:** Adds flexibility in controlling neuron sparsity and activation patterns.

---

## **9. Softmax**

* **Purpose:** Converts a vector of values into probabilities that sum to 1.
* **Why it was made:** Essential for multi-class classification.

---

## **10. LogSoftmax**

* **Purpose:** Logarithm of softmax outputs.
* **Why it was made:** Improves numerical stability for loss computations (e.g., cross-entropy).

---

## **11. Swish**

* **Purpose:** Smooth, non-monotonic activation: ( x \cdot \text{sigmoid}(x) ).
* **Why it was made:** Empirically shown to outperform ReLU in deep networks by reducing gradient instability.

---

## **12. Mish**

* **Purpose:** Smooth, non-monotonic activation: ( x \cdot \tanh(\ln(1 + e^x)) ).
* **Why it was made:** Similar motivation to Swish but with slightly better empirical performance and smoother transitions.

---

## **13. HardSigmoid**

* **Purpose:** Computationally cheaper approximation of Sigmoid.
* **Why it was made:** Useful in resource-constrained environments (mobile, embedded devices).

---

## **14. HardSwish**

* **Purpose:** Approximation of Swish with low computational cost.
* **Why it was made:** Used in mobile networks like MobileNetV3 to retain Swish benefits efficiently.

---

## **15. Maxout**

* **Purpose:** Takes the maximum of multiple linear functions.
* **Why it was made:** Generalizes ReLU, can approximate any convex function, avoids dead neuron problem, works well with dropout.

---

## Summary

| Activation          | Range             | Smooth | Monotonic | Key Feature / Use Case                                       |
| ------------------- | ----------------- | ------ | --------- | ------------------------------------------------------------ |
| **Sigmoid**         | (0, 1)            | Yes    | Yes       | Probabilities, early NN; suffers vanishing gradient          |
| **TanH**            | (-1, 1)           | Yes    | Yes       | Zero-centered, faster convergence than Sigmoid               |
| **ReLU**            | [0, ∞)            | No     | Yes       | Simple, avoids vanishing gradient for positives              |
| **LeakyReLU**       | (-∞, ∞)           | No     | Yes       | Solves dead neuron problem                                   |
| **ELU**             | (-α, ∞)           | Yes    | Yes       | Smooth for negatives, faster learning                        |
| **GeLU**            | (-∞, ∞)           | Yes    | No        | Smooth, probabilistic gating, used in Transformers           |
| **SeLU**            | (-∞, ∞)           | Yes    | Yes       | Self-normalizing, stabilizes deep networks                   |
| **ThresholdedReLU** | [0, ∞)            | No     | Yes       | Activates only above a threshold                             |
| **Softmax**         | (0, 1)            | Yes    | Yes       | Multi-class probability output                               |
| **LogSoftmax**      | (-∞, 0)           | Yes    | Yes       | Numerically stable version of Softmax                        |
| **Swish**           | (-0.28, ∞) approx | Yes    | No        | Smooth, non-monotonic, improves deep network training        |
| **Mish**            | (-0.31, ∞) approx | Yes    | No        | Smooth, non-monotonic, better gradient flow than Swish       |
| **HardSigmoid**     | (0, 1)            | Approx | Yes       | Cheap Sigmoid approximation, resource-efficient              |
| **HardSwish**       | (0, ∞) approx     | Approx | Yes       | Efficient Swish approximation for mobile networks            |
| **Maxout**          | (-∞, ∞)           | No     | No        | Maximum of multiple linear functions, flexible approximation |


