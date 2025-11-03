# Knee KL Grade Classification: Ensemble Evaluation Report

## 1. Project Methodology Overview

The model evaluated is a **5-fold ensemble** designed to classify knee X-rays into one of five Kellgren-Lawrence (KL) grades (0 to 4).

### Ensemble Method (Soft Voting)
Our final prediction comes from combining 5 separate models. For every test image, each model outputs its probability scores for the 5 grades. The ensemble uses **Soft Voting** by **averaging these probabilities** across all 5 models, and the class with the highest average score is the final prediction. 

---

## 2. Metric Analysis and Interpretation

We report two key metrics, with **Balanced Accuracy** being the most critical due to the nature of medical classification tasks (where class imbalance is common).

### A. Standard Accuracy (The Overall Score)

* **What it Measures:** The overall percentage of correctly classified samples.
    $$
    \text{Accuracy} = \frac{\text{Total Correct Predictions}}{\text{Total Number of Predictions}}
    $$
* **Result:** **51.17%**
* **Interpretation & Pitfall:** While this score is above the random chance baseline (20%), it is an unreliable metric for assessing clinical utility. If the dataset were highly imbalanced (e.g., 90% Grade 0), this score could be misleadingly high, masking a failure to detect rare, severe cases (the **Accuracy Paradox**).

---

### B. Balanced Accuracy (The Fair Score)

* **What it Measures:** The average accuracy (or **Recall**) achieved for **each individual KL grade**. This ensures that the model's performance on rare classes (like Grade 4) is weighted equally to its performance on common classes (like Grade 0).
* **How it's Calculated:**
    $$
    \text{Balanced Accuracy} = \frac{\sum_{i=0}^{4} \text{Recall}_{\text{Grade}_i}}{5}
    $$
* **Result:** **52.92%**
* **Interpretation:** This is the most honest representation of the model's performance. The fact that the Balanced Accuracy (52.92%) is only slightly higher than the Standard Accuracy (51.17%) suggests that the model's performance is consistently limited across all grades, or that the test set is relatively balanced. **A $\approx 53\%$ balanced accuracy indicates significant struggle in reliably distinguishing between the five KL grades.**

---

## 3. Conclusion and Recommendations

### Summary of Performance
The 5-fold ensemble model, trained for 20 epochs per fold, achieved a final **Balanced Accuracy of 52.92%**.

### Key Insight
The low value for Balanced Accuracy is the primary actionable insight. It shows that the model's overall predictive power is limited, suggesting that the model is often confused when differentiating between the five severity grades.

### Next Steps for Improvement
Future efforts should be directed toward methods that specifically boost performance on the most difficult-to-classify grades. This could involve:
* Increasing the number of **epochs** to allow for greater convergence during training.
* Exploring more **advanced data augmentation** techniques.
* Testing **alternative model architectures** or using a specialized **Ordinal Classification** loss function, which is often better suited for ordered labels like the KL scale.