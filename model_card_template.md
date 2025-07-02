# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a **Logistic Regression Classifier** implemented using the `scikit-learn` library in Python. Its purpose is to predict whether an individual's income is greater than 50,000 USD ('>50K') or less than or equal to 50,000 USD ('<=50K') based on various demographic and employment-related features from the Adult Census Income dataset.

## Intended Use
This model is intended for **research, educational, and exploratory data analysis purposes**. It can be used by data scientists, researchers, and potentially policy analysts to understand factors correlated with income levels in the provided dataset.

It is **NOT intended for high-stakes decision-making**, such as employment screening, credit scoring, or any application that could lead to unfair or discriminatory outcomes against individuals. The model is built on historical data (1994) and should not be used to make predictions about current populations or for making critical decisions that impact individuals' lives.

## Training Data
The model was trained on a subset of the **Adult Census Income dataset (`census.csv`)**.
* **Source:** The data is derived from the 1994 Census database.
* **Size:** The training dataset comprised approximately **80%** of the total `census.csv` records. The exact number of records can be derived from the total dataset size, which is typically around 32,561 entries, leading to roughly 26,048 records for training.
* **Features:** The dataset includes numerical features (e.g., `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and categorical features (e.g., `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).
* **Preprocessing:** Categorical features were transformed using **OneHotEncoding**, and the target label (`salary`) was binarized ('>50K' to 1, '<=50K' to 0) using **LabelBinarizer**. Continuous features were used as is, without scaling.

## Evaluation Data
The model was evaluated on a held-out test subset of the **Adult Census Income dataset (`census.csv`)**.
* **Source:** Same as the training data, but a distinct portion.
* **Size:** The test dataset comprised approximately **20%** of the total `census.csv` records, roughly 6,513 records.
* **Preprocessing:** The test data underwent the same preprocessing steps as the training data, utilizing the *same encoder and label binarizer fitted on the training data* to ensure consistency in feature representation.

## Metrics
The model's performance was evaluated using **Precision, Recall, and F1-score**.

* **Precision:** The proportion of positive identifications that were actually correct.
* **Recall:** The proportion of actual positives that were identified correctly.
* **F1-score:** The harmonic mean of Precision and Recall, providing a balance between the two. The `fbeta_score` with `beta=1` was used, making it equivalent to the F1-score.

**Overall Model Performance (on the test dataset):**
* **Precision:**  0.7088
* **Recall:** 0.2717
* **F1-score:** 0.3928

**Performance on Categorical Slices (Observations from `slice_output.txt`):**
The model's performance varies significantly across different slices of the categorical features. This highlights the importance of slice-based analysis.

* **`workclass`:** Performance varied, with high F1-scores for very small groups like 'Never-worked' and 'Without-pay' (likely due to high precision/recall on very few samples), but generally lower F1 for larger categories like 'Private' (F1: 0.3822).
* **`education`:** (Example from your output sample was `10th`). Performance can vary.
* **`occupation`:** Performance differs across occupations. For instance, 'Sales' (F1: 0.3696) and 'Transport-moving' (F1: 0.3038) showed moderate F1-scores, while others might differ.
* **`relationship`:** The 'Husband' slice showed a relatively higher F1-score (0.3914) compared to 'Own-child' (F1: 0.2791) or 'Other-relative' (F1: 0.4000).
* **`race`:** Metrics varied across racial groups. For example, 'Asian-Pac-Islander' showed F1 of 0.4110, while 'Black' had 0.3910. The 'Other' race had a perfect F1 (0.7500) likely due to a very small sample size.
* **`sex`:** Performance differed between genders. 'Female' showed F1 of 0.4010, while 'Male' showed F1 of 0.3910.
* **`native-country`:** This feature exhibited wide variations, with many countries having very small sample counts, leading to highly variable or extreme (0.0 or 1.0) F1-scores (e.g., 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Poland' etc. with small counts showing perfect or zero scores). The 'United-States' slice, being the largest, showed metrics closer to the overall model performance (F1: 0.3948).

These variations suggest potential biases or differential performance based on specific demographic and socio-economic groups present in the data.

## Ethical Considerations
* **Bias in Data:** The Adult Census Income dataset reflects historical demographics and societal biases from 1994. The model inherits these biases, which can lead to unfair or inaccurate predictions for underrepresented or historically disadvantaged groups. For instance, the differing performance across 'race', 'sex', and 'native-country' categories observed in the slice analysis highlights this.
* **Sensitive Attributes:** Features like `race`, `sex`, and `native-country` are sensitive attributes. Using a model trained on such data for real-world high-stakes applications could lead to discriminatory outcomes.
* **Interpretability:** While Logistic Regression is relatively interpretable, the one-hot encoding of many categorical features can make direct interpretation of individual coefficients challenging.
* **Exacerbating Inequality:** If deployed irresponsibly, a model like this could inadvertently perpetuate or exacerbate existing societal inequalities related to income, especially if predictions are used to inform policy or resource allocation without careful consideration of fairness.

## Caveats and Recommendations
* **Data Timeliness:** The dataset is from 1994 and may not accurately represent current economic or demographic realities. Using it for contemporary predictions is inappropriate.
* **Limited Features:** The model uses a fixed set of features. More advanced feature engineering, or incorporating additional relevant data (e.g., regional economic indicators, policy changes), could improve performance and robustness.
* **Model Simplicity:** Logistic Regression is a linear model. While interpretable, it might not capture complex non-linear relationships in the data. Experimenting with more sophisticated models (e.g., Gradient Boosting, Neural Networks) could yield higher overall metrics, but would require careful consideration of interpretability and potential for increased bias.
* **Fairness Research:** Further research into fairness metrics (e.g., statistical parity, equalized odds, predictive parity) and bias mitigation techniques (e.g., re-sampling, adversarial debiasing) is strongly recommended, especially for sensitive attributes, if similar models are to be considered for any real-world application.
* **Domain Expertise:** Consult with domain experts (economists, sociologists) to validate findings and understand the real-world implications of model predictions and biases.