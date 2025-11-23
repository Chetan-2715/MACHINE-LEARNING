# 📘 Machine Learning Lab – Experiments 1 to 5

This repository contains the core Machine Learning laboratory experiments with simple explanations, advantages, disadvantages, and code implementations.

## 📘 Lab 1 – Python Libraries for Data Science

### 📌 **Overview**
Lab 1 introduces the core Python libraries used in Data Science and Machine Learning: **Pandas**, **NumPy**, **Scikit-learn**, and **Matplotlib**.  
The main objective is to understand how to manipulate data using DataFrames and visualize it using different types of plots.

### 📌 **Aim**
To implement DataFrames using the **Pandas** library and visualize them using different chart types available in the **Matplotlib** library.

---

## 📚 **Theory Summary**

The lab covers five commonly used data visualization techniques:

### 🔹 **1. Bar Graph**
Used to compare values across different categories using rectangular bars.  
**Syntax:** `plt.bar(x, y)`

### 🔹 **2. Pie Chart**
Represents data proportions as slices of a circle.  
**Syntax:** `plt.pie(values, labels=...)`

### 🔹 **3. Box Plot**
Displays data distribution through quartiles and helps detect outliers.  
**Syntax:** `plt.boxplot(data)`

### 🔹 **4. Histogram**
Shows the frequency distribution of continuous data by grouping values into bins.  
**Syntax:** `plt.hist(values, bins=...)`

### 🔹 **5. Line Chart with Subplots**
Used for showing trends over a sequence and displaying multiple plots in one figure.  
**Syntax:** `plt.subplots(nrows, ncols)`

---

## ✅ **Advantages**
- Introduces the most essential tools for any data science or ML workflow.
- Visualization helps understand dataset patterns quickly.
- Libraries like Pandas and NumPy make data manipulation extremely efficient.
- Matplotlib provides flexible and customizable plotting options.
- Builds foundational skills needed for all future ML algorithms.

---

## 📘 Lab 2 – Evaluation Metrics

### 📌 **Overview**
Lab 2 focuses on understanding and calculating the core performance metrics used to evaluate **classification models** in Machine Learning. These metrics help determine how well a model performs in real-world scenarios.

### 📌 **Aim**
To study and implement the fundamental evaluation metrics: **Accuracy, Precision, Recall, and F1-Score** using a classification model.

---

## 📚 **Theory Summary**

### 🔹 **Confusion Matrix**
A Confusion Matrix is an **N × N** table that compares actual labels with predicted labels. It forms the basis for all classification metrics.  
It contains four key values:

- **TP (True Positive):** Model correctly predicts positive class  
- **TN (True Negative):** Model correctly predicts negative class  
- **FP (False Positive):** Model predicts positive but actual is negative (Type-1 Error)  
- **FN (False Negative):** Model predicts negative but actual is positive (Type-2 Error)  

---

## 🧮 **Metrics Covered**

### 🔹 **1. Accuracy**
\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]
Measures overall correctness but **fails for imbalanced datasets**.

---

### 🔹 **2. Precision**
\[
Precision = \frac{TP}{TP + FP}
\]
Indicates how many predicted positives were actually correct.  
Useful when **False Positives** are costly.

---

### 🔹 **3. Recall (Sensitivity)**
\[
Recall = \frac{TP}{TP + FN}
\]
Measures how many actual positives were correctly predicted.  
Critical when **False Negatives** are more risky.

---

### 🔹 **4. F1-Score**
\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]
Harmonic mean of Precision & Recall — balances both.

---

## ✅ **Advantages**
- Offers detailed insights into model performance.  
- Metrics like Precision & Recall help evaluate models in **imbalanced datasets**.  
- Confusion matrix shows exact error types (FP, FN).  
- Helps choose the right model by comparing multiple evaluation metrics.  

---

## 📘 Lab 3 – Train and Test Sets (Data Splitting)

### 📌 **Overview**
Lab 3 focuses on understanding the concept of splitting a dataset into **training** and **testing** sets.  
This step is crucial in any machine learning workflow because it ensures that a trained model can generalize to unseen data and does not simply memorize the dataset.

### 📌 **Aim**
To study and implement the creation of **Train-Test Splits** using Scikit-learn’s `train_test_split` function.

---

## 📚 **Theory Summary**

### 🔹 **Why Split Data?**
Splitting the data prevents:
- **Overfitting** (model memorizes training data)
- **Biased evaluation**
- **Poor generalization** on real-world data

A model should be trained on one portion of the data and evaluated on another, unseen portion.

---

### 🔹 **The `train_test_split` Function**
Provided by **sklearn.model_selection**, it divides dataset features (X) and labels (y) into:

- **X_train** – Training features  
- **X_test** – Testing features  
- **y_train** – Training labels  
- **y_test** – Testing labels  

### 🔑 **Important Parameters**
- `test_size` → Defines what portion of data goes into the test set  
  - Example: `test_size=0.2` → 20% test, 80% train  
- `random_state` → Sets a fixed seed so the split is reproducible each time  

### 🔄 **Process**
1. Load dataset  
2. Separate features (X) and target (y)  
3. Apply `train_test_split`  
4. Train the model using training set  
5. Evaluate using testing set  

---
## ✅ Advantages
- Ensures fair and objective evaluation of ML models.  
- Prevents overfitting by testing on unseen data.  
- Simple and easy to apply using Scikit-learn.  
- Allows adjustable split ratios (e.g., 70/30, 80/20, 90/10).  

---

## 📘 Lab 4 – Linear Regression

### 📌 **Overview**
Lab 4 focuses on understanding and implementing the **Linear Regression** algorithm — one of the most fundamental supervised machine learning techniques used for predicting continuous values.

### 📌 **Aim**
To study the Linear Regression algorithm and implement it for predicting a continuous dependent variable.

---

## 📚 **Theory Summary**

### 🔹 **Simple Linear Regression**
Linear Regression models the relationship between:
- **Independent variable (X)**
- **Dependent variable (Y)**

Using the equation:

\[
Y_i = \beta_0 + \beta_1 X_i
\]

Where:  
- \( \beta_0 \) = Intercept  
- \( \beta_1 \) = Slope  
- \( Y_i \) = Predicted output  
- \( X_i \) = Input feature  

---



## ✅ **Advantages**
- Simple and easy to implement.  
- Computationally efficient and fast.  
- Works well when variables have a linear relationship.  
- Easy to interpret coefficients (slope & intercept).  
- Useful as a baseline model in regression tasks.  

---

## ❌ **Disadvantages**
- Sensitive to outliers, which can distort the best-fit line.  
- Poor performance on complex or nonlinear datasets.  
- Multiple independent variables may cause multicollinearity issues.  

---

## 📘 Lab 5 – Multivariable Regression

### 📌 **Overview**
Lab 5 focuses on **Multivariable (Multiple) Linear Regression**, an extension of simple linear regression.  
Instead of using a single feature to predict the target, this algorithm uses **multiple independent variables** to model the relationship more accurately.

### 📌 **Aim**
To study and implement the Multivariable Regression algorithm and understand how multiple features together influence a continuous dependent variable.

---

## 📚 **Theory Summary**

### 🔹 **What is Multivariable Regression?**
Multivariable Regression models the dependent variable \( y \) as a linear combination of multiple features:

\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n
\]

Here:
- \( x_1, x_2, ..., x_n \) = independent variables  
- \( \beta_1, \beta_2, ..., \beta_n \) = regression coefficients (feature weights)  
- \( \beta_0 \) = intercept  

---


## ✅ **Advantages**
- Uses multiple features, leading to more accurate predictions.  
- Easy to implement and interpret using Scikit-learn.  
- Provides insight into **feature importance** through coefficients.  
- Works well when the relationship between variables is approximately linear.  
- More informative than simple linear regression.

---

## ❌ **Disadvantages**
- Assumes a linear relationship between all features and target.  
- Performance decreases if dataset contains outliers or noise.  
- Requires a large dataset for stable and meaningful coefficient estimation.  
- Not suitable for capturing complex non-linear patterns.

---

## 📘 Lab 6 – Decision Tree Algorithm Implementation

### 📌 **Overview**
Lab 6 explores the **Decision Tree Algorithm**, a powerful and intuitive supervised learning method used for both **classification** and **regression**.  
Its structure is similar to a flowchart and enables automated decision-making based on feature values.

### 📌 **Aim**
To study and implement the Decision Tree Algorithm and evaluate its performance on a classification dataset.

---

## 📚 **Theory Summary**

### 🔹 **What is a Decision Tree?**
A Decision Tree is a hierarchical model consisting of:

- **Root Node** → The topmost node where the first split happens  
- **Internal Nodes** → Represent decisions based on feature conditions  
- **Branches** → Paths connecting nodes  
- **Leaf Nodes** → Final output labels (class prediction)

---

### 🔹 **How it Works**
Decision Trees follow a **divide-and-conquer** strategy:

1. Select the best feature to split the data (using metrics like Gini or Entropy).  
2. Recursively split subsets until:
   - Classes become pure  
   - Maximum depth or stopping criteria is reached  

This is a **greedy algorithm**, meaning it selects the best split at each step without backtracking.

---

### 🔹 **Visualization**
The manual includes a graphical example (like surfing decisions based on environmental factors).  
Using Scikit-learn, trees can be visualized with:

```python
tree.plot_tree(classifier)
```
---
## ✅ Advantages
- Easy to understand and interpret (resembles human decision-making).  
- Handles numerical and categorical features.  
- Requires minimal data preprocessing (no scaling or normalization).  
- Works for both classification and regression.  
- Can capture non-linear relationships effectively.  

---

## ❌ Disadvantages
- Highly prone to overfitting, especially with deep trees.  
- Slight variations in data can drastically change the tree (high variance).  
- Not ideal for very large datasets compared to ensemble methods.  
- Decision boundaries may be unstable and overly complex.  
- Does not generalize as well as Random Forests or Gradient Boosting.  
---

## 📘 Lab 7 – Random Forest Algorithm Implementation

### 📌 **Overview**
Lab 7 focuses on implementing the **Random Forest Algorithm**, a powerful ensemble-based supervised machine learning technique used for both **classification** and **regression** tasks.  
Random Forest improves performance by combining the results of multiple decision trees, making it more accurate and robust than a single tree.

### 📌 **Aim**
To study and implement the Random Forest Algorithm and evaluate its performance on a dataset using classification metrics.

---

## 📚 **Theory Summary**

### 🔹 **What is Random Forest?**
Random Forest is an **ensemble learning method** that builds multiple decision trees and aggregates their outputs to produce a more reliable final prediction.

- For **Classification** → Majority Voting  
- For **Regression** → Average of predictions  

The algorithm introduces randomness by:
1. Selecting random subsets of training data (Bootstrap sampling)  
2. Selecting random subsets of features for splitting nodes  

This reduces overfitting and increases generalization.

---

### 🔹 **How it Works**
1. Input training dataset  
2. Create multiple bootstrap samples  
3. Train a **Decision Tree** on each sample  
4. Combine the outputs of all trees:
   - **Voting** (Classification)  
   - **Averaging** (Regression)  
5. Produce the final prediction  

The manual includes a diagram demonstrating:
- Training Set → Multiple Tree Subsets → Individual Tree Predictions → Final Aggregated Output

---

## ✅ Advantages
- Reduces overfitting compared to single decision trees.  
- Provides higher accuracy and robustness.  
- Works well with large, high-dimensional datasets.  
- Handles missing values and noisy data effectively.  
- Suitable for both classification and regression.  
- Provides feature importance for interpretability.  

---

## ❌ Disadvantages
- More computationally expensive than a single tree.  
- Training and prediction can be slow with large numbers of trees.  
- Harder to interpret compared to a single decision tree.  
- Memory usage increases with number of trees.  
- May not perform well on very high-dimensional sparse data compared to linear models.  
---

## 📘 Lab 8 – Naive Bayes Classification Algorithm Implementation

### 📌 **Overview**
Lab 8 focuses on the implementation of the **Naive Bayes Classification Algorithm**, a probabilistic supervised learning method widely used for classification tasks — especially in **text classification**, spam filtering, and sentiment analysis.

### 📌 **Aim**
To study and implement the Naive Bayes Classification Algorithm and evaluate its performance on a classification dataset.

---

## 📚 **Theory Summary**

### 🔹 **What is Naive Bayes?**
Naive Bayes is a **generative learning algorithm** that models the distribution of data for each class and then applies **Bayes’ Theorem** to predict class membership.

It contrasts with *discriminative* models (like logistic regression) that directly learn the boundary between classes.

---

### 🔹 **Bayes' Theorem**
The algorithm is based on:

\[
P(Y|X) = \frac{P(X \text{ and } Y)}{P(X)}
\]

Where:
- **Prior Probability** → Initial probability of an event  
- **Posterior Probability** → Updated probability after observing new evidence  

Naive Bayes assumes **feature independence**, hence the name **“naive.”**

---
## ✅ Advantages
- Simple, fast, and easy to implement.  
- Works extremely well for text classification and high-dimensional data.  
- Requires very little training data compared to other classifiers.  
- Handles continuous, categorical, and binary data.  
- Performs surprisingly well despite the independence assumption.  

---

## ❌ Disadvantages
- Not ideal for datasets with highly correlated features.  
- Probability outputs may be unreliable when feature independence is violated.  
- Performs poorly when class distributions overlap significantly.  
---

## 📘 Lab 9 – K-Nearest Neighbor (K-NN) Algorithm Implementation

### 📌 **Overview**
Lab 9 focuses on implementing the **K-Nearest Neighbor (K-NN)** algorithm — a simple, intuitive, and instance-based supervised learning method. K-NN is used primarily for **classification**, where predictions are made based on the closest data points in the feature space.

### 📌 **Aim**
To study and implement the K-NN Classification Algorithm and understand how distance-based decisions are made for predicting class labels.

---

## 📚 **Theory Summary**

### 🔹 **What is K-NN?**
K-NN is a **non-parametric**, **lazy learning** algorithm.  
It does not build a model but makes decisions based on the **K nearest neighbors** in the dataset.

### 🔹 **How It Works**
When a new data point arrives:
1. Compute distances between the point and all training points  
2. Select the **K nearest neighbors**  
3. Perform **majority voting** among their class labels  
4. Assign the most common class to the new point  

---

### 🔹 **Choosing the K-value**
- **Small K (e.g., K = 1 or 3)**  
  - Highly flexible model  
  - Prone to **overfitting**  
  - Complex decision boundaries

- **Large K (e.g., K = 10 or 20)**  
  - Smooth, generalized boundaries  
  - Risk of **underfitting**

Selecting an appropriate K-value is essential for balanced performance.

---

### 🔹 **Core Steps of the K-NN Algorithm**
1. Import K-NN model  
2. Create feature (X) and target (y) variables  
3. Split data into training & testing sets  
4. Instantiate the K-NN model using `n_neighbors = K`  
5. Train the classifier  
6. Predict outcomes for test data  

---

## ✅ Advantages
- Simple, easy to understand, and intuitive.  
- No training time required (lazy learner).  
- Works well for non-linear decision boundaries.  
- Effective when data is evenly distributed and well-labeled.  
- Adapts naturally as new data is added.  

---

## ❌ Disadvantages
- Slow prediction time for large datasets (computes distance from all points).  
- Highly sensitive to irrelevant or noisy features.  
- Requires feature scaling (distance-based algorithm).  
- Does not work well with high-dimensional data (curse of dimensionality).  
- Performance heavily depends on selecting the right K-value.  
---

## 📘 Lab 10 – Support Vector Machine (SVM) Algorithm Implementation

### 📌 **Overview**
Lab 10 focuses on implementing the **Support Vector Machine (SVM)** algorithm — a highly effective supervised learning technique used for **classification**, **regression**, and **outlier detection**.  
SVM is powerful because it constructs the **optimal separating hyperplane** between different classes.

### 📌 **Aim**
To study and implement the SVM Algorithm and understand how margins, hyperplanes, and kernel functions contribute to accurate classification.

---

## 📚 **Theory Summary**

### 🔹 **What is SVM?**
Support Vector Machine finds the **best possible decision boundary** (hyperplane) that separates data points of different classes with maximum margin.

### 🔹 **Key Concepts**

#### **1. Hyperplane**
A decision boundary that separates different class regions.  
In 2D → A line  
In 3D → A plane  
In higher dimensions → A hyperplane  

#### **2. Support Vectors**
The closest data points to the hyperplane.  
They determine:
- The position of the hyperplane  
- The margin width  

#### **3. Margin**
The distance between the hyperplane and the nearest support vectors.  
SVM’s goal is to **maximize the margin**, which improves generalization.

#### **4. Hard Margin vs Soft Margin**
- **Hard Margin** → No misclassification allowed  
- **Soft Margin** → Allows some misclassification using slack variables (ideal for real-world noisy data)

#### **5. Kernels**
SVM uses kernel functions to handle non-linear data by mapping it to a higher-dimensional space:
- `linear` → Linearly separable data  
- `poly` → Polynomial decision boundary  
- `rbf` → Radial Basis Function for complex nonlinear boundaries  
- `sigmoid` → Neural-network-like behavior  

The RBF kernel is widely used for classification tasks.

---
## ✅ Advantages
- Works very well on high-dimensional datasets.  
- Effective for linearly and non-linearly separable data.  
- Uses kernel trick to handle complex nonlinear boundaries.  
- Robust to overfitting, especially with proper regularization.  
- Performs well even with small training datasets.  

---

## ❌ Disadvantages
- Training time can be slow for very large datasets.  
- Requires careful selection of kernel and hyperparameters.  
- Not ideal for datasets with heavy noise/overlapping classes.  
- Hard to interpret compared to decision trees.  
- Memory-intensive for large sample sizes.  
---

## ✨ Key Highlights
- All experiments use one consistent dataset  
- Both Classification & Regression versions implemented  
- Clear code structuring  
- Clean plots & performance metrics  

---

## 👨‍💻 Author
**Chetan Shabadi**

---

## ⭐ Support
If this repo helped you, don’t forget to **star the repository** ⭐  
Let others find it useful too!  
