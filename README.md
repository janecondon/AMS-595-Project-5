# AMS 595 Project 5: Page Rank Algorithm, PCA, Linear Regression, and Gradient Descent

---

## Overview

This project implements four data analysis and machine learning algorithms:

1. **PageRank Algorithm** – Rank web pages based on importance/link structure.  
2. **Principal Component Analysis (PCA)** – Perform dimensionality reduction on height and weight data.  
3. **Linear Regression** – Predict house prices using least squares regression model.  
4. **Gradient Descent** – Minimize a loss function using gradient descent.  

---

## 1. PageRank Algorithm

We implement the PageRank algorithm to rank websites based on how many links point to them. A small web network is represented by the following probability matrix \(M\), where \(M[i,j]\) is the probability that a user on page \(j\) clicks a link to page \(i\):

$$
M = \begin{bmatrix}
0 & 0 & 1/2 & 0 \\
1/3 & 0 & 0 & 1/2 \\
1/3 & 1/2 & 0 & 1/2 \\
1/3 & 1/2 & 1/2 & 0
\end{bmatrix}
$$


### Steps:

- Compute the dominant eigenvector of \(M\).  
- Iterate until convergence (difference between consecutive rank vectors < \(10^{-6}\)).  
- Normalize the eigenvector so that the sum of ranks equals 1.  

**Result:** Pages 3 and 4 have the highest PageRank scores (~0.316), indicating higher long-term probability of visits.

---

## 2. Dimensionality Reduction via PCA

We apply PCA to a dataset of 100 standardized height and weight measurements:

$$
\text{Data} = \begin{bmatrix}
h_1 & w_1 \\
h_2 & w_2 \\
\vdots & \vdots \\
h_{100} & w_{100}
\end{bmatrix}
$$

### Steps:

1. Compute the covariance matrix.  
2. Perform eigenvalue decomposition.  
3. Identify principal components:

- First PC: \([0.621, 0.784]\) – explains ~56% of variance.  
- Second PC: \([0.784, 0.621]\) – explains ~44% of variance.  

4. Project the data onto the first principal component to reduce dimensionality to 1D.  

**Observation:** There is a positive correlation between height and weight.

---

## 3. Linear Regression via Least Squares

We use linear regression to predict house prices based on square footage, number of bedrooms, and age:

$$
X =
\begin{bmatrix}
2100 & 3 & 20 \\
2500 & 4 & 15 \\
1800 & 2 & 30 \\
2200 & 3 & 25
\end{bmatrix},
\quad
y =
\begin{bmatrix}
460 \\
540 \\
330 \\
400
\end{bmatrix}
$$

### Steps:

- Solve $X \beta = y$ using least squares to obtain regression coefficients.  
- Predict house prices for new examples.  

**Notes:**  
- Square footage positively impacts price.  
- Number of bedrooms and age have negative coefficients, though small sample size and scaling may affect interpretation.  
- Using `scipy.linalg.lstsq` is preferred for flexibility over `solve`.

---

## 4. Gradient Descent for Minimizing Loss Function

Given a matrix $X \in \mathbb{R}^{100 \times 50}$ and a target matrix A, we minimize the mean squared error loss:

$$
f(X) = \frac{1}{2} \sum_{i,j} (X_{i,j} - A_{i,j})^2, \quad \nabla f(X) = X - A
$$

### Steps:

- Define the loss function and gradient.  
- Implement gradient descent using `scipy.optimize.minimize`.  
- Stop when the change in loss $< 10^{-6}$
 or after 1000 iterations.  

**Result:**  
- The algorithm converges in 3 iterations.  
- Loss decreases monotonically and levels near zero.

---


## Usage

1. Clone the repository:  
   ```bash
   git clone <repo-url>
