# Iris Flower Classifier

This project is an implementation of Logistic Regression with Gradient Descent and the One-vs-Rest strategy for Multiclass Classification, built from scratch with NumPy and Pandas. It leverages the [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris) from [UCI Machine Learning repository](https://archive.ics.uci.edu/) to perform precise species classification based on the sepal and petal characteristics of iris flowers.
The trained model's weights are then employed to serve predictions through a website powered by FastAPI and HTMX.

## Preview

https://github.com/Vyvy-vi/iris-flower-classifier/assets/62864373/4d32b28f-fd88-45f9-b937-7fedd195de77


## Live Version

This page is currently deployed. [View the live website.](https://iris-flower-ml.vercel.app/)

## Implementation details
### Features given in the dataset:
- Petal length
- Petal width
- Sepal length
- Sepal width

### Model used:
- `m` - training examples, `w` - weights vector, `b` - bias
- Logistic Regression Model:
  $$f_{w,b}(x) = g(\textbf{w} . \textbf{x} + b)$$
    where `g` is the sigmoid function given by:
  $$g(z) = \frac{1}{e^{-z}}$$
- Cost function for logistic regression:
  $$\textbf{J}(\textbf{w}, b) = \frac{1}{m} \sum_{i = 0}^{m - 1}  \left( loss(f_{\textbf{w}, b}(\textbf{x}^{(i)}), y^{(i)}) \right)$$
    where the loss function is the cost for a single data point and is given by:
  $$loss(f_{\textbf{w},b}(x^{(i)}),y^{(i)}) = \left(−y^{(i)}\log(f_{\textbf{w},b}(\textbf{x}^{(i)})\right) − \left((1 − y^{(i)})\log(1 - f_{\textbf{w}, b}(\textbf{x}^{(i)}))\right)$$
    where $f_{\textbf{w},b}(\textbf(x)^{(i)})$ is the model's prediction and $y^{(i)}$ is the actual label
  $$f_{\textbf{w},b}(\textbf{x}^{(i)}) = g(\textbf{w} . \textbf{x}^{(i)} + b)$$
 

## Setup
- Clone this project:
  ```
  git clone https://github.com/Vyvy-vi/iris-flower-classifier/
  ```
- [Install Python3](https://www.python.org/downloads/)
- Install dependencies
  ```
  pip install -r requirements.txt
  ```
- Run Jupyter Notebook
  ```
  jupyter notebook
  ```
- Run application
  ```
  python3 main.py
  ```

## Usage

- Training the Model: To train the model and generate the weights and bias, run
  the classification-logistic-regression-from-scratch.ipynb Jupyter notebook.
(run the `jupyter notebook` command) 
- Running the Web Application: Execute the web application using `python main.py`. This starts the web server, making the prediction service available at http://localhost:8000.
- Making Predictions: Input sepal and petal measurements via the web interface and receive predictions for the iris flower species.


## Feedback and Bugs

If you have feedback or a bug report, please feel free to [open a GitHub issue](https://github.com/Vyvy-vi/iris-flower-classifier/issues/new/choose)!

## License

This software is licensed under [The MIT License](LICENSE).

Copyright 2023 Vyom Jain.

