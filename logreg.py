import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

M_INIT = 0
B_INIT = 0
EPOCH = 1000
LEARNING_RATE = 0.01
ERR = 0.001

def sigmoid(m, b, x):
    return 1 / (1 + np.exp(-(m * x + b)))

def getGradients(data, m, b):
    x = data["x"]
    label = data["label"]
    curr_sigmoid = partial(sigmoid, m, b)

    m_gradient = np.sum(x * (curr_sigmoid(x) - label))
    b_gradient = np.sum(curr_sigmoid(x) - label)
    return m_gradient, b_gradient

def getParameters(data):
    m = M_INIT
    b = B_INIT

    for i in range(EPOCH):
        m_gradient, b_gradient = getGradients(data, m , b)

        if np.sqrt(m_gradient**2 + b_gradient**2) <= ERR:
            print("Error small enough")
            return m, b
        
        m -= LEARNING_RATE * m_gradient
        b -= LEARNING_RATE * b_gradient
    
    return m, b


def main():
    data = pd.read_csv("data/log.csv", header=None, names=["x", "label"])
    m, b = getParameters(data)

    plt.figure(figsize=(10, 6), num="Logistic Regression")

    cat0_data = data[data["label"] == 0]
    cat1_data = data[data["label"] == 1]

    plt.scatter(cat0_data["x"], cat0_data["label"], color="red", marker="o", label="Cat 0")
    plt.scatter(cat1_data["x"], cat1_data["label"], color="blue", marker="o", label="Cat 1")

    # now draw the sigmoid line
    x = np.linspace(0, 12, 50)
    y = sigmoid(m, b, x)

    new_input = 8.1
    if sigmoid(m, b, new_input) > 0.55:
        cat = "Category 1"
    elif sigmoid(m, b, new_input) < 0.45:
        cat = "Category 0"
    else:
        cat = "Uncertain" 

    plt.text(0.05, 0.80, f"$Category: {cat}$", fontsize=12, transform=plt.gca().transAxes,
         bbox=dict(facecolor="white", alpha=0.5))
    
    plt.plot(x, y, color="black", linewidth=2, label=f"Best sigmoid fit line m={m:.2f} and b={b:.2f}", linestyle="--", alpha=0.3)
    plt.scatter(new_input, sigmoid(m, b, new_input), color="purple", marker="*", label="New input", s=150)

    plt.xlabel("x")
    plt.ylabel("Probabilities")
    plt.grid(visible=True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()