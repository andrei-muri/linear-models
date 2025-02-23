import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

M_INIT = 0
B_INIT = 0
EPOCH = 800
ERR = 0.01
LEARNING_RATE = 0.01

def get_gradient(points, m, b):
    # The gradient is the vector containing the partial derivatives for each parameter
    # We use it to move through the input space to get to a minimum
    x = points["x"].values
    y = points["y"].values
    n = len(points)
    
    error = y - (m * x + b)
    # Partial derivatives of the loss function
    m_gradient = (-2/n) * np.sum(x * error)
    b_gradient = (-2/n) * np.sum(error)
    
    return (m_gradient, b_gradient)

def get_loss(points, m, b):
    # The loss function which is the Variance around the fit line
    # We want to minimize it
    x = points["x"]
    y = points["y"]
    n = len(points)
    loss = np.sum((y - (m*x + b)) ** 2) / n

    return loss

def get_variance(points):
    # the variance is a metric that tells how far apart are the values from their mean
    # the formula is (1/n) * Sum((y - mean)^2)
    mean = points["y"].mean()
    y = points["y"]
    n = len(points)
    variance = np.sum((y - mean) ** 2) / n

    return variance

def get_r_squared(points, m, b):
    # Here R^2 is calculated using variances instead of SS. This approach yields the same result
    r_squared = 0
    mean_variance = get_variance(points)
    fit_variance = get_loss(points, m, b)

    if mean_variance != 0:
        r_squared = 1 - (fit_variance / mean_variance)
    
    return r_squared


def get_parameters(points):
    # Choose starting values for the parameters
    m = M_INIT
    b = B_INIT

    for i in range(EPOCH):
        # compute the gradient
        m_gradient, b_gradient = get_gradient(points, m , b)
        #if we are close enough to the desired minimum stop
        if abs(m_gradient) <= ERR and abs(b_gradient) <=ERR:
            print("Error small enough")
            return (m, b)
        # Now gradient descent:
        m -= LEARNING_RATE * m_gradient
        b -= LEARNING_RATE * b_gradient

    return (m, b)



def main():
    data = pd.read_csv("data/lin.csv", header=None, names=["x", "y"])
    m, b = get_parameters(data)
    # calculate r squared
    r_squared = get_r_squared(data, m, b)


    plt.figure(figsize=(8, 6), num="Linear Relationship Plot")

    plt.scatter(data["x"], data["y"], color="red", marker="o", alpha=0.7, label="Data Points")\
    
    #Draw the best-fit line
    x_range = np.array([data["x"].min(), data["x"].max()]) 
    y_range = m * x_range + b  
    plt.plot(x_range, y_range, color="blue", linewidth=2, label=f"Best Fit Line: y = {m:.2f}x + {b:.2f}")

    plt.xlabel("X values", fontsize=12, fontweight="bold")
    plt.ylabel("Y values", fontsize=12, fontweight="bold")
    plt.title("Scatter Plot of X vs Y with Linear Trend", fontsize=14)

    plt.text(0.05, 0.70, f"$R^2 = {r_squared:.4f}$", fontsize=12, transform=plt.gca().transAxes,
         bbox=dict(facecolor="white", alpha=0.5))


    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()