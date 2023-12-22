import numpy as np

print("Hello world!")

# mình có S (n*n), X là symetric
# G >= 0, X >= 0
# S = X * X^T
np.random.seed(42)

G = np.array([[1., 1., 1., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 0., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 1., 1., 1.]])

def update(G,X):
    numerator = G @ X # Top
    denominator = 2 * X @ X.T @ X  # Bottom
    newX = X * (1/2 + (numerator / denominator))

    return newX

def get_loss(G,X):
    tar = G - X @ X.T
    loss = np.linalg.norm(tar) ** 2
    
    return loss


X = np.random.rand(5, 5) # Start
iter = 10000

######### Main function
print('Start X: ')
print(X)
print('Start loss: ', get_loss(G,X))

for i in range(iter):
    print(f'>> Iter {i}')
    X = update(G,X)
    print('Updated X: ')
    print(X)
    print('Loss: ', get_loss(G,X))



print(X @ X.T)