import numpy as np

# --------------------------
# 1. Dummy input (5x5 image)
# --------------------------
x = np.array([
    [3, 0, 1, 2, 7],
    [1, 5, 8, 9, 3],
    [2, 7, 2, 5, 1],
    [0, 1, 3, 1, 7],
    [4, 2, 1, 6, 2]
])

# --------------------------
# 2. Single 3x3 filter
# --------------------------
kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, -1]
])

# --------------------------
# 3. Convolution
# --------------------------
def conv2d(img, ker):
    H, W = img.shape
    k = ker.shape[0]
    out = np.zeros((H - k + 1, W - k + 1))

    for i in range(H - k + 1):
        for j in range(W - k + 1):
            region = img[i:i+k, j:j+k]
            out[i, j] = np.sum(region * ker)
    return out

conv_out = conv2d(x, kernel)
print("Conv output:\n", conv_out)

# --------------------------
# 4. ReLU
# --------------------------
relu_out = np.maximum(0, conv_out)
print("\nReLU output:\n", relu_out)

# --------------------------
# 5. Max Pool (2x2, stride=2)
# --------------------------
def max_pool(img):
    H, W = img.shape
    out = np.zeros((H//2, W//2))
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            out[i//2, j//2] = np.max(img[i:i+2, j:j+2])
    return out

pool_out = max_pool(relu_out)
print("\nPool output:\n", pool_out)

# --------------------------
# 6. Flatten
# --------------------------
flat = pool_out.flatten()
print("\nFlatten:", flat)

# --------------------------
# 7. Fully connected layer
# --------------------------
W = np.random.randn(3, len(flat))
b = np.random.randn(3)

fc_out = W @ flat + b
print("\nFC output:", fc_out)

# --------------------------
# 8. Softmax
# --------------------------
probs = np.exp(fc_out) / np.sum(np.exp(fc_out))
print("\nSoftmax probabilities:", probs)
