import numpy as np

def softmax(x):
    y = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - y)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_production(Q, K, V):
    multi_qk = np.dot(Q, K.T)
    d_k = K.shape[-1]
    scaled_attention_logits = multi_qk / np.sqrt(d_k)
    attention_weights = softmax(scaled_attention_logits)
    output = np.dot(attention_weights, V)
    return attention_weights, output

    
Q = np.array([[1, 0, 1],[0, 1, 0]])
K = np.array([[1, 0, 1],[0, 1, 0],[1, 1, 1]])
V = np.array([[10, 0], [0, 10], [5, 5]])

scaled_dot_production(Q, K, V)


