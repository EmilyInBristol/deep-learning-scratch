import torch
import torch.nn as nn

# Scaled Dot-Product Attention mechanism
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1 / (d_k ** 0.5)

    def forward(self, Q, K, V):
        # Compute the attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get the attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Compute the final output (weighted sum of the values)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output, attention_weights

# Create a simple example
"""
d_model = 4  # Model dimensionality
Q = torch.rand(1, 5, d_model, requires_grad=True)  # Query matrix
K = torch.rand(1, 5, d_model, requires_grad=True)  # Key matrix
V = torch.rand(1, 5, d_model, requires_grad=True)  # Value matrix

# Initialize attention
attention = ScaledDotProductAttention(d_k=d_model)

# Forward pass
output, attention_weights = attention(Q, K, V)

# Loss and backward pass
loss = output.mean()
loss.backward()  # Calculate gradients for Q, K, V

# Display gradients
print("Gradient of Q:", Q.grad)
print("Gradient of K:", K.grad)
print("Gradient of V:", V.grad)
"""

input_ids = torch.randint(0, 10000, (1, 20))
print(input_ids)