import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create large tensors and perform computations
a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

start = time.time()
c = torch.matmul(a, b)
end = time.time()

print(f"Matrix multiplication took {end - start:.2f} seconds")
