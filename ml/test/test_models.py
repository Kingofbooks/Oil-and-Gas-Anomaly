import torch

from ml.model import TranADNetwork


batch_size = 8
window_size = 120
num_features = 22


x = torch.randn(
    batch_size,
    window_size,
    num_features,
)


model = TranADNetwork(
    input_size=num_features,
)


output = model(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nModel:")
print(model)