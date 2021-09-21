import numpy as np
import matplotlib.pyplot as plt
import torch
import march

a = torch.FloatTensor([[100, 300, 50], [200, 200, 60], [300, 100, 100]])
b = march.marchingSquare(a)
print(torch.max(b), torch.min(b))
plt.imshow(b)
plt.colorbar()
plt.show()