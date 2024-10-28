import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()
x = np.linspace(0, 1, 100)
fx = 1.0/(1.0+np.exp(-30*(x-0.5)))
plt.plot(x, fx)
plt.xlabel("Market Odds")
plt.ylabel("Felt Odds")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Felt Odds v.s. Market Odds")
plt.xticks([0.0, 0.5, 1.0])
plt.yticks([0.0, 1.0])
plt.grid(True)
plt.savefig(os.path.join(os.getenv("HK_FLOW_FILE", "."), "manifold.png"))
print(1.0/(1.0+np.exp(-30*(0.56-0.5))))