# Dimension Insensitive Euclidean Metric
Unofficial Torch implementation for the paper Dimension Insensitive Euclidean Metric (DIEM).<br><br/>

Run diem_loss.py to test it, for now the output values for the orthogonal and normal tensors are not matching the proportions of the paper, I'm unsure if its related to the distribution of the Tensors.

Fell free to report any issue or pull request.
## Install torch
```
pip install torch
```

## Usage example
```python
import torch

# Dimension of your Tensors.
dim = 128

# Get Expected values and standard deviation for "M" random Tensors of dimension "n"
E_d, sigma_squared = compute_expected_distance_and_variance(n=dim, M=10000)

# Computes the DIEM loss between two normal random Tensors, Input and Target
input = torch.normal(mean=0, std=1, size=(1, dim))
target = torch.normal(mean=0, std=1, size=(1, dim))

diem = diem_loss(
        Input=input,
        Target=target,
        E_d=E_d,
        sigma_squared=sigma_squared,
        v_max=3.0, #Maxium value in your Tensors, you can use torch.max()
        v_min=-3.0 #Minimum value in your Tensors, you can use torch.min()
)

print(diem) # Returns a float
```

```bibtex
@misc{tessari2024diem,
      title={Surpassing Cosine Similarity for Multidimensional Comparisons: Dimension Insensitive Euclidean Metric (DIEM)}, 
      author={Federico Tessari and Neville Hogan},
      year={2024},
      eprint={2407.08623},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
