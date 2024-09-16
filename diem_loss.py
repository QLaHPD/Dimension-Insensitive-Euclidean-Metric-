import torch


def compute_expected_distance_and_variance(n: int, v_min: float = -3.0, v_max: float = 3.0, M: int = 100000):
    """
    Computes the expected Euclidean distance E[d(n)] and variance sigma(n)^2
    for random Tensors of dimension n.

    Parameters:
    - n (int): Dimension of the Tensors.
    - v_min (float): Minimum value of the vector elements.
    - v_max (float): Maximum value of the vector elements.
    - M (int): Number of samples to generate.

    Returns:
    - E_d (torch.Tensor): Expected Euclidean distance.
    - sigma_squared (torch.Tensor): Variance of the Euclidean distances.
    """

    random_a = torch.normal(mean=0, std=1, size=(M, n))
    random_b = torch.normal(mean=0, std=1, size=(M, n))

    random_a = torch.clamp(input=random_a, min=v_min, max=v_max)
    random_b = torch.clamp(input=random_b, min=v_min, max=v_max)

    distances = torch.norm(random_a - random_b, dim=1)

    E_d = torch.mean(distances)
    sigma_squared = torch.var(distances, unbiased=True)

    return E_d, sigma_squared


def diem_loss(Input: torch.Tensor,
              Target: torch.Tensor,
              E_d: torch.Tensor,
              sigma_squared: torch.Tensor,
              v_min: float = -3.0,
              v_max: float = 3.0):
    """
    Computes the DIEM loss between two Tensors, Input and Target.
    v_min and v_max assume the values within ~99.7% of a normal distribution.

    Parameters:
    - Input (torch.Tensor): First Tensor.
    - Target (torch.Tensor): Second Tensor.
    - E_d (torch.Tensor): Expected Euclidean distance.
    - sigma_squared (torch.Tensor): Variance of Euclidean distance.
    - v_min (float): Minimum value of the vector elements.
    - v_max (float): Maximum value of the vector elements.

    Returns:
    - diem (torch.Tensor): DIEM loss value.
    """

    assert Input.shape == Target.shape, "Tensors 'Input' and 'Target' must have the same dimensions."

    d = torch.norm(Input - Target)

    diem = ((v_max - v_min) / sigma_squared) * (d - E_d)

    return diem


if __name__ == '__main__':
    dim = 102  # Dimension of the Tensors

    # Compute expected distance and variance once
    E_d, sigma_squared = compute_expected_distance_and_variance(dim)
    print(f"Expected Distance (E_d): {E_d.item():.4f}")
    print(f"Variance (sigma_squared): {sigma_squared.item():.4f}")
    print("---------------------------------------------------")

    # Test Case 1: Input and Target are equal
    input_equal = torch.normal(mean=0, std=1, size=(1, dim))
    input_equal = torch.clamp(input_equal, min=-3.0, max=3.0)
    target_equal = input_equal.clone()  # Exactly the same tensor

    diem_equal = diem_loss(Input=input_equal,
                           Target=target_equal,
                           E_d=E_d,
                           sigma_squared=sigma_squared,
                           v_max=3.0,
                           v_min=-3.0)

    print("Test Case 1: Input and Target are equal")
    print(f"DIEM Loss: {diem_equal.item():.4f}")
    print("---------------------------------------------------")

    # Test Case 2: Input and Target are random normal tensors
    input_random = torch.normal(mean=0, std=1, size=(1, dim))
    target_random = torch.normal(mean=0, std=1, size=(1, dim))

    input_random = torch.clamp(input_random, min=-3.0, max=3.0)
    target_random = torch.clamp(target_random, min=-3.0, max=3.0)

    diem_random = diem_loss(Input=input_random,
                            Target=target_random,
                            E_d=E_d,
                            sigma_squared=sigma_squared,
                            v_max=3.0,
                            v_min=-3.0)

    print("Test Case 2: Input and Target are random normal tensors")
    print(f"DIEM Loss: {diem_random.item():.4f}")
    print("---------------------------------------------------")

    # Test Case 3: Input and Target are orthogonal
    # Generate an orthogonal vector using Gram-Schmidt process
    # Since generating truly orthogonal high-dimensional random Tensors is non-trivial,
    # we'll create two Tensors and make them orthogonal.

    # Start with a random vector
    input_orthogonal = torch.normal(mean=0, std=1, size=(1, dim))
    input_orthogonal = torch.clamp(input_orthogonal, min=-3.0, max=3.0)

    # Generate another random vector
    random_vector = torch.normal(mean=0, std=1, size=(1, dim))
    random_vector = torch.clamp(random_vector, min=-3.0, max=3.0)

    # Make target orthogonal to input
    projection = (torch.dot(input_orthogonal.flatten(), random_vector.flatten()) /
                  torch.dot(input_orthogonal.flatten(), input_orthogonal.flatten())) * input_orthogonal
    target_orthogonal = random_vector - projection
    target_orthogonal = torch.clamp(target_orthogonal, min=-3.0, max=3.0)

    # Verify orthogonality
    dot_product = torch.dot(input_orthogonal.flatten(), target_orthogonal.flatten())
    print(f"Dot product (should be close to 0): {dot_product.item():.4f}")

    diem_orthogonal = diem_loss(Input=input_orthogonal,
                                Target=target_orthogonal,
                                E_d=E_d,
                                sigma_squared=sigma_squared,
                                v_max=3.0,
                                v_min=-3.0)

    print("Test Case 3: Input and Target are orthogonal")
    print(f"DIEM Loss: {diem_orthogonal.item():.4f}")
    print("---------------------------------------------------")

    # Test Case 4: Input and Target are opposites
    input_opposite = torch.normal(mean=0, std=1, size=(1, dim))
    input_opposite = torch.clamp(input_opposite, min=-3.0, max=3.0)
    target_opposite = -input_opposite  # Exact opposite

    # Ensure target is within bounds after negation
    target_opposite = torch.clamp(target_opposite, min=-3.0, max=3.0)

    diem_opposite = diem_loss(Input=input_opposite,
                              Target=target_opposite,
                              E_d=E_d,
                              sigma_squared=sigma_squared,
                              v_max=3.0,
                              v_min=-3.0)

    print("Test Case 4: Input and Target are opposites")
    print(f"DIEM Loss: {diem_opposite.item():.4f}")
    print("---------------------------------------------------")
