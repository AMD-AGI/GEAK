---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "best-practices"
category: "testing"
subcategory: "gpu-testing"
tags: ["testing", "gpu", "validation", "quality-assurance"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "40min"
---

# GPU Testing Best Practices

Comprehensive guide to testing GPU-accelerated applications on AMD hardware.

## Unit Testing

### Basic GPU Tests

```python
import unittest
import torch

class TestGPUOperations(unittest.TestCase):
    def setUp(self):
        """Ensure GPU is available"""
        self.assertTrue(torch.cuda.is_available(), "GPU not available")
        self.device = torch.device("cuda")
    
    def test_tensor_creation(self):
        """Test basic tensor operations"""
        x = torch.randn(100, 100, device=self.device)
        self.assertEqual(x.device.type, "cuda")
        self.assertEqual(x.shape, (100, 100))
    
    def test_matrix_multiplication(self):
        """Test matrix operations"""
        a = torch.randn(100, 100, device=self.device)
        b = torch.randn(100, 100, device=self.device)
        c = torch.matmul(a, b)
        
        self.assertEqual(c.shape, (100, 100))
        self.assertEqual(c.device.type, "cuda")
    
    def test_model_inference(self):
        """Test model forward pass"""
        model = torch.nn.Linear(100, 10).to(self.device)
        x = torch.randn(32, 100, device=self.device)
        
        output = model(x)
        self.assertEqual(output.shape, (32, 10))
    
    def tearDown(self):
        """Clean up GPU memory"""
        torch.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main()
```

### Testing with pytest

```python
import pytest
import torch

@pytest.fixture
def gpu_device():
    """Fixture for GPU device"""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return torch.device("cuda")

@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Auto cleanup after each test"""
    yield
    torch.cuda.empty_cache()

def test_gpu_memory_allocation(gpu_device):
    """Test memory allocation"""
    initial_memory = torch.cuda.memory_allocated()
    
    # Allocate tensor
    x = torch.randn(1000, 1000, device=gpu_device)
    
    allocated_memory = torch.cuda.memory_allocated()
    assert allocated_memory > initial_memory
    
    # Free tensor
    del x
    torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated()
    assert final_memory == initial_memory

def test_multi_gpu(gpu_device):
    """Test multi-GPU operations"""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    
    # Create tensors on different GPUs
    x = torch.randn(100, 100, device="cuda:0")
    y = torch.randn(100, 100, device="cuda:1")
    
    assert x.device.index == 0
    assert y.device.index == 1

@pytest.mark.slow
def test_large_model_inference(gpu_device):
    """Test large model (marked as slow)"""
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        device_map=gpu_device
    )
    
    input_ids = torch.randint(0, 1000, (1, 10), device=gpu_device)
    output = model(input_ids)
    
    assert output.logits.shape[0] == 1
```

Run tests:
```bash
# All tests
pytest test_gpu.py

# Skip slow tests
pytest test_gpu.py -m "not slow"

# Verbose output
pytest test_gpu.py -v

# With coverage
pytest test_gpu.py --cov=my_module
```

## Integration Testing

### LLM Inference Testing

```python
import pytest
from vllm import LLM, SamplingParams

@pytest.fixture(scope="module")
def llm_model():
    """Load model once for all tests"""
    return LLM(model="gpt2", dtype="bfloat16")

def test_single_inference(llm_model):
    """Test single prompt inference"""
    sampling_params = SamplingParams(max_tokens=50)
    outputs = llm_model.generate(["Hello world"], sampling_params)
    
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0

def test_batch_inference(llm_model):
    """Test batch inference"""
    prompts = [f"Prompt {i}" for i in range(10)]
    sampling_params = SamplingParams(max_tokens=50)
    
    outputs = llm_model.generate(prompts, sampling_params)
    
    assert len(outputs) == 10
    for output in outputs:
        assert len(output.outputs[0].text) > 0

def test_different_parameters(llm_model):
    """Test various sampling parameters"""
    test_cases = [
        {"temperature": 0.7, "top_p": 0.9},
        {"temperature": 1.0, "top_k": 50},
        {"temperature": 0.0},  # Greedy
    ]
    
    for params in test_cases:
        sampling_params = SamplingParams(**params, max_tokens=20)
        outputs = llm_model.generate(["Test"], sampling_params)
        assert len(outputs) == 1
```

### Training Testing

```python
def test_training_step():
    """Test single training step"""
    model = torch.nn.Linear(100, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    x = torch.randn(32, 100).cuda()
    y = torch.randint(0, 10, (32,)).cuda()
    
    output = model(x)
    loss = criterion(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0
    assert not torch.isnan(loss)

def test_training_convergence():
    """Test that model learns"""
    # Simple XOR problem
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
        torch.nn.Sigmoid()
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    # Training data
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).cuda()
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).cuda()
    
    # Train
    initial_loss = None
    for epoch in range(1000):
        output = model(X)
        loss = criterion(output, y)
        
        if epoch == 0:
            initial_loss = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1  # Loss should decrease significantly
```

## Performance Testing

### Throughput Testing

```python
import time

def test_inference_throughput():
    """Measure inference throughput"""
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="gpt2", dtype="bfloat16")
    prompts = ["Test prompt"] * 100
    sampling_params = SamplingParams(max_tokens=100)
    
    # Warmup
    llm.generate(prompts[:10], sampling_params)
    
    # Benchmark
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    
    print(f"Throughput: {throughput:.2f} tokens/s")
    
    # Assert minimum throughput
    assert throughput > 100  # At least 100 tokens/s

def test_latency():
    """Measure inference latency"""
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="gpt2")
    sampling_params = SamplingParams(max_tokens=50)
    
    latencies = []
    for _ in range(10):
        start = time.time()
        llm.generate(["Test"], sampling_params)
        latency = time.time() - start
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency: {avg_latency:.3f}s")
    
    # Assert maximum latency
    assert avg_latency < 1.0  # Less than 1 second
```

### Memory Testing

```python
def test_memory_usage():
    """Monitor GPU memory usage"""
    import torch
    
    initial_memory = torch.cuda.memory_allocated()
    max_memory = torch.cuda.get_device_properties(0).total_memory
    
    # Load model
    model = torch.nn.Linear(1000, 1000).cuda()
    
    model_memory = torch.cuda.memory_allocated() - initial_memory
    memory_percent = (model_memory / max_memory) * 100
    
    print(f"Model memory: {model_memory/1e9:.2f}GB ({memory_percent:.1f}%)")
    
    # Assert memory within bounds
    assert memory_percent < 90  # Less than 90% of GPU memory

def test_memory_leak():
    """Check for memory leaks"""
    import torch
    
    initial_memory = torch.cuda.memory_allocated()
    
    # Perform operations
    for _ in range(100):
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated()
    
    # Memory should return to initial state
    assert abs(final_memory - initial_memory) < 1e6  # Less than 1MB difference
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/gpu-tests.yml
name: GPU Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: [self-hosted, gpu]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.2
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Check GPU
      run: |
        python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
        rocm-smi
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v -m "not slow"
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Docker Testing

```dockerfile
# Dockerfile.test
FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_2.1.1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install pytest pytest-cov

COPY . .

CMD ["pytest", "tests/", "-v", "--cov=src"]
```

Run:
```bash
docker build -f Dockerfile.test -t test-image .
docker run --device=/dev/kfd --device=/dev/dri \
    --group-add video test-image
```

## Test Organization

### Directory Structure

```
tests/
├── unit/
│   ├── test_models.py
│   ├── test_inference.py
│   └── test_training.py
├── integration/
│   ├── test_pipeline.py
│   └── test_api.py
├── performance/
│   ├── test_throughput.py
│   └── test_latency.py
├── fixtures/
│   ├── models.py
│   └── data.py
└── conftest.py
```

### conftest.py

```python
# tests/conftest.py
import pytest
import torch

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available"""
    return torch.cuda.is_available()

@pytest.fixture(autouse=True)
def reset_gpu():
    """Reset GPU state before each test"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Best Practices

1. **Always clean up GPU memory** after tests
2. **Use fixtures** for expensive setups (model loading)
3. **Mark slow tests** so they can be skipped
4. **Test edge cases**: OOM, invalid inputs, etc.
5. **Monitor memory** and check for leaks
6. **Use CI/CD** for automated testing
7. **Test on target hardware** (AMD GPUs)

## References

- [pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing](https://pytorch.org/docs/stable/testing.html)
- [ROCm Testing Guide](https://rocm.docs.amd.com/)

