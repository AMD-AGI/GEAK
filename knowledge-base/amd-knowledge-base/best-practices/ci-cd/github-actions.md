---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "best-practices"
category: "ci-cd"
subcategory: "github-actions"
tags: ["ci-cd", "github-actions", "automation", "testing"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "35min"
---

# GitHub Actions for AMD GPU Projects

CI/CD pipelines for GPU-accelerated projects using GitHub Actions.

## Basic GPU Workflow

```yaml
# .github/workflows/gpu-ci.yml
name: GPU CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-gpu:
    runs-on: [self-hosted, amd-gpu]
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Verify GPU
      run: |
        rocm-smi
        python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
    
    - name: Install Dependencies
      run: |
        pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.2
        pip install -r requirements.txt
    
    - name: Run Tests
      run: |
        pytest tests/ -v --cov=src
    
    - name: Upload Coverage
      uses: codecov/codecov-action@v3
```

## Multi-Stage Pipeline

```yaml
name: Full CI/CD Pipeline

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Lint
      run: |
        pip install black flake8 mypy
        black --check .
        flake8 .
        mypy src/
  
  test-unit:
    runs-on: [self-hosted, amd-gpu]
    needs: lint
    steps:
    - uses: actions/checkout@v3
    - name: Unit Tests
      run: pytest tests/unit/ -v
  
  test-integration:
    runs-on: [self-hosted, amd-gpu]
    needs: test-unit
    steps:
    - uses: actions/checkout@v3
    - name: Integration Tests
      run: pytest tests/integration/ -v
  
  benchmark:
    runs-on: [self-hosted, amd-gpu]
    needs: test-integration
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v3
    - name: Run Benchmarks
      run: python benchmarks/run.py
```

## Docker Build and Push

```yaml
name: Docker Build

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Docker meta
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: myregistry/myimage
    
    - name: Login to Registry
      uses: docker/login-action@v2
      with:
        registry: myregistry.io
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}
    
    - name: Build and Push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
```

## Model Training Workflow

```yaml
name: Train Model

on:
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model to train'
        required: true
        default: 'llama-2-7b'

jobs:
  train:
    runs-on: [self-hosted, multi-gpu]
    steps:
    - uses: actions/checkout@v3
    
    - name: Train Model
      run: |
        python train.py --model ${{ github.event.inputs.model_name }}
    
    - name: Upload Model
      uses: actions/upload-artifact@v3
      with:
        name: trained-model
        path: outputs/
```

## Best Practices

1. Use self-hosted runners for GPU workloads
2. Cache dependencies to speed up builds
3. Run linting before GPU tests
4. Use matrix builds for multiple configurations
5. Set appropriate timeouts
6. Use secrets for credentials

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)

