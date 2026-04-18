# ONNX Runtime with ROCm - Getting Started

*Complete guide to running ONNX models on AMD GPUs using ROCm backend*

## Overview

ONNX Runtime is Microsoft's cross-platform, high-performance ML inferencing accelerator. With ROCm support, you can run ONNX models efficiently on AMD GPUs, enabling deployment of models trained in various frameworks (PyTorch, TensorFlow, etc.) with optimal AMD GPU acceleration.

## Features

- **Multi-Framework Support**: Run models from PyTorch, TensorFlow, scikit-learn
- **AMD GPU Acceleration**: Optimized ROCm execution provider
- **Production Ready**: Industry-grade performance and reliability
- **Language Bindings**: Python, C++, C#, Java, JavaScript
- **Model Optimization**: Graph optimization and quantization
- **Broad Operator Coverage**: 150+ built-in operators

## Installation

### Prerequisites
```bash
# Install ROCm first
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev

# Verify ROCm installation
rocminfo
```

### Python Installation
```bash
# Install ONNX Runtime with ROCm support
pip install onnxruntime-rocm

# Or install from source for latest features
git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime
./build.sh --config RelWithDebInfo --build_shared_lib --parallel \
  --use_rocm --rocm_home=/opt/rocm
```

### Verify Installation
```python
import onnxruntime as ort

# Check available providers
print("Available providers:", ort.get_available_providers())
# Should include 'ROCMExecutionProvider'

# Check ROCm devices
print("ROCm device count:", ort.get_device())
```

## Basic Usage

### Loading and Running Models
```python
import onnxruntime as ort
import numpy as np

# Create session with ROCm provider
providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('model.onnx', providers=providers)

# Get input/output info
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input: {input_name}, shape: {session.get_inputs()[0].shape}")
print(f"Output: {output_name}, shape: {session.get_outputs()[0].shape}")

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run([output_name], {input_name: input_data})
result = outputs[0]

print(f"Output shape: {result.shape}")
```

### Session Configuration
```python
# Configure ROCm execution provider
rocm_provider_options = {
    'device_id': 0,
    'arena_extend_strategy': 'kSameAsRequested',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
    'do_copy_in_default_stream': True,
}

session = ort.InferenceSession(
    'model.onnx',
    providers=[('ROCMExecutionProvider', rocm_provider_options)]
)
```

## Computer Vision Models

### Image Classification (ResNet)
```python
import onnxruntime as ort
import cv2
from PIL import Image
import torchvision.transforms as transforms

class ResNetONNX:
    def __init__(self, model_path):
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Convert to numpy
        input_array = input_tensor.numpy()
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_array})
        
        # Get prediction
        logits = outputs[0]
        predicted_class = np.argmax(logits, axis=1)[0]
        confidence = np.max(softmax(logits))
        
        return predicted_class, confidence

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Usage
classifier = ResNetONNX('resnet50.onnx')
class_id, confidence = classifier.predict('image.jpg')
print(f"Predicted class: {class_id}, Confidence: {confidence:.3f}")
```

### Object Detection (YOLO)
```python
import cv2

class YOLOv5ONNX:
    def __init__(self, model_path, img_size=640):
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.img_size = img_size
        
    def preprocess(self, img):
        # Resize and pad
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Pad to square
        img_padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img_padded[:new_h, :new_w] = img_resized
        
        # Normalize and transpose
        img_input = img_padded.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))  # HWC -> CHW
        img_input = np.expand_dims(img_input, 0)  # Add batch dim
        
        return img_input, scale
    
    def detect(self, image_path, conf_threshold=0.5):
        img = cv2.imread(image_path)
        input_tensor, scale = self.preprocess(img)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Post-process detections
        detections = outputs[0][0]  # Remove batch dimension
        
        boxes = []
        for detection in detections:
            confidence = detection[4]
            if confidence > conf_threshold:
                x, y, w, h = detection[:4]
                # Convert to original image coordinates
                x /= scale
                y /= scale
                w /= scale
                h /= scale
                
                class_id = int(np.argmax(detection[5:]))
                boxes.append({
                    'bbox': [x, y, w, h],
                    'class_id': class_id,
                    'confidence': confidence
                })
        
        return boxes

# Usage
detector = YOLOv5ONNX('yolov5s.onnx')
detections = detector.detect('image.jpg')
```

## Natural Language Processing

### BERT for Text Classification
```python
from transformers import AutoTokenizer

class BERTClassifierONNX:
    def __init__(self, model_path, tokenizer_name='bert-base-uncased'):
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = 512
    
    def predict(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='np',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Run inference
        input_feed = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        outputs = self.session.run(None, input_feed)
        logits = outputs[0]
        
        # Get prediction
        predicted_class = np.argmax(logits, axis=1)[0]
        probabilities = softmax(logits[0])
        
        return predicted_class, probabilities

# Usage
classifier = BERTClassifierONNX('bert_classifier.onnx')
class_id, probs = classifier.predict("This movie is fantastic!")
```

### Named Entity Recognition
```python
class BERTNerONNX:
    def __init__(self, model_path, tokenizer_name='bert-base-cased'):
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # NER labels (example for CoNLL-2003)
        self.labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 
                      'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    
    def predict(self, text):
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='np',
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )
        
        # Run inference
        input_feed = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        outputs = self.session.run(None, input_feed)
        predictions = np.argmax(outputs[0], axis=2)
        
        # Map predictions to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        entities = []
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions[0])):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                label = self.labels[pred_id]
                entities.append((token, label))
        
        return entities
```

## Model Optimization

### Graph Optimization
```python
# Enable graph optimizations
sess_options = ort.SessionOptions()

# Set optimization level
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Enable memory pattern optimization
sess_options.enable_mem_pattern = True

# Enable CPU-GPU memory arena optimization
sess_options.enable_cpu_mem_arena = True

# Create optimized session
session = ort.InferenceSession(
    'model.onnx',
    sess_options,
    providers=['ROCMExecutionProvider']
)
```

### Dynamic Quantization
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize model for faster inference
quantize_dynamic(
    model_input='model.onnx',
    model_output='model_quantized.onnx',
    weight_type=QuantType.QUInt8
)

# Load quantized model
session = ort.InferenceSession(
    'model_quantized.onnx',
    providers=['ROCMExecutionProvider']
)
```

### Batch Processing
```python
def batch_inference(session, inputs, batch_size=32):
    """Process inputs in batches for better GPU utilization"""
    results = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        
        # Pad batch if needed
        if len(batch) < batch_size:
            padding = [batch[-1]] * (batch_size - len(batch))
            batch.extend(padding)
        
        # Convert to numpy array
        batch_array = np.array(batch)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: batch_array})
        
        results.extend(outputs[0][:len(inputs[i:i + batch_size])])
    
    return results
```

## Performance Monitoring

### Profiling
```python
# Enable profiling
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

session = ort.InferenceSession(
    'model.onnx', 
    sess_options,
    providers=['ROCMExecutionProvider']
)

# Run inference
outputs = session.run(None, {input_name: input_data})

# Get profiling results
prof_file = session.end_profiling()
print(f"Profiling results saved to: {prof_file}")
```

### Memory Usage Monitoring
```python
def get_memory_usage():
    """Monitor GPU memory usage"""
    import subprocess
    
    result = subprocess.run(['rocm-smi', '-u'], 
                          capture_output=True, text=True)
    return result.stdout

# Monitor before and after inference
print("Before inference:")
print(get_memory_usage())

outputs = session.run(None, {input_name: input_data})

print("After inference:")
print(get_memory_usage())
```

## Multi-GPU Support

### Model Parallelism
```python
def create_multi_gpu_sessions(model_path, device_ids=[0, 1]):
    """Create sessions on multiple GPUs"""
    sessions = []
    
    for device_id in device_ids:
        provider_options = {'device_id': device_id}
        providers = [('ROCMExecutionProvider', provider_options)]
        
        session = ort.InferenceSession(model_path, providers=providers)
        sessions.append(session)
    
    return sessions

def distribute_inference(sessions, inputs):
    """Distribute inference across multiple GPUs"""
    batch_size = len(inputs) // len(sessions)
    results = []
    
    import concurrent.futures
    
    def run_inference(session, batch):
        input_name = session.get_inputs()[0].name
        return session.run(None, {input_name: batch})
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for i, session in enumerate(sessions):
            start_idx = i * batch_size
            if i == len(sessions) - 1:  # Last GPU gets remaining data
                batch = inputs[start_idx:]
            else:
                batch = inputs[start_idx:start_idx + batch_size]
            
            future = executor.submit(run_inference, session, batch)
            futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result()[0])
    
    return results
```

## C++ API Usage

### Basic C++ Implementation
```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

class ONNXInference {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    
public:
    ONNXInference(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference")
        , session(nullptr) {
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        
        // Add ROCm provider
        OrtROCMProviderOptions rocm_options{};
        rocm_options.device_id = 0;
        session_options.AppendExecutionProvider_ROCM(rocm_options);
        
        // Create session
        session = Ort::Session(env, model_path.c_str(), session_options);
        
        // Get input/output names
        size_t num_input_nodes = session.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            input_names.push_back(session.GetInputName(i, allocator));
        }
        
        size_t num_output_nodes = session.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            output_names.push_back(session.GetOutputName(i, allocator));
        }
    }
    
    std::vector<float> run_inference(const std::vector<float>& input_data,
                                   const std::vector<int64_t>& input_shape) {
        // Create input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input_data.data()), 
            input_data.size(), input_shape.data(), input_shape.size());
        
        // Run inference
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names.data(), input_tensors.data(),
                                        input_names.size(), output_names.data(),
                                        output_names.size());
        
        // Extract results
        float* float_array = output_tensors.front().GetTensorMutableData<float>();
        size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        
        return std::vector<float>(float_array, float_array + output_size);
    }
};

// Usage
int main() {
    try {
        ONNXInference inference("model.onnx");
        
        std::vector<float> input_data(3 * 224 * 224, 1.0f);  // Example input
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        
        auto results = inference.run_inference(input_data, input_shape);
        
        std::cout << "Inference completed. Output size: " << results.size() << std::endl;
        
    } catch (const Ort::Exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Troubleshooting

### Common Issues

#### Provider Not Available
```python
# Check if ROCm provider is available
available_providers = ort.get_available_providers()
if 'ROCMExecutionProvider' not in available_providers:
    print("ROCm provider not available. Available providers:", available_providers)
    print("Install onnxruntime-rocm or check ROCm installation")
```

#### Memory Issues
```python
# Reduce memory usage
sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = False  # Disable memory pattern optimization
sess_options.enable_cpu_mem_arena = False  # Disable memory arena

# Or limit GPU memory
provider_options = {
    'device_id': 0,
    'gpu_mem_limit': 1 * 1024 * 1024 * 1024  # 1GB limit
}

session = ort.InferenceSession(
    'model.onnx',
    sess_options,
    providers=[('ROCMExecutionProvider', provider_options)]
)
```

#### Performance Issues
```python
# Enable all optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

# Check model performance
import time

start_time = time.time()
outputs = session.run(None, {input_name: input_data})
end_time = time.time()

print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
```

## Best Practices

### Model Conversion
```python
# Convert PyTorch model to ONNX
import torch

def convert_pytorch_to_onnx(model, example_input, output_path):
    model.eval()
    
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
```

### Session Reuse
```python
# Create session once, reuse for multiple inferences
class ModelInference:
    def __init__(self, model_path):
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, input_data):
        return self.session.run(None, {self.input_name: input_data})[0]

# Use it
model = ModelInference('model.onnx')
for data in dataset:
    result = model.predict(data)
```

### Input Validation
```python
def validate_input(session, input_array):
    """Validate input shape and type"""
    input_info = session.get_inputs()[0]
    expected_shape = input_info.shape
    expected_type = input_info.type
    
    # Check shape (ignore batch dimension)
    if len(input_array.shape) != len(expected_shape):
        raise ValueError(f"Expected {len(expected_shape)} dimensions, got {len(input_array.shape)}")
    
    # Check data type
    if input_array.dtype != np.float32 and expected_type == 'tensor(float)':
        input_array = input_array.astype(np.float32)
    
    return input_array
```

## Resources

### Documentation
- [ONNX Runtime ROCm Provider](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [ROCm Documentation](https://rocmdocs.amd.com/)

### Related Guides
- [PyTorch with ROCm](../pytorch/pytorch-rocm-basics.md)
- [Model Deployment](../../layer-5-llm/02-inference/deployment/production-serving.md)

### Performance Optimization
- [GPU Optimization](../../best-practices/performance/gpu-optimization.md)
- [Memory Optimization](../../best-practices/performance/memory-optimization.md)

---
*Tags: onnx-runtime, rocm, model-inference, amd-gpu, cross-platform, optimization*
*Estimated reading time: 50 minutes*