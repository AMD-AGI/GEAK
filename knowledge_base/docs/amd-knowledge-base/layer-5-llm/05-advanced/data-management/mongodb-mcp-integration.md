---
layer: "5"
category: "advanced"
subcategory: "data-management"
tags: ["mongodb", "mcp", "database", "data-analysis", "experiments"]
rocm_version: "7.0+"
last_updated: 2025-11-04
---

# MongoDB MCP Integration for AI/ML Workflows

Query MongoDB databases directly from your IDE for experiment tracking,
GPU monitoring data, and training metrics analysis.

## Quick Setup

```bash
amd-ai-devtool setup --mongodb-url mongodb://localhost:27017/experiments
```

## Use Cases for AMD GPU Development

### 1. Training Metrics Storage

Store ML experiment results:

```python
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client.ml_experiments

db.training_runs.insert_one({
    'experiment_id': 'exp-001',
    'model': 'llama-3-8b',
    'gpu': 'MI300X',
    'accuracy': 0.94,
    'duration_hours': 12.5
})

# Query via Cursor:
# "Show me training runs on MI300X with accuracy > 0.9"
```

### 2. GPU Monitoring Integration

Log AMD SMI data during training:

```python
import amdsmi
from pymongo import MongoClient

db = MongoClient('mongodb://localhost:27017/').gpu_monitoring

def log_gpu_metrics(experiment_id):
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_processor_handles()
    
    for i, device in enumerate(devices):
        db.metrics.insert_one({
            'experiment_id': experiment_id,
            'gpu_id': i,
            'utilization': amdsmi.amdsmi_get_gpu_activity(device)['gfx_activity'],
            'temperature': amdsmi.amdsmi_get_temp_metric(device, 
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT) / 1000
        })
    
    amdsmi.amdsmi_shut_down()

# Query via Cursor:
# "Show GPU temperatures for experiment exp-001"
```

### 3. Experiment Comparison

Compare training runs:

```python
# Store multiple runs
for run_id in range(5):
    db.training_runs.insert_one({
        'run_id': run_id,
        'hyperparameters': {'lr': 0.001 * (2 ** run_id), 'batch_size': 32},
        'final_accuracy': 0.85 + (run_id * 0.02),
        'training_time_hours': 10 - run_id
    })

# Query via Cursor:
# "Compare training runs by learning rate and show accuracy vs training time"
```

## Connection Examples

```bash
# Local development
amd-ai-devtool setup --mongodb-url mongodb://localhost:27017/dev

# Production with auth
amd-ai-devtool setup --mongodb-url mongodb://admin:secret@prod:27017/ml_prod

# MongoDB Atlas
amd-ai-devtool setup --mongodb-url "mongodb+srv://user:pass@cluster.mongodb.net/analytics"

# Combined with GitHub MCP
amd-ai-devtool setup --github-token ghp_xxx --mongodb-url mongodb://localhost:27017
```

## Example Queries to Ask Your IDE

Once configured, you can ask:
- "Show me all documents in the training_runs collection"
- "Count experiments where GPU temperature exceeded 80°C"
- "Find the top 5 training runs by accuracy"
- "List all collections in the experiments database"
- "Show me GPU utilization patterns for experiment exp-001"

## Integration with Training Loops

### PyTorch Training Example

```python
import amdsmi
import torch
from pymongo import MongoClient
from datetime import datetime

class ExperimentLogger:
    def __init__(self, experiment_id, mongodb_url='mongodb://localhost:27017/'):
        self.experiment_id = experiment_id
        self.db = MongoClient(mongodb_url).ml_experiments
        
        # Initialize AMD SMI
        amdsmi.amdsmi_init()
        self.devices = amdsmi.amdsmi_get_processor_handles()
    
    def log_epoch(self, epoch, metrics):
        """Log epoch metrics with GPU stats."""
        # Get GPU metrics
        gpu_metrics = []
        for i, device in enumerate(self.devices):
            util = amdsmi.amdsmi_get_gpu_activity(device)
            temp = amdsmi.amdsmi_get_temp_metric(device,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT)
            power = amdsmi.amdsmi_get_power_info(device)
            
            gpu_metrics.append({
                'gpu_id': i,
                'utilization': util['gfx_activity'],
                'temperature': temp / 1000,
                'power_watts': power['current_socket_power'] / 1000000
            })
        
        # Store in MongoDB
        self.db.training_logs.insert_one({
            'experiment_id': self.experiment_id,
            'epoch': epoch,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'gpu_stats': gpu_metrics
        })
    
    def __del__(self):
        amdsmi.amdsmi_shut_down()

# Usage in training loop
logger = ExperimentLogger('exp-llama3-8b-001')

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, dataloader, optimizer)
    val_accuracy = validate(model, val_dataloader)
    
    logger.log_epoch(epoch, {
        'train_loss': train_loss,
        'val_accuracy': val_accuracy
    })
```

### Query Training Progress via Cursor

After logging, ask your IDE:
- "Show me the training loss trend for experiment exp-llama3-8b-001"
- "What was the GPU temperature during epoch 10?"
- "Compare GPU utilization across all training runs"

## Best Practices

1. **Index frequently queried fields**:
   ```python
   db.training_runs.create_index('experiment_id')
   db.training_runs.create_index('timestamp')
   ```

2. **Use separate databases for dev/prod**:
   ```bash
   # Development
   amd-ai-devtool setup --mongodb-url mongodb://localhost:27017/ml_dev
   
   # Production
   amd-ai-devtool setup --mongodb-url mongodb://prod:27017/ml_prod
   ```

3. **Regular cleanup of old experiments**:
   ```python
   from datetime import datetime, timedelta
   
   # Delete experiments older than 30 days
   cutoff = datetime.now() - timedelta(days=30)
   db.training_runs.delete_many({'timestamp': {'$lt': cutoff}})
   ```

4. **Aggregate metrics for faster queries**:
   ```python
   # Store daily summaries
   pipeline = [
       {'$group': {
           '_id': '$experiment_id',
           'avg_accuracy': {'$avg': '$metrics.accuracy'},
           'total_epochs': {'$sum': 1}
       }}
   ]
   daily_summary = list(db.training_logs.aggregate(pipeline))
   db.daily_summaries.insert_many(daily_summary)
   ```

## See Also

- [MongoDB MCP Setup Guide](../../../../docs/MCP_GUIDE.md#mongodb-mcp-integration) - Complete configuration guide
- [AMD SMI Usage](../../../layer-2-compute-stack/rocm-systems/amd-smi-usage.md) - GPU monitoring with AMD SMI
- [Training Optimization](../../03-training/optimization/memory-optimization.md) - Optimize training performance

---

*MongoDB MCP integration enables seamless database queries from your IDE, making experiment tracking and analysis effortless.*

