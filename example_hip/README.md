# HIP2HIP Example Repository

This folder contains the HIP2HIP example showcased in the upcoming GEAK-HIP blog (to be released).

## Instruction file 

There is a file `FromRe_instructions.json`, which includes:

```json
{
  "task": "task name / id for identification",
  "instruction": "general instruction for the LLM (modifiable)",
  "file_path": "directory where compile and execution commands should be run",
  "file_name": "target file to be optimized"
}
```

### Field Details

`task` – identifier for the optimization or example  
`instruction` – general instruction passed to the LLM  
`file_path` – directory where compilation and execution occur  
`file_name` – target file to be optimized  
- If in `file_path`, just use the filename (e.g., `silu.hip`).  
- Otherwise, use a relative path from `file_path`.

Note: The `"output"` field is deprecated. Use the configuration setup to set output directories.

## Execution Logic

By default, the agent runs according to `FromRe_instructions.json` unless overridden in the configuration.  
All available example instructions are in `FromRe_instructions_FULL.json`. Copy the desired entry there into `FromRe_instructions.json` to run a case.

## Example Directories

### rocm-examples/Applications
Contains 6 toy examples (`bitsonic_sort`, `convolution`, etc.) showing typical HIP implementations.  
To build and run:
```bash
make
./application_<kernel_name>
```

### MMCV
Collection of kernels from public MMCV code. Main test files are `test_<kernel_name>.py`, which load kernels via pybind.  
Run:
```bash
python test_<kernel_name>.py
```

### point_to_voxel
Main file: `main.hip`  
Build and run:
```bash
make
./application_point_to_voxel
```

### silu
Main file: `silu.hip`  
Build and run:
```bash
make
./application_silu
```
