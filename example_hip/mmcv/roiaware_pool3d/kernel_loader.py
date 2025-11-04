from torch.utils.cpp_extension import load

roiaware_pool3d_ext = load(name="roiaware_pool3d",
                           extra_include_paths=["roiaware_pool3d/src/include"],
                           sources=["roiaware_pool3d/src/roiaware_pool3d_kernel.cu", "roiaware_pool3d/src/roiaware_pool3d.cpp"],
                           verbose=True)


