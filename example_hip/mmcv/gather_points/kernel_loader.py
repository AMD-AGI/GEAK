from torch.utils.cpp_extension import load

gather_points_ext = load(name="gather_points",
                         extra_include_paths=["gather_points/src/include"],
                         sources=["gather_points/src/gather_points_cuda.cu", "gather_points/src/gather_points.cpp"],
                         verbose=True)


