from torch.utils.cpp_extension import load

interpolate_ext = load(name="three_interpolate",
                       extra_include_paths=["three_interpolate/src/include"],
                       sources=["three_interpolate/src/three_interpolate_cuda.hip", "three_interpolate/src/three_interpolate.cpp"],
                       verbose=True)


