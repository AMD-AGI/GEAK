from torch.utils.cpp_extension import load

interpolate_ext = load(name="three_nn",
                       extra_include_paths=["three_nn/src/include"],
                       sources=["three_nn/src/three_nn_cuda.hip", "three_nn/src/three_nn.cpp"],
                       verbose=True)


