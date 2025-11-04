from torch.utils.cpp_extension import load

knn_ext = load(name="knn",
               extra_include_paths=["knn/src/include"],
               sources=["knn/src/knn_cuda.hip", "knn/src/knn.cpp"],
               verbose=True)


