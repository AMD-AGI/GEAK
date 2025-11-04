from torch.utils.cpp_extension import load

assign_score_withk_ext = load(name="assign_score_withk",
                              extra_include_paths=["assign_score_withk/src/include"],
                              sources=["assign_score_withk/src/assign_score_withk_cuda.hip", "assign_score_withk/src/assign_score_withk.cpp"],
                              verbose=True)


