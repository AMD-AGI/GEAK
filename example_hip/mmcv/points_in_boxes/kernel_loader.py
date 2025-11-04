from torch.utils.cpp_extension import load

points_in_boxes_ext = load(name="points_in_boxes",
                           extra_include_paths=["points_in_boxes/src/include"],
                           sources=["points_in_boxes/src/points_in_boxes_cuda.hip", "points_in_boxes/src/points_in_boxes.cpp"],
                           verbose=True)


