from torch.utils.cpp_extension import load

ball_query_ext = load(name="ball_query",
                      extra_include_paths=["ball_query/src/include"],
                      sources=["ball_query/src/ball_query_cuda.hip", "ball_query/src/ball_query.cpp"],
                      verbose=True)


