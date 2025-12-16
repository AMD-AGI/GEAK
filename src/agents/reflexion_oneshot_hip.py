# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import os
from tqdm import tqdm
from loguru import logger
from dataclasses import asdict
from agents.Reflexion import Reflexion
from utils.utils import extract_function_signatures, clear_code, extract_function_calls
from prompts import prompt_for_reflection_hip
from memories.Memory import MemoryClassMeta
from models.Base import BaseModel
from retrievers.retriever import BM25Retriever
from prompts import prompt_for_generation
from concurrent.futures import ThreadPoolExecutor, as_completed


class Reflexion_Oneshot_HIP(Reflexion):
    def __init__(self, model: BaseModel, dataset, corpus_path):
        self.model = model
        self.dataset = dataset
        self.memories = []

        self.memory_init()

    def memory_init(self):
        class Memory(metaclass=MemoryClassMeta, field_names=["ps", 
                                                             "err_msg", 
                                                             "reflection", 
                                                             "function_signatures", 
                                                             "oneshot",
                                                             "pass_call", 
                                                            ]):
            pass
        
        for ps in self.dataset.problem_states:
            tmp_mem = Memory(ps=ps, 
                             err_msg=None, 
                             reflection=None, 
                            #  function_signatures=fs_mem, 
                             function_signatures=None,
                             oneshot=ps.test_code,
                             pass_call=False,
                             )
            self.memories.append(tmp_mem)

