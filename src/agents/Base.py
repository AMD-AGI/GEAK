import os
import json
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.Base import BaseModel
from dataloaders.ProblemState import ProblemState
from memories.Memory import BaseMemory



class BaseAgent:

    def __init__(self, model: BaseModel, dataset):
        self.model = model
        self.dataset = dataset
        self.memories = self.memory_init()

    def memory_init(self):
        return [BaseMemory(ps) for ps in self.dataset.problem_states]
    def run_single_pass(self, mem: BaseMemory, verbose=False, temperature=0):
        pass

    def run(self, output_path=None, multi_thread=True, verbose=False, datalen=None, mem_path=None, temperature=0):
        data_len = datalen if datalen else len(self.dataset)
        with tqdm(total=data_len) as pbar:
            if multi_thread:
                thread_num = 3
                
                with ThreadPoolExecutor(max_workers=thread_num) as executor:
                    futures = {executor.submit(self.run_single_pass, mem, temperature): mem for mem in self.memories[:data_len]}
                    for future in as_completed(futures):
                        pbar.update(1)
            else:
                for mem in self.memories[:data_len]:
                    self.run_single_pass(mem, verbose, temperature=temperature)
                    pbar.update(1)
                    
        
        if output_path is not None:
            self.dataset.write_file(output_path)
        
        if mem_path is not None:
            self.write_memories(mem_path)
    
    def write_memories(self, file_path):
        with open(file_path, "w") as f:
            for mem in self.memories:
                output = asdict(mem)
                f.write(json.dumps(output) + "\n")

class SequentialBaseAgent(BaseAgent):    
    def run(self, output_path=None, multi_thread=True, verbose=False, datalen=None, iteration_num=0, temperature=0):
        data_len = datalen if datalen else len(self.dataset)
        
        for iter in range(iteration_num):
            logger.info(f"\n=== Iteration {iter} ===")
            root, extension = os.path.splitext(output_path)
            iter_path = f"{root}_{iter}{extension}"
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    thread_num = 3
                    
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.run_single_pass, mem, temperature): mem for mem in self.memories[:data_len]}
                        for future in as_completed(futures):
                            pbar.update(1)
                        # list(tqdm(executor.map(self.run_single_pass, self.dataset.problem_states[:data_len], [verbose]*data_len)), total=data_len)
                else:
                    for mem in self.memories[:data_len]:
                        self.run_single_pass(mem, verbose, temperature=temperature)
                        pbar.update(1)
                        
            
            if output_path is not None:
                self.dataset.write_file(iter_path)