from agents.Base import SequentialBaseAgent, BaseAgent
from utils.utils import clear_code
from prompts import prompt_for_reflection
from memories.Memory import ReflexionMemory
from models.Base import BaseModel



class Reflexion(SequentialBaseAgent):

    def __init__(self, model: BaseModel, dataset):
        self.model = model
        self.dataset = dataset
        self.memories = self.memory_init()
    
    def memory_init(self):
        return [ReflexionMemory(ps) for ps in self.dataset.problem_states]

    def run_single_pass(self, mem: ReflexionMemory, verbose=False):
        if mem.ps.pass_call:
            return
        # generate solution
        text = mem.ps.instruction

        if mem.ps.solution:
            text += f"\nPrevious attempt implementation:{mem.ps.solution}"
        
        if mem.err_msg:
            text += f"\nTest messages for previous attempt:{mem.err_msg}"
        
        if mem.reflection:
            text += f"\nReflection on previous attempt:{mem.reflection}"

        text += "Please output the codes only without explanation, which we can run directly."
        msg = [
            {"role": "user", "content": text},
        ]
        response = self.model.generate(msg)
        mem.ps.solution = clear_code(response)

        # run script on gpu
        is_pass, err_msg = self.dataset.run_single_call(mem.ps)

        # generate reflection
        if not is_pass:
            mem.err_msg = err_msg
            reflect_txt = prompt_for_reflection.prompt.format(
                problem=mem.ps.instruction,
                solution=mem.ps.solution,
                test_result=err_msg
            )
            reflect_msg = [
                {
                    "role": "user",
                    "content": reflect_txt
                }
            ]
            mem.reflection = self.model.generate(reflect_msg)
        # else:
        #     mem.ps.pass_call = True

        return

