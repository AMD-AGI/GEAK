from agents.Base import BaseAgent
from utils.utils import clear_code

class DirectPrompt(BaseAgent):
    def run_single_pass(self, mem, verbose=False):
        text = mem.ps.instruction
        text += "Please output the codes only without explanation, which we can run directly."
        msg = [
            {"role": "user", "content": text},
        ]
        response = self.model.generate(msg)
        mem.ps.solution = clear_code(response)
        