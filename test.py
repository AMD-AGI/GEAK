# Initialize environment
import os, sys

# Auto-detect paths
TUTORIAL_DIR = os.getcwd() if os.getcwd().endswith('tutorial') else os.path.join(os.getcwd(), 'tutorial')
GEAK_DIR = os.path.dirname(TUTORIAL_DIR)
SRC_DIR = os.path.join(GEAK_DIR, 'geak_agent')

# Add geak_agent to path (so src/utils doesn't get shadowed)
if GEAK_DIR not in sys.path:
    sys.path.insert(0, GEAK_DIR)
if TUTORIAL_DIR not in sys.path:
    sys.path.insert(0, TUTORIAL_DIR)

# Import tutorial utilities
from tutorial_utils import setup_environment, print_header
TUTORIAL_DIR, SRC_DIR, CORPUS_PATH, TutorialDataloader = setup_environment()
print_header('✓ GEAK-Agent Tutorial Ready')


# ═══════════════════════════════════════════════════════════════
#                       🔑 API CONFIGURATION
# ═══════════════════════════════════════════════════════════════

USE_Cambricon_API = False                            # Set False for OpenAI
Cambricon_API_KEY = 'Your Cambricon API key'                # Your Cambricon API key
OPENAI_API_KEY = ''              # Your OpenAI key

# ───────────────────────────────────────────────────────────────
API_KEY = Cambricon_API_KEY if USE_Cambricon_API else OPENAI_API_KEY
MODEL_ID = 'claude-sonnet-4' if USE_Cambricon_API else 'gpt-4o'
print(f'✓ API: {"Cambricon" if USE_Cambricon_API else "OpenAI"} | Model: {MODEL_ID}')


#from geak_agent.models.Claude import ClaudeModel
from geak_agent.models.OpenAI import StandardOpenAIModel
from geak_agent.agents.GaAgent import GaAgent

# Load kernels
dataset = TutorialDataloader(
    kernel_names=['embedding_triton_kernel.py'],
    corpus_path=CORPUS_PATH
)

# Initialize model & agent
model = StandardOpenAIModel(api_key=API_KEY, model_id=MODEL_ID)

agent = GaAgent(model=model, dataset=dataset, corpus_path=CORPUS_PATH, descendant_num=3)

print(f'✓ Loaded {len(dataset)} kernels | Agent ready with {len(agent.memories)} tasks')

# Preview kernel instructions (optional)
from tutorial_utils import display_kernel_info
display_kernel_info(dataset.problem_states)


from tutorial_utils import print_config

OUTPUT_DIR = os.path.join(TUTORIAL_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#                       ⚙️ CONFIGURATION
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    'iteration_num': 1,       # Optimization iterations
    'temperature': 1.0,       # LLM creativity (0.0-2.0)
    'descendant_num': 3,      # Candidates per iteration
    'ancestor_num': 5,        # Reference solutions
    'gpu_id': 0,              # GPU device
    'target_mlu': 'MLU590',   # Target Cambricon GPU
}
print_config(CONFIG, '🚀 Running GEAK-Agent')


# Run the agent
agent.run(
    output_path=os.path.join(OUTPUT_DIR, 'tutorial_results.jsonl'),
    multi_thread=False,
    **CONFIG,
    start_iter=0, descendant_debug=1, profiling=True, start_idx=0
)
print('\n✅ Complete!')

from tutorial_utils import load_results, display_results_summary, display_generated_code

results, iteration = load_results(OUTPUT_DIR)
display_results_summary(results, iteration)


# View generated code
display_generated_code(results)

