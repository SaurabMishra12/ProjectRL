import os
import json

abstranet_dir = r"c:\research work 25-05-26\LLM evolve\abstranet_mvp"
projectrl_dir = r"c:\Users\msaur\OneDrive\Desktop\projectRL"

files_to_read = [
    r"src\reasoning_graph.py",
    r"src\cognitive_state.py",
    r"src\semantic_extractor.py",
    r"src\ppo_controller.py",
    r"src\reward_engine.py",
    r"src\self_critic.py"
]

refactored_code = ""
for f in files_to_read:
    path = os.path.join(abstranet_dir, f)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            # Clean imports
            lines = content.split('\n')
            cleaned = []
            for line in lines:
                if line.startswith("import ") or line.startswith("from ") or line.startswith('"""'):
                    continue
                cleaned.append(line)
            refactored_code += "\n" + "\n".join(cleaned) + "\n"

# Now we inject this into build_kaggle.py
build_script_path = os.path.join(projectrl_dir, "build_kaggle.py")
with open(build_script_path, "r", encoding="utf-8") as file:
    build_script = file.read()

# We will inject the refactored_code before the checkpoint_code
insertion_point = 'checkpoint_code = """'

if insertion_point in build_script and "PPOHypergraphActorCritic" not in build_script:
    new_build_script = build_script.replace(
        insertion_point, 
        f'giant_code += """\n{refactored_code}\n"""\n\n' + insertion_point
    )
    
    # Also update the execution_block to use PPOMetaController
    new_build_script = new_build_script.replace('controller = IBController(config)', 'controller = PPOMetaController(num_agents=config.n_agents)')
    new_build_script = new_build_script.replace('frozen_controller = IBController(config)', 'frozen_controller = PPOMetaController(num_agents=config.n_agents)')
    
    with open(build_script_path, "w", encoding="utf-8") as file:
        file.write(new_build_script)
    print("Injected refactored code into build_kaggle.py")
else:
    print("Refactored code already injected or insertion point not found.")
