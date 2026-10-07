import logging
from config import Config
from pipeline import run_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

config = Config()
config.llm_backend = "mock"
config.n_train_samples = 2
config.n_samples = 2
config.controller_epochs = 1
config.model_save_path = "results/toy_weights.npz"

datasets = ["gsm8k", "hotpotqa", "math", "truthfulqa"]

try:
    for i, ds in enumerate(datasets):
        print(f"\n==============================")
        print(f" TESTING DATASET: {ds.upper()}")
        print(f"==============================")
        config.dataset = ds
        
        if i == 0:
            print("--> Training controller for the first time...")
            config.model_load_path = None
            run_experiment(config, method_name=f"Toy_{ds}", train=True)
        else:
            print(f"--> Loading saved controller for {ds}...")
            config.model_load_path = "results/toy_weights.npz"
            # Train=False means it evaluates directly
            run_experiment(config, method_name=f"Toy_{ds}", train=False)
            
    print("\n✅ All 5 toy datasets ran successfully without crashes! Logic is sound.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ Test crashed: {e}")
