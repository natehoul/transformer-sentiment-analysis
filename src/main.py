from pathlib import Path
import sys


# Package management BS
src_dir = Path(__file__).parent
module_dirs = ['datasets', 'models', 'training']
module_dirs = [str(Path(src_dir, module)) for module in module_dirs]

for module in module_dirs:
    sys.path.insert(1, module)


from training.train import train


if __name__ == "__main__":
    train('binary_music_lr02_batch128_epochs2_debugging')