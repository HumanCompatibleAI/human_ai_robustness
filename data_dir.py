# 1. Make a copy of this file and re-name it to `data_dir.py`
# 2. Customize the next line to import the data path for the project you are working on
#    For an example of how to do so, check out:
#       https://github.com/HumanCompatibleAI/human_model_theory/blob/micah_dev/human_ai_theory/__init__.py
#
# NOTE: the `data_dir.py` file will be automatically gitignored so as to not overwrite the ones
# of people working on other projects

from human_ai_robustness import PROJECT_DIR

DATA_DIR = PROJECT_DIR + "/data/"