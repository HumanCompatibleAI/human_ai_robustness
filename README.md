# Human-AI Robustness

This code can be used to reproduce the results in the paper [Evaluating and Improving the Robustness of Collaborative Agents](insert_link). *Note that this repository uses a specific older commits of the [human_ai_coord repository](https://github.com/HumanCompatibleAI/human_ai_coord) and sub-repos therein, and should not be expected to work with the current version of those repos (at the time of writing, the branch `pk-dev3` of each sub-repo was used)*.

For more information about the Overcooked-AI environment, check out [this](https://github.com/HumanCompatibleAI/overcooked_ai) repo.

## Installation

When cloning the repository, make sure you also clone the submodules:
```
$ git clone --recursive git@github.com:HumanCompatibleAI/human_ai_robustness.git
```

If you want to clone a specific branch with its submodules, use:
```
$ git clone --single-branch --branch BRANCH_NAME --recursive git@github.com:HumanCompatibleAI/human_ai_robustness.git
```

It is useful to setup a conda environment with Python 3.7:
```
$ conda create -n hair python=3.7
$ conda activate hair
```

To complete the installation, run:
```
               $ cd human_aware_rl/human_ai_coord
human_aware_rl/human_ai_coord $ ./install.sh
```

Then, from within `human_ai_robustness`, run `python setup.py develop`.


Next install tensorflow (the GPU **or** non-GPU version depending on your setup):
```
$ pip install tensorflow==1.13.1
```

```
$ pip install tensorflow-gpu==1.13.1
```

## Verify Installation

To verify your installation, you can try running the following command from the `human_ai_coord/human_aware_rl` folder:

```
python run_tests.py
```

Note that using tensorflow-gpu will not enable to pass this tests (and others) due to intrinsic randomness introduced by GPU computations. We recommend to first install tensorflow (non-GPU), run the desired tests, and then install tensorflow-gpu.

On OSX, you may run into an error saying that Python must be installed as a framework. You can fix it by [telling Matplotlib to use a different backend](https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/).

## Repo Structure Overview and Examples

Here we highlight the most important parts of the repo that relate to our paper [Evaluating and Improving the Robustness of Collaborative Agents](insert_link).

`human_ai_coord/human_aware_rl/ppo/ppo_pop.py`: train one agent with PPO in Overcooked with either a single fixed agent or a population of agents. For example, to run a local test of `ppo_pop.py` for a population of 1 theory-of-mind (ToM) model on the layout `Bottleneck`, run the following from within `human_ai_coord/human_aware_rl` (replace `"tom"` with `"bc_pop"` to train with a BC agent instead): 

```
python ppo/ppo_pop.py with LOCAL_TESTING=True layout_name="bottleneck" OTHER_AGENT_TYPE="tom" POP_SIZE=1
```

`human_ai_coord/human_aware_rl/robustness_expts/`: folder with experiment scripts used to generate experimental results in the paper

`human_ai_robustness/overcooked_interactive.py`: script to play Overcooked in the terminal against trained agents. For example, to play interactively with a ToM agent on layout `Bottleneck`:

```
python overcooked_interactive.py -t tom -l bottleneck
```

`human_ai_robustness/qualitative_robustness_expt.py`: run our suite of qualitative tests. For example, to run the qualitative tests with one of our trained ppo agents on the layout `Bottleneck`, run (the superscipt `s` in the agent name refers to using diverse starts):

```
python qualitative_robustness_expt.py -l bottleneck -a_f final_neurips_agents/example_bottleneck/ -a_n bot_20tom_s -nv 1”
```

`human_ai_robustness/agent.py`: the class ToMModel is the ToM model used throughout our results.

`human_ai_robustness/data/bc_runs/bc_runs.zip`: this contains all of the BC agents used as partners to the ppo agent for the reuslts in our [paper](insert_link).

`human_ai_robustness/data/final_trained_agents/final_trained_agents.zip`: this contains all of our trained ppo agents used in our [paper](insert_link).

# Reproducing results

All results can be reproduced by first running the `.sh` scripts under `human_ai_coord/human_aware_rl/robustness_expts/`. This will train all of the agents; to run said agents on the suite of qualitative tests, run the `.sh` scripts under `human_ai_robustness/qt_experiments/`.

