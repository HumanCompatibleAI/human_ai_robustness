CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l coordination_ring -a_n final_neurips_agents/cring/cring_1tom_ns -n 30
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l coordination_ring -a_n final_neurips_agents/cring/cring_20tom_ns -n 30
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l coordination_ring -a_n final_neurips_agents/cring/cring_1bc_ns -n 30
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l coordination_ring -a_n final_neurips_agents/cring/cring_20bc_ns -n 30
CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l coordination_ring -a_n final_neurips_agents/cring/cring_20mixed_ns -n 30
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l coordination_ring -a_n final_neurips_agents/cring/cring_20mixed_s -n 30
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l counter_circuit -a_n final_neurips_agents/cc/cc_1tom_ns -n 30
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l counter_circuit -a_n final_neurips_agents/cc/cc_20tom_ns -n 30
CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l counter_circuit -a_n final_neurips_agents/cc/cc_1bc_ns -n 30
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l counter_circuit -a_n final_neurips_agents/cc/cc_20bc_ns -n 30
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l counter_circuit -a_n final_neurips_agents/cc/cc_20mixed_ns -n 30
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l counter_circuit -a_n final_neurips_agents/cc/cc_20mixed_s -n 30

