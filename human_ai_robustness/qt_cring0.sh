CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l coordination_ring -a_f final_neurips_agents/cring/ -a_n cring_1tom_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l coordination_ring -a_f final_neurips_agents/cring/ -a_n cring_20tom_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l coordination_ring -a_f final_neurips_agents/cring/ -a_n cring_1bc_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l coordination_ring -a_f final_neurips_agents/cring/ -a_n cring_20bc_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l coordination_ring -a_f final_neurips_agents/cring/ -a_n cring_20mixed_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l coordination_ring -a_f final_neurips_agents/cring/ -a_n cring_20mixed_s -n 50 -nv 5
