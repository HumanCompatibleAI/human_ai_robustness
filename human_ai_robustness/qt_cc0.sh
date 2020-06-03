CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l counter_circuit -a_f final_neurips_agents/cc/ -a_n cc_1tom_ns -n 30 -nv 5
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l counter_circuit -a_f final_neurips_agents/cc/ -a_n cc_20tom_ns -n 30 -nv 5
CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l counter_circuit -a_f final_neurips_agents/cc/ -a_n cc_1bc_ns -n 30 -nv 5
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l counter_circuit -a_f final_neurips_agents/cc/ -a_n cc_20bc_ns -n 30 -nv 5
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l counter_circuit -a_f final_neurips_agents/cc/ -a_n cc_20mixed_ns -n 30 -nv 5
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l counter_circuit -a_f final_neurips_agents/cc/ -a_n cc_20mixed_s -n 30 -nv 5
