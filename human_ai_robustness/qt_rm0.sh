CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_1tom_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_20tom_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_1bc_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_20bc_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_20mixed_ns -n 50 -nv 5
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_20mixed_s -n 50 -nv 5
CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_100tom_s -n 50 -nv 5
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_500tom_s -n 50 -nv 5
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_5tom_s -n 50 -nv 5
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_20tomrand_s -n 50 -nv 5
CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_1bc_s -n 50 -nv 5
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_20bc_s -n 50 -nv 5
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_1tom_s -n 50 -nv 5
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l large_room -a_f final_neurips_agents/rm/ -a_n rm_20tom_s -n 50 -nv 5
