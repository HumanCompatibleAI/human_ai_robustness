CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l counter_circuit -a_n bc -n 50 -nv 5
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l coordination_ring -a_n bc -n 50 -nv 5
CUDA_VISIBLE_DEVICES=2 python qualitative_robustness_expt.py -l bottleneck -a_n bc -n 50 -nv 5
CUDA_VISIBLE_DEVICES=3 python qualitative_robustness_expt.py -l large_room -a_n bc -n 50 -nv 5
CUDA_VISIBLE_DEVICES=0 python qualitative_robustness_expt.py -l centre_objects -a_n bc -n 50 -nv 5
CUDA_VISIBLE_DEVICES=1 python qualitative_robustness_expt.py -l centre_pots -a_n bc -n 50 -nv 5
