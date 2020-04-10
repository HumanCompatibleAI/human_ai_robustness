
"""
Here we take data for the neurips agents on Cring playing the set of qualitative tests. We then do the following plotting:
    • Data:
        ◦ Should be 4 agents, with 5 seeds each
        ◦ Ignore test 2
    • Plot 1:
        ◦ Avg over seeds
        ◦ Plot each agent on a separate subplot (4 subplot)
        ◦ Plot the 6 tests in a bar chart
            ▪ Highlight test 5, which the TOM also does badly on
    • Plot 2:
        ◦ Avg over seeds
        ◦ Plot each test on a separate subplot (6 subplots – but highlight test 5)
        ◦ Plot the 4 agents in a bar chart
    • Plot 3:
        ◦ Learn how to do error bars!

"""
import numpy as np
import matplotlib.pyplot as plt

def take_train_only(data, agent_base_names):
    """Data is in this order: cc_1_mantom TRAIN SEED0, cc_1_mantom VAL SEED0, cc_1_mantom TRAIN SEED1,...
    But we want all the "Train" agents together. Here we rearrange the data"""
    rearranged_data = []
    for i, agent_base_name in enumerate(agent_base_names):
        num_seeds = 3 if agent_base_name is not 'cc_20_mixed' else 2
        selection_train = [0, 2, 4] if agent_base_name is not 'cc_20_mixed' else [0, 2]
        # selection_val = [1, 3, 5] if agent_base_name is not 'cc_20_mixed' else [1, 3]
        this_agent_data = [data[3*2*i + j] for j in range(num_seeds*2)]
        [rearranged_data.append(this_agent_data[j]) for j in selection_train]
        # [rearranged_data.append(this_agent_data[j]) for j in selection_val]
    assert len(rearranged_data) == len(data) / 2
    return rearranged_data


original_data = [[None, 46.0, 29.0, 13.0, 25.0, 51.0, None, None, 24.0, 40.0], [None, 45.0, 29.0, 13.0, 28.0, 57.0, None, None, 25.0, 39.0], [None, 29.0, 18.0, 31.0, 28.0, 79.0, None, None, 45.0, 56.0], [None, 34.0, 17.0, 34.0, 28.0, 57.0, None, None, 48.0, 55.0], [None, 34.0, 29.0, 33.0, 47.0, 91.0, None, None, 25.0, 53.0], [None, 40.0, 29.0, 23.0, 35.0, 91.0, None, None, 14.0, 49.0], [None, 49.0, 22.0, 10.0, 2.0, 64.0, None, None, 25.0, 29.0], [None, 38.0, 17.0, 9.0, 3.0, 60.0, None, None, 18.0, 27.0], [None, 11.0, 7.0, 36.0, 27.0, 18.0, None, None, 54.0, 54.0], [None, 11.0, 8.0, 21.0, 5.0, 22.0, None, None, 25.0, 51.0], [None, 42.0, 29.0, 19.0, 22.0, 48.0, None, None, 26.0, 25.0], [None, 45.0, 32.0, 27.0, 48.0, 36.0, None, None, 31.0, 27.0], [None, 52.0, 16.0, 13.0, 51.0, 80.0, None, None, 5.0, 57.0], [None, 52.0, 17.0, 12.0, 63.0, 88.0, None, None, 5.0, 48.0], [None, 23.0, 13.0, 3.0, 38.0, 72.0, None, None, 6.0, 36.0], [None, 21.0, 10.0, 4.0, 32.0, 72.0, None, None, 10.0, 35.0], [None, 59.0, 21.0, 15.0, 28.0, 76.0, None, None, 19.0, 85.0], [None, 61.0, 21.0, 12.0, 30.0, 66.0, None, None, 1.0, 79.0], [None, 24.0, 9.0, 5.0, 28.0, 64.0, None, None, 13.0, 52.0], [None, 27.0, 9.0, 7.0, 44.0, 74.0, None, None, 16.0, 39.0], [None, 16.0, 3.0, 2.0, 15.0, 76.0, None, None, 10.0, 46.0], [None, 16.0, 5.0, 5.0, 20.0, 72.0, None, None, 12.0, 42.0], [None, 66.0, 23.0, 12.0, 57.0, 47.0, None, None, 7.0, 92.0], [None, 68.0, 25.0, 14.0, 63.0, 47.0, None, None, 5.0, 91.0], [None, 61.0, 37.0, 16.0, 48.0, 57.0, None, None, 6.0, 45.0], [None, 61.0, 40.0, 24.0, 52.0, 56.0, None, None, 10.0, 37.0], [None, 57.0, 20.0, 16.0, 39.0, 63.0, None, None, 9.0, 90.0], [None, 60.0, 20.0, 26.0, 78.0, 68.0, None, None, 13.0, 94.0]]


assert len(original_data) == 3*8+2*2
assert len(original_data[0]) == 10

# Settings:
# num_seeds = <-- below!
agent_base_names = ['cc_1_mantom', 'cc_20_mantoms', 'cc_1_bc', 'cc_20_bcs', 'cc_20_mixed']

data = take_train_only(original_data, agent_base_names)

agent_names = ['cc_1_mantom_train', 'cc_20_mantoms_train', 'cc_1_bc_train', 'cc_20_bcs_train', 'cc_20_mixed_train']

# bests = ['train', 'val']

tests = [1, 3, 4, 5, 8, 9]
weighting = [2, 2, 1, 1, 1, 1]

# Average over seeds and remove test 2:
# avg_over_seeds_dict = {agent_names[i]: {} for i in range(len(agent_names))}
avg_over_seeds_list = [[] for _ in range(len(agent_names))]
sd_over_seeds_list = [[] for _ in range(len(agent_names))]

for i, agent_name in enumerate(agent_names):
    for k, test in enumerate(tests):

        num_seeds = 3 if agent_name not in ['cc_20_mixed_train', 'cc_20_mixed_val'] else 2
        assert len(data) == (3*4 + 2*1)

        this_avg = np.mean([data[num_seeds*i + j][test] for j in range(num_seeds)])
        this_sd = np.std([data[num_seeds*i + j][test] for j in range(num_seeds)])
        # avg_over_seeds_dict[agent_name]['test{}'.format(test)] = this_avg
        avg_over_seeds_list[i].append(this_avg)
        sd_over_seeds_list[i].append(this_sd)

num_seeds = None  # Reset just in case

# Plot 1:
#       Plot each agent on a separate subplot (4 subplot)
#       Plot the 6 tests in a bar chart
#       Highlight test 5, which the TOM also does badly on

colours = ['b', 'r', 'y', 'c', 'm', 'g']

f, ((ax0, ax1, ax4), (ax2, ax3, empty)) = plt.subplots(2, 3, sharex='col', sharey='row')

x_axis = ['test{}'.format(test) for test in tests]

axs = [ax0, ax1, ax2, ax3, ax4]
assert len(axs) == len(agent_names)

for i, ax in enumerate(axs):
    ax.bar(x_axis, avg_over_seeds_list[i], 0.4, alpha=0.4, color=colours, yerr = sd_over_seeds_list[i])  # thickness
    ax.title.set_text(agent_names[i])
    ax.set_ylabel('% success')
    ax.set_ylim(0, 100)
    ax.grid()

# plt.tight_layout()
plt.show()


# Plot 2:
#         ◦ Avg over seeds
#         ◦ Plot each test on a separate subplot (6 subplots – but highlight test 5)
#         ◦ Plot the 4 agents in a bar chart

colours = ['b', 'r', 'y', 'c']
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
x_axis = agent_names

axs = [ax1, ax2, ax3, ax4, ax5, ax6]

for i, ax in enumerate(axs):
    avg_to_plot = [avg_over_seeds_list[k][i] for k in range(len(agent_names))]
    assert len(avg_over_seeds_list) == len(agent_names)
    sd_to_plot = [sd_over_seeds_list[k][i] for k in range(len(agent_names))]
    ax.bar(x_axis, avg_to_plot, 0.4, alpha=0.4, color=colours, yerr=sd_to_plot)  # thickness
    ax.title.set_text('test{}'.format(tests[i]))
    ax.set_ylabel('% success')
    ax.set_ylim(0, 100)
    ax.grid()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

# plt.tight_layout()
plt.show()


# Plot 3:
#       Avg over seeds and over tests
#       Plot both weighted and non-weighted avg

# Take mean over tests:
mean_over_tests = []
weighted_mean_over_tests = []

for i in range(len(data)):
    mean_over_tests.append(np.mean([data[i][test] for test in tests if data[i][test] is not None]))
    weighted_mean = sum([data[i][test]*weighting[j] for j, test in enumerate(tests) if data[i][test] is not None])/sum(weighting)
    weighted_mean_over_tests.append(weighted_mean)

mean_over_seeds = []
sd_over_seeds = []
weighted_mean_over_seeds = []
sd_of_weighted = []

# Find mean and sd over seeds:
for i in range(len(agent_names)):
    num_seeds = 3 if agent_names[i] not in ['cc_20_mixed_train', 'cc_20_mixed_val'] else 2
    mean_over_seeds.append(np.mean([mean_over_tests[i*num_seeds + j] for j in range(num_seeds)]))
    sd_over_seeds.append(np.std([mean_over_tests[i*num_seeds + j] for j in range(num_seeds)]))
    weighted_mean_over_seeds.append(np.mean([weighted_mean_over_tests[i*num_seeds + j] for j in range(num_seeds)]))
    sd_of_weighted.append(np.std([weighted_mean_over_tests[i*num_seeds + j] for j in range(num_seeds)]))

colours = ['b', 'r', 'y', 'c', 'm']
f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row')
x_axis = agent_names

ax1.bar(x_axis, mean_over_seeds, 0.4, alpha=0.4, color=colours, yerr=sd_over_seeds)  # thickness
ax1.title.set_text('Mean over seeds')
ax1.set_ylabel('% success')
ax1.set_ylim(0, 100)
ax1.grid()
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

ax2.bar(x_axis, weighted_mean_over_seeds, 0.4, alpha=0.4, color=colours, yerr=sd_of_weighted)  # thickness
ax2.title.set_text('Weighted mean over seeds')
ax2.set_ylabel('% success')
ax2.set_ylim(0, 100)
ax2.grid()
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

plt.tight_layout()
plt.show()




# # TEST:
# n= 6
#
# m1 = (0.10,0.12,0.10,0.11,0.14,0.10)
# m2=(0.21,0.21,0.20,0.22,0.20,0.21)
# m3=(0.29,0.27,0.28,0.24,0.23,0.23)
# m4=(0.41,0.39,0.35,0.37,0.41,0.40)
# x=[1,2,3,4,5,6]
#
# fig, ax = plt.subplots()
#
# index = np.arange(n)
# bar_width = 0.2
#
# opacity = 0.4
# error_config = {'ecolor': '0.3'}
# r1 = ax.bar(index, m1, bar_width,
#                  alpha=opacity,
#                  color='b',
#
#                  error_kw=error_config)
#
# r2 = ax.bar(index + bar_width, m2, bar_width,
#                  alpha=opacity,
#                  color='r',
#
#                  error_kw=error_config)
#
# r3 = ax.bar(index + bar_width+ bar_width, m3, bar_width,
#                  alpha=opacity,
#                  color='y',
#                  error_kw=error_config)
# r4 = ax.bar(index + bar_width+ bar_width+ bar_width, m4, bar_width,
#                  alpha=opacity,
#                  color='c',
#                  error_kw=error_config)
# plt.xlabel('D')
# plt.ylabel('Anz')
# plt.title('Th')
#
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
#
# ax1.bar(x,m1, 0.2) # thickness=0.2
# ax2.bar(x,m2, 0.2)
# ax3.plot(x,m3)
# ax4.plot(x,m4)
#
# plt.tight_layout()
# plt.show()