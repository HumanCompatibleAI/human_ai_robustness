
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


data = [[None, 84.0, 36.0, 4.0, 2.0, 85.0, None, None, 0.0, 85.0], [None, 51.0, 16.0, 7.0, 0.0, 61.0, None, None, 17.0, 57.0], [None, 71.0, 17.0, 13.0, 19.0, 68.0, None, None, 6.0, 62.0], [None, 69.0, 25.0, 9.0, 11.0, 39.0, None, None, 1.0, 85.0], [None, 58.0, 21.0, 34.0, 22.0, 48.0, None, None, 14.0, 74.0], [None, 79.0, 52.0, 9.0, 8.0, 71.0, None, None, 14.0, 86.0], [None, 41.0, 21.0, 3.0, 24.0, 68.0, None, None, 10.0, 82.0], [None, 61.0, 35.0, 12.0, 40.0, 68.0, None, None, 4.0, 78.0], [None, 33.0, 25.0, 12.0, 13.0, 53.0, None, None, 38.0, 81.0], [None, 53.0, 22.0, 36.0, 29.0, 67.0, None, None, 59.0, 75.0], [None, 71.0, 21.0, 23.0, 8.0, 83.0, None, None, 21.0, 88.0], [None, 71.0, 28.0, 25.0, 44.0, 77.0, None, None, 23.0, 67.0], [None, 56.0, 29.0, 15.0, 31.0, 40.0, None, None, 16.0, 74.0], [None, 75.0, 32.0, 19.0, 22.0, 68.0, None, None, 14.0, 79.0], [None, 59.0, 24.0, 10.0, 0.0, 62.0, None, None, 27.0, 71.0], [None, 57.0, 39.0, 15.0, 45.0, 79.0, None, None, 14.0, 90.0], [None, 38.0, 37.0, 17.0, 16.0, 88.0, None, None, 1.0, 88.0], [None, 57.0, 18.0, 17.0, 44.0, 60.0, None, None, 23.0, 73.0], [None, 44.0, 23.0, 15.0, 8.0, 76.0, None, None, 19.0, 80.0], [None, 55.0, 24.0, 34.0, 23.0, 87.0, None, None, 33.0, 87.0]]

assert len(data) == 4*5
assert len(data[0]) == 10

# Settings:
num_seeds = 5
agent_names = ['ppo_sp', 'ppo_bc_tr', 'ppo_bc_te', 'ppo_hm']
tests = [1, 3, 4, 5, 8, 9]
assert num_seeds == int(len(data)/len(agent_names))

# Average over seeds and remove test 2:
# avg_over_seeds_dict = {agent_names[i]: {} for i in range(len(agent_names))}
avg_over_seeds_list = [[] for _ in range(len(agent_names))]
sd_over_seeds_list = [[] for _ in range(len(agent_names))]

for i, agent in enumerate(agent_names):
    for test in tests:

        this_avg = np.mean([data[num_seeds*i + j][test] for j in range(num_seeds)])
        this_sd = np.std([data[num_seeds*i + j][test] for j in range(num_seeds)])
        # avg_over_seeds_dict[agent]['test{}'.format(test)] = this_avg
        avg_over_seeds_list[i].append(this_avg)
        sd_over_seeds_list[i].append(this_sd)

# Plot 1:
#       Plot each agent on a separate subplot (4 subplot)
#       Plot the 6 tests in a bar chart
#       Highlight test 5, which the TOM also does badly on

colours = ['b', 'r', 'y', 'c', 'm', 'g']

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

x_axis = ['test{}'.format(test) for test in tests]

axs = [ax1, ax2, ax3, ax4]

for i, ax in enumerate(axs):
    ax.bar(x_axis, avg_over_seeds_list[i], 0.4, alpha=0.4, color=colours, yerr = sd_over_seeds_list[i])  # thickness
    ax.title.set_text(agent_names[i])
    ax.set_ylabel('% success')
    ax.set_ylim(0, 100)
    ax.grid()

plt.tight_layout()
# plt.show()


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
    sd_to_plot = [sd_over_seeds_list[k][i] for k in range(len(agent_names))]
    ax.bar(x_axis, avg_to_plot, 0.4, alpha=0.4, color=colours, yerr=sd_to_plot)  # thickness
    ax.title.set_text('test{}'.format(tests[i]))
    ax.set_ylabel('% success')
    ax.set_ylim(0, 100)
    ax.grid()

plt.tight_layout()
# plt.show()



# Plot 3:
#       Avg over seeds and over tests
#       Plot both weighted and non-weighted avg

# Take mean over tests:
mean_over_tests = []
weighted_mean_over_tests = []
weighting = [2, 2, 1, 1, 1, 1]

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
    mean_over_seeds.append(np.mean([mean_over_tests[i*num_seeds + j] for j in range(num_seeds)]))
    sd_over_seeds.append(np.std([mean_over_tests[i*num_seeds + j] for j in range(num_seeds)]))
    weighted_mean_over_seeds.append(np.mean([weighted_mean_over_tests[i*num_seeds + j] for j in range(num_seeds)]))
    sd_of_weighted.append(np.std([weighted_mean_over_tests[i*num_seeds + j] for j in range(num_seeds)]))

colours = ['b', 'r', 'y', 'c']
f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row')
x_axis = agent_names

ax1.bar(x_axis, mean_over_seeds, 0.4, alpha=0.4, color=colours, yerr=sd_over_seeds)  # thickness
ax1.title.set_text('Mean over seeds')
ax1.set_ylabel('% success')
ax1.set_ylim(0, 100)
ax1.grid()

ax2.bar(x_axis, weighted_mean_over_seeds, 0.4, alpha=0.4, color=colours, yerr=sd_of_weighted)  # thickness
ax2.title.set_text('Weighted mean over seeds')
ax2.set_ylabel('% success')
ax2.set_ylim(0, 100)
ax2.grid()

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