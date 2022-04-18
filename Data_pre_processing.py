'''
made by XHU-WNCG
2022.4
'''
from dataset import WNCG_Dataset
from hparams import hparams
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Aspect_ratio = []
Aspect_ratio_type = []
para = hparams()
m_Dataset = WNCG_Dataset(para)
for item in range(len(m_Dataset.number)):
    if(m_Dataset.number[item]):
        #print(m_Dataset.center_x_position[item])
        #print(m_Dataset.center_y_position[item])
        delta_x = ((m_Dataset.right_bottom_x_position[item]) - m_Dataset.left_top_x_position[item])
        delta_y = ((m_Dataset.right_bottom_y_position[item]) - m_Dataset.left_top_y_position[item])
        Aspect_ratio.append(delta_x/delta_y)
        dict = {
            0: 0,
            6: 1,
            7: 2,
            8: 3,
            9: 4,
            10: 5,
            25: 6,
            41: 7,
            105: 8,
            110: 9,
            115: 10,
            148: 11,
            156: 12,
            222: 13,
            228: 14,
            235: 15,
            256: 16,
            280: 17,
            310: 18,
            387: 19,
            392: 20,
            394: 21,
            398: 22,
            401: 23,
            402: 24,
            430: 25,
            480: 26,
            485: 27,
            673: 28,
        }
        Aspect_ratio_type.append(dict.get(m_Dataset.number[item]))
for_plot = []
for_plot_indx = []
count = []
for _ in range(0, 30):
    i, j = 0, 0
    a = []
    for_plot_indx.append(_)
    for i in range(len(Aspect_ratio_type)):
        if(Aspect_ratio_type[i] == _):
            a.append((Aspect_ratio[i]))
            j = j + 1
    for_plot.append(np.mean(a))
    count.append(len(a))
print(for_plot)
# # plot
# fig, ax = plt.subplots(figsize=(15, 8))
# VP = ax.boxplot(for_plot, positions=for_plot_indx, widths=1.5, patch_artist=True,
#                 showmeans=False, showfliers=False,
#                 medianprops={"color": "#8ECFC9", "linewidth": 0.5},
#                 boxprops={"facecolor": "#82b0d2", "edgecolor": "white",
#                           "linewidth": 0.5},
#                 whiskerprops={"color": "#8ECFC9", "linewidth": 1.5},
#                 capprops={"color": "C0", "linewidth": 1.5})
# plt.xlabel('species')
# plt.ylabel('Aera')
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

#plt.show()
# sns
# plot



# fig, ax = plt.subplots(figsize=(15, 8))
# #plt.figure(figsize=(19, 6.5))
# ax.bar(range(30), count, width=1, edgecolor="white", linewidth=0.7)
# plt.xlabel('species')
# plt.ylabel('Number')
# # ax.set(xlim=(0, 30), xticks=np.arange(1, 8),
# #        ylim=(0, 25), yticks=np.arange(1, 250))
#
# plt.show()

