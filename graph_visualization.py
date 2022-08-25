import matplotlib.pyplot as plt
import numpy as np

x_ticks = np.array([1000, 800, 600, 400, 200])
y_ticks= np.array([75, 80, 85, 90])
y_lim = np.array([75, 92])
linewidth='2.3'

points = np.array([1024, 768, 512, 256, 128]) # number of points
ssg = np.array([64.5, 61.0, 56.2,54.3, 56.3])
ssg_dp = np.array([88.9 , 88.8, 87.3, 85.8, 83.2])
msg_dp = np.array([89.5, 89.4, 89.5, 86, 82.3])
mrg_dp = np.array([89.7,90.0,88.8,87.2,84.3])

def draw_graph():

    plt.figure(figsize=(6,3))
    # add plot
    plt.plot(points,ssg_dp,label='SSG+DP', marker='s', color='#F1C232', linewidth=linewidth)
    plt.plot(points,mrg_dp,label='MRG+DP', marker='s', color='#0099C6', linewidth=linewidth)
    plt.plot(points,msg_dp,label='MSG+DP', marker='s', color='#CC0000', linewidth=linewidth)
    plt.plot(points,ssg,label='SSG', marker='s', color='#FF9900',linewidth=linewidth)

    # config graph
    plt.xlabel('Number of Points', fontsize=13)
    plt.ylabel('Accuracy (%)',fontsize=13)
    plt.xticks(x_ticks,fontsize=13)
    plt.yticks(y_ticks,fontsize=13)
    plt.ylim(y_lim)
    plt.grid(axis='x')
    plt.gca().invert_xaxis()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().tick_params(length=0)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False, fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.savefig('my_robustness_test.png')

if __name__ == '__main__' :
    draw_graph()