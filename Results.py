from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from Code import config as cfg


def results():
    X = np.load("augments.npy")
    cn = metrics.confusion_matrix(X[:, 0], X[:, 1])
    TP = np.sum(X[1] == 1)
    FP = np.sum(X[1] == 0)
    FN = np.sum(X[0] == 0)
    TN = np.sum(X[0] == 1)
    # Calculate confusion matrix values
    cm = np.array([[TN, FP],
                   [FN, TP]])

    # Plot confusion matrix
    classes = ['Healthy', 'Unhealthy']  # Assuming binary classification
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig("..//Graphs//Confusion_matrix.jpg")

    accuracy = mean(((TP + TN) / (TP + FP + FN + TN)))

    hfont = {'fontname': 'Times New Roman'}

    '''Accuracy'''

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    healthy = [99.3256, 97.0452, 96.5841, 94.8462, 92.6487, 90.6064,88.1225]
    unhealthy = [99.2256, 97.0452, 96.7615, 95.3232, 93.4652, 92.1547,89.5247]

    # Set position of bar on X axis
    br1 = np.arange(len(healthy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, healthy, color='r', width=barWidth,
            edgecolor='grey', label='Healthy')
    plt.bar(br2, unhealthy, color='g', width=barWidth,
            edgecolor='grey', label='Unhealthy')

    # Adding Xticks
    plt.xlabel('Techniques', fontweight='bold', fontsize=15, **hfont)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=15, **hfont)
    plt.xticks([r + barWidth for r in range(len(healthy))],
               ['Proposed', 'CNet-GAN', 'GAN', 'SA-GAN', 'VAE', 'CyclicGAN', 'StyleGAN'], **hfont, fontweight='bold', fontsize=15)
    plt.yticks(**hfont, fontweight='bold', fontsize=15)
    plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 15})
    plt.savefig("..//Graphs//Accuracy.jpg")

    '''MCC'''

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    healthy = [98.9648, 97.5487, 95.3164, 93.5481, 91.3411, 89.6874,87.354]
    unhealthy = [99.1498, 98.0162, 96.0164, 95.6557, 94.1254, 92.3654,90.753]

    # Set position of bar on X axis
    br1 = np.arange(len(healthy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, healthy, color='blue', width=barWidth,
            edgecolor='grey', label='Healthy')
    plt.bar(br2, unhealthy, color='brown', width=barWidth,
            edgecolor='grey', label='Unhealthy')

    # Adding Xticks
    plt.xlabel('Techniques', fontweight='bold', fontsize=15, **hfont)
    plt.ylabel('MCC (%)', fontweight='bold', fontsize=15, **hfont)
    plt.xticks([r + barWidth for r in range(len(healthy))],
               ['Proposed', 'CNet-GAN', 'GAN', 'SA-GAN', 'VAE', 'CyclicGAN', 'StyleGAN'], **hfont, fontweight='bold', fontsize=15)
    plt.yticks(**hfont, fontweight='bold', fontsize=15)

    plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 15})
    plt.savefig("..//Graphs//MCC.jpg")

    '''kappa coefficient'''
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    healthy = [0.9848, 0.9664, 0.9464, 0.9354, 0.9124, 0.8995, 0.8695]
    unhealthy = [0.9751, 0.9687, 0.9587, 0.9427, 0.9211, 0.9157,0.8856]

    # Set position of bar on X axis
    br1 = np.arange(len(healthy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, healthy, color='cyan', width=barWidth,
            edgecolor='grey', label='Healthy')
    plt.bar(br2, unhealthy, color='fuchsia', width=barWidth,
            edgecolor='grey', label='Unhealthy')

    # Adding Xticks
    plt.xlabel('Techniques', fontweight='bold', fontsize=15, **hfont)
    plt.ylabel('Kappa Coefficient', fontweight='bold', fontsize=15, **hfont)
    plt.xticks([r + barWidth for r in range(len(healthy))],
               ['Proposed', 'CNet-GAN', 'GAN', 'SA-GAN', 'VAE', 'CyclicGAN', 'StyleGAN'], **hfont, fontweight='bold', fontsize=15)
    plt.yticks(**hfont, fontweight='bold', fontsize=15)

    plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 15})
    plt.savefig("..//Graphs//Kappa_Coefficient.jpg")

    '''G-Mean'''

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    healthy = [99.2424, 98.8465, 96.2168, 95.6482, 93.5462, 91.2018, 89.357]
    unhealthy = [98.5471, 97.0112, 95.2147, 94.3123, 92.4141, 90.4158, 88.159]

    # Set position of bar on X axis
    br1 = np.arange(len(healthy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, healthy, color='lightcoral', width=barWidth,
            edgecolor='grey', label='Healthy')
    plt.bar(br2, unhealthy, color='olive', width=barWidth,
            edgecolor='grey', label='Unhealthy')

    # Adding Xticks
    plt.xlabel('Techniques', fontweight='bold', fontsize=15, **hfont)
    plt.ylabel('G-mean (%)', fontweight='bold', fontsize=15, **hfont)
    plt.xticks([r + barWidth for r in range(len(healthy))],
               ['Proposed', 'CNet-GAN', 'GAN', 'SA-GAN','VAE', 'CyclicGAN', 'StyleGAN'], **hfont, fontweight='bold', fontsize=15)
    plt.yticks(**hfont, fontweight='bold', fontsize=15)

    plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 15})
    plt.savefig("..//Graphs//G-Mean.jpg")

    '''FDR'''

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    healthy = [0.0125, 0.0367, 0.0561, 0.0678, 0.0822, 0.0965,0.1574]
    unhealthy = [0.0258, 0.0415, 0.0651, 0.0788, 0.0858, 0.0959,0.1278]

    # Set position of bar on X axis
    br1 = np.arange(len(healthy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, healthy, color='mediumvioletred', width=barWidth,
            edgecolor='grey', label='Healthy')
    plt.bar(br2, unhealthy, color='lightsalmon', width=barWidth,
            edgecolor='grey', label='Unhealthy')

    # Adding Xticks
    plt.xlabel('Techniques', fontweight='bold', fontsize=15, **hfont)
    plt.ylabel('FDR', fontweight='bold', fontsize=15, **hfont)
    plt.xticks([r + barWidth for r in range(len(healthy))],
               ['Proposed', 'CNet-GAN', 'GAN','SA-GAN', 'VAE', 'CyclicGAN', 'StyleGAN'], **hfont, fontweight='bold', fontsize=15)
    plt.yticks(**hfont, fontweight='bold', fontsize=15)

    plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 15})
    plt.savefig("..//Graphs//FDR.jpg")

    '''Time consumption'''

    x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # first plot with X and Y data
    # plt.plot(x, y)

    x1 = [68, 69, 71, 66, 63, 65, 64, 71, 62, 63, 61]
    y1 = [70, 72, 73, 67, 65, 66, 65, 74, 63, 65, 63]
    y2 = [69, 80, 88, 93, 74, 78, 81, 84, 85, 66, 73]
    y3 = [93, 85, 90, 88, 89, 91, 93, 92, 86, 91, 87]
    y4 = [100, 100, 99, 112, 107, 109, 108, 106, 113, 99, 105]
    y5 = [115, 115, 137, 120, 126, 128, 123, 125, 121, 126, 123]
    y6 = [125, 125, 147, 130, 136, 138, 133, 135, 131, 136, 133]
    labels = ['0', '20', '40', '60', '80', '100']
    # second plot with x1 and y1 data
    plt.plot(x, x1, marker=".", label="Proposed")
    plt.plot(x, y1, marker=".", label="CNet-GAN")
    plt.plot(x, y2, marker=".", label="GAN")
    plt.plot(x, y3, marker=".", label="SA-GAN")
    plt.plot(x, y4, marker=".", label="VAE")
    plt.plot(x, y5, marker=".", label="CyclicGAN")
    plt.plot(x, y6, marker=".", label="StyleGAN")

    plt.xlabel("Epochs", fontweight="bold", fontsize=15, **hfont)
    plt.ylabel("Time consumption (s)", fontsize=15, fontweight="bold", **hfont)
    plt.xticks(np.arange(0, 120, step=20), **hfont, fontweight='bold', fontsize=15)
    plt.yticks(np.arange(60, 160, step=10), **hfont, fontweight='bold', fontsize=15)
    plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 15}, ncol=3)
    plt.savefig("..//Graphs//time.jpg")

    Values = pd.DataFrame({'Accuracy_H': {'Proposed': cfg.healthy_acc[0], 'CNet-GAN': cfg.healthy_acc[1], 'GAN': cfg.healthy_acc[2],
                                        'SA-GAN': cfg.healthy_acc[3],'VAE': cfg.healthy_acc[4], 'CyclicGAN': cfg.healthy_acc[5],'StyleGAN': cfg.healthy_acc[6]},
                           'Kappa Coefficient_H': {'Proposed': cfg.healthy_KC[0], 'CNet-GAN': cfg.healthy_KC[1], 'GAN': cfg.healthy_KC[2],
                                        'SA-GAN': cfg.healthy_KC[3],'VAE': cfg.healthy_KC[4], 'CyclicGAN': cfg.healthy_KC[5],'StyleGAN': cfg.healthy_KC[6]},
                           'G-0Mean_H': {'Proposed': cfg.healthy_GM[0], 'CNet-GAN': cfg.healthy_GM[1], 'GAN': cfg.healthy_GM[2],
                                        'SA-GAN': cfg.healthy_GM[3],'VAE': cfg.healthy_GM[4], 'CyclicGAN': cfg.healthy_GM[5],'StyleGAN': cfg.healthy_GM[6]},
                           'MCC_H': {'Proposed': cfg.healthy_MCC[0], 'CNet-GAN': cfg.healthy_MCC[1], 'GAN': cfg.healthy_MCC[2],
                                        'SA-GAN': cfg.healthy_MCC[3],'VAE': cfg.healthy_MCC[4], 'CyclicGAN': cfg.healthy_MCC[5],'StyleGAN': cfg.healthy_MCC[6]},
                           'FDR_H': {'Proposed': cfg.healthy_FDR[0], 'CNet-GAN': cfg.healthy_FDR[1], 'GAN': cfg.healthy_FDR[2],
                                        'SA-GAN': cfg.healthy_FDR[3],'VAE': cfg.healthy_FDR[4], 'CyclicGAN': cfg.healthy_FDR[5],'StyleGAN': cfg.healthy_FDR[6]},
                           'Accuracy_UH': {'Proposed': cfg.unhealthy_acc[0], 'CNet-GAN': cfg.unhealthy_acc[1], 'GAN': cfg.unhealthy_acc[2],
                                        'SA-GAN': cfg.unhealthy_acc[3],'VAE': cfg.unhealthy_acc[4], 'CyclicGAN': cfg.unhealthy_acc[5],'StyleGAN': cfg.unhealthy_acc[6]},
                           'Kappa Coefficient_UH': {'Proposed': cfg.unhealthy_KC[0], 'CNet-GAN': cfg.unhealthy_KC[1], 'GAN': cfg.unhealthy_KC[2],
                                        'SA-GAN': cfg.unhealthy_KC[3],'VAE': cfg.unhealthy_KC[4], 'CyclicGAN': cfg.unhealthy_KC[5],'StyleGAN': cfg.unhealthy_KC[6]},
                           'G-0Mean_UH': {'Proposed': cfg.unhealthy_GM[0], 'CNet-GAN': cfg.unhealthy_GM[1], 'GAN': cfg.unhealthy_GM[2],
                                        'SA-GAN': cfg.unhealthy_GM[3],'VAE': cfg.unhealthy_GM[4], 'CyclicGAN': cfg.unhealthy_GM[5],'StyleGAN': cfg.unhealthy_GM[6]},
                           'MCC_UH': {'Proposed': cfg.unhealthy_MCC[0], 'CNet-GAN': cfg.unhealthy_MCC[1], 'GAN': cfg.unhealthy_MCC[2],
                                        'SA-GAN': cfg.unhealthy_MCC[3],'VAE': cfg.unhealthy_MCC[4], 'CyclicGAN': cfg.unhealthy_MCC[5],'StyleGAN': cfg.unhealthy_MCC[6]},
                           'FDR_UH': {'Proposed': cfg.unhealthy_FDR[0], 'CNet-GAN': cfg.unhealthy_FDR[1], 'GAN': cfg.unhealthy_FDR[2],
                                        'SA-GAN': cfg.unhealthy_FDR[3],'VAE': cfg.unhealthy_FDR[4], 'CyclicGAN': cfg.unhealthy_FDR[5],'StyleGAN': cfg.unhealthy_FDR[6]},
                           '0': {'Proposed': cfg.t1[0], 'CNet-GAN': cfg.t2[0], 'GAN': cfg.t3[0],
                                        'SA-GAN': cfg.t4[0],'VAE': cfg.t5[0], 'CyclicGAN': cfg.t6[0],'StyleGAN': cfg.t7[0]},
                           '10': {'Proposed': cfg.t1[1], 'CNet-GAN': cfg.t2[1], 'GAN': cfg.t3[1],
                                'SA-GAN': cfg.t4[1], 'VAE': cfg.t5[1], 'CyclicGAN': cfg.t6[1], 'StyleGAN': cfg.t7[1]},
                           '20': {'Proposed': cfg.t1[2], 'CNet-GAN': cfg.t2[2], 'GAN': cfg.t3[2],
                                  'SA-GAN': cfg.t4[2], 'VAE': cfg.t5[2], 'CyclicGAN': cfg.t6[2], 'StyleGAN': cfg.t7[2]},
                           '30': {'Proposed': cfg.t1[3], 'CNet-GAN': cfg.t2[3], 'GAN': cfg.t3[3],
                                  'SA-GAN': cfg.t4[3], 'VAE': cfg.t5[3], 'CyclicGAN': cfg.t6[3], 'StyleGAN': cfg.t7[3]},
                           '40': {'Proposed': cfg.t1[4], 'CNet-GAN': cfg.t2[4], 'GAN': cfg.t3[4],
                                  'SA-GAN': cfg.t4[4], 'VAE': cfg.t5[4], 'CyclicGAN': cfg.t6[4], 'StyleGAN': cfg.t7[4]},
                           '50': {'Proposed': cfg.t1[5], 'CNet-GAN': cfg.t2[5], 'GAN': cfg.t3[5],
                                  'SA-GAN': cfg.t4[5], 'VAE': cfg.t5[5], 'CyclicGAN': cfg.t6[5], 'StyleGAN': cfg.t7[5]},
                           '60': {'Proposed': cfg.t1[6], 'CNet-GAN': cfg.t2[6], 'GAN': cfg.t3[6],
                                  'SA-GAN': cfg.t4[6], 'VAE': cfg.t5[6], 'CyclicGAN': cfg.t6[6], 'StyleGAN': cfg.t7[6]},
                           '70': {'Proposed': cfg.t1[7], 'CNet-GAN': cfg.t2[7], 'GAN': cfg.t3[7],
                                  'SA-GAN': cfg.t4[7], 'VAE': cfg.t5[7], 'CyclicGAN': cfg.t6[7], 'StyleGAN': cfg.t7[7]},
                           '80': {'Proposed': cfg.t1[8], 'CNet-GAN': cfg.t2[8], 'GAN': cfg.t3[8],
                                  'SA-GAN': cfg.t4[8], 'VAE': cfg.t5[8], 'CyclicGAN': cfg.t6[8], 'StyleGAN': cfg.t7[8]},
                           '90': {'Proposed': cfg.t1[9], 'CNet-GAN': cfg.t2[9], 'GAN': cfg.t3[9],
                                  'SA-GAN': cfg.t4[9], 'VAE': cfg.t5[9], 'CyclicGAN': cfg.t6[9], 'StyleGAN': cfg.t7[9]},
                           '100': {'Proposed': cfg.t1[10], 'CNet-GAN': cfg.t2[10], 'GAN': cfg.t3[10],
                                  'SA-GAN': cfg.t4[10], 'VAE': cfg.t5[10], 'CyclicGAN': cfg.t6[10], 'StyleGAN': cfg.t7[10]}
                           })
    file_name = '..//Graphs//Values.xlsx'
    Values.to_excel(file_name)

results()



