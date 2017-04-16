import numpy as np
import pickle

import os

path = "/Users/Vajpeyi/Desktop/Music/AI_Duet_Emotions/TrainingData/"
folders = ["PosExamples","NegExamples"]
os.chdir(path)

for folder in folders:
    os.chdir(path+'/'+folder)
    f = open(folder+'.txt', 'w')
    data = []
    for j in range(5):
        x = np.random.randint(0, 127, 20)
        for num in x:
            f.write(str(num) + ' ')
        f.write('\n')
    f.close;