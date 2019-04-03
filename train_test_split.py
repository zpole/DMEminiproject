import os
import random
import shutil
from math import floor

if __name__ == '__main__':
    root_dir = 'webkb'
    classes = ["course", "department", "faculty", "other", "project", "staff", "student"]
    unis = ['cornell', 'misc', 'texas', 'washington', 'wisconsin']
    for cls in classes:
        for uni in unis:
            dir = root_dir + '/' + cls + '/' + uni + '/'
            traindir = 'train/' + cls + '/' + uni + '/'
            testdir = 'test/' + cls + '/' + uni + '/'
            os.makedirs(traindir)
            os.makedirs(testdir)
            filelist = os.listdir(dir)
            testlist = random.choices(filelist, k=floor(len(filelist)/10))
            for f in filelist:
                if f in testlist:
                    shutil.copy(dir + f, testdir + f)
                else:
                    shutil.copy(dir + f, traindir + f)

