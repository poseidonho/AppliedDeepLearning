import os
import shutil
import random
import glob
from time import sleep
import subprocess
from pathlib import Path

# Set Classes
classes = 'Machine'

# Set number of file to be move
target = 91
# Getting path for source and destination of images
root_img_dir  = 'D:/My Pciture 2020/ConputerVision/Applied Deep Learning/dl4j/' + classes
val_img_dir   = 'C:/Users/howen/Downloads/yolov5-master/datasets/images/mydata/val'
test_img_dir  = 'D:/My Pciture 2020/ConputerVision/Applied Deep Learning/datasets/mydata/images/test'
train_img_dir = 'D:/My Pciture 2020/ConputerVision/Applied Deep Learning/datasets/mydata/images/train'


# Getting path for source and destination of labels
root_lbl_dir  = 'D:/My Pciture 2020/ConputerVision/Applied Deep Learning/dl4j/'+  classes +'/annotations'
val_lbl_dir   = 'C:/Users/howen/Downloads/yolov5-master/datasets/labels/mydata/val'
test_lbl_dir  = 'D:/My Pciture 2020/ConputerVision/Applied Deep Learning/datasets/mydata/images/test/annotations'
train_lbl_dir = 'D:/My Pciture 2020/ConputerVision/Applied Deep Learning/datasets/mydata/images/train/annotations'

# Convert to the directory path
img_root_dir = os.listdir(root_img_dir)
val_img = os.path.normpath(val_img_dir)
test_img = os.path.normpath(test_img_dir)
train_img = os.path.normpath(train_img_dir)

lbl_root_dir = os.listdir(root_lbl_dir)
val_lbl = os.path.normpath(val_lbl_dir)
test_lbl = os.path.normpath(test_lbl_dir)
train_lbl = os.path.normpath(train_lbl_dir)


# Count the total files
count_img = len(img_root_dir)
count_lbl = len(lbl_root_dir)

'''
# Random split for val 
for i in range(target):
    
    #Random Choose file
    filename = random.choice(img_root_dir)
    if not (os.path.isdir(os.path.join(root_img_dir,filename))):
        while (os.path.isdir(os.path.join(root_img_dir,filename))):
            filename = random.choice(img_root_dir)
        fname = os.path.join(val_img, filename)    
        while os.path.isfile(fname):
            filename = random.choice(img_root_dir)
            fname = os.path.join(val_img, filename)
        basename_no_ext =  os.path.splitext(filename)[0]

        source_img = os.path.join(root_img_dir, filename)
        source_lbl = os.path.join(root_lbl_dir , basename_no_ext + '.txt')

        shutil.move(source_img, val_img) 
        shutil.move(source_lbl, val_lbl)
        print("{} Complete to val".format(basename_no_ext))
        sleep(0.1)
'''

# Update Path
img_root_dir = os.listdir(root_img_dir)
lbl_root_dir = os.listdir(root_lbl_dir)

# Random split for test 
for i in range(target):
    
  #Random Choose file
    filename = random.choice(img_root_dir)
    if not (os.path.isdir(os.path.join(root_img_dir,filename))):
        while (os.path.isdir(os.path.join(root_img_dir,filename))):
            filename = random.choice(img_root_dir)
        fname = os.path.join(test_img, filename)
        while os.path.isfile(fname):
            filename = random.choice(img_root_dir)
            fname = os.path.join(test_img, filename)
        basename_no_ext =  os.path.splitext(filename)[0]    
        source_img = os.path.join(root_img_dir, filename)
    #    source_lbl = os.path.join(root_lbl_dir , basename_no_ext + '.txt')
        source_lbl = os.path.join(root_lbl_dir , basename_no_ext + '.xml')
    
        shutil.move(source_img, test_img) 
        shutil.move(source_lbl, test_lbl)
        print("{} Complete to test".format(basename_no_ext))
        sleep(0.1)

# Update Path
img_root_dir = os.listdir(root_img_dir)
lbl_root_dir = os.listdir(root_lbl_dir)
# Move remaining files
for fileremain in img_root_dir:
    
    if not (os.path.isdir(os.path.join(root_img_dir,fileremain))):
        
        fname = os.path.join(train_img, fileremain)
        while os.path.isfile(fname):
            filename = random.choice(img_root_dir)
            fname = os.path.join(train_img, filename)        
        basename_no_ext =  os.path.splitext(fileremain)[0]
        source_img = os.path.join(root_img_dir, fileremain)
        #source_lbl = os.path.join(root_lbl_dir , basename_no_ext + '.txt')
        source_lbl = os.path.join(root_lbl_dir , basename_no_ext + '.xml')
        shutil.move(source_img, train_img) 
        shutil.move(source_lbl, train_lbl)
        print("{} Complete to train".format(basename_no_ext))
        #sleep(1)
