import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

annotations_Dir= 'D:/My Pciture 2020/ConputerVision/Applied Deep Learning/resize/Machine/annotations/'

Save_Dir= 'D:/My Pciture 2020/ConputerVision/Applied Deep Learning/resize/Machine/annotations/yolo/'

classes = ['Teddy', 'Machine']



def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(file,dir_path, output_path): 
    in_file = open(dir_path+file)
    print(in_file)
    basename_no_ext =  os.path.splitext(file)[0]
    #if os.path.splitext(file)[1] is not 'xml':
      #return
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

#cwd = getcwd()


if not os.path.exists(Save_Dir):
   os.makedirs(Save_Dir)

list_file = open(Save_Dir + 'classes.txt', 'w')
for classname in classes:
    list_file.write(classname)
    list_file.write('\n')
list_file.close()

for file in os.listdir(annotations_Dir):
       
   convert_annotation(file,annotations_Dir, Save_Dir)

print("Finished processing: " + Save_Dir)