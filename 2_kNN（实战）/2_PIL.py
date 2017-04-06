# -*- coding: utf-8 -*-
from PIL import Image
from numpy import *
import numpy as np

img = Image.open('2/4.png')
new_img = img.convert('1')

new_img.save('2/4_1.png')


pixels = new_img.load()

arr = zeros([new_img.width, new_img.height],int8)

for x in range(new_img.height):
    for y in range(new_img.width):
        if pixels[x,y]>126: 
            arr[x,y]=0
        else: arr[x,y]=1

xm = np.matrix(arr)

#np.savetxt('2.txt',arr,fmt='%s',newline='\n')
np.savetxt('2/txt/4_1.txt',xm.T,fmt='%s')


'''
l1 = arr.tolist()

file=open('data.txt','w')  
file.write(str(l1));  
file.close()
'''
