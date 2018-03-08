# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:17:33 2018

@author: 俊松
"""
import os
import pandas as pd
from PIL import Image, ImageDraw

def draw_boxes(path, name, rectangle = False):    
    file_pic = '\\image_1000\\%s.jpg'%name  # 图片的地址
    file_text = '\\txt_1000\\%s.txt'%name   # 描述文件的地址
    if not os.path.exists(path + file_pic):
        print('无法找到图片')
        return
    if not os.path.exists(path + file_text):
        print('无法找到描述文件')
        return
    
    img = Image.open(path + file_pic) #读图片
    draw = ImageDraw.Draw(img)
    pwd = os.getcwd()                 #读.txt文件
    os.chdir(os.path.dirname(path + file_text))
    text_point = pd.read_csv(os.path.basename(path + file_text))
    os.chdir(pwd)
    
    #print(text_point) #能获得数据
    for idx, row in text_point.iterrows():
        point = row.tolist()    # 依次读取八个点的数据
        x = [point[i] for i in [0, 2, 4, 6]]
        y = [point[i] for i in [1, 3, 5, 7]]
        point = [(a,b) for a,b in zip(x,y)] 
        draw.polygon(point, outline=(0,128,255)) # 画多边形
 
        if rectangle: # 如果要画长方形
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            draw.rectangle((x_min, y_min, x_max, y_max), outline = (0, 0, 255))
            
    return img   

if __name__ == '__main__':    
    path = 'C:\\Users\\俊松\\Documents\\我的工作区\\我的学习\\台湾大学李课程pdf\\ICPR\\ICPR_text_train_part1_20180211' # 文件夹地址
    name = 'TB1..FLLXXXXXbCXpXXunYpLFXX'
    img = draw_boxes(path, name, True)
    #img.resize((400,400)).save(path + '\\demo.jpg') # 保存图片 会出现画图不完整
    img.show()
    #测试代码

#     测试代码
#    img = Image.open(path + '\\demo.jpg')
#    draw = ImageDraw.Draw(img)
#    rect = (95,60,167,118)
#    draw.rectangle(rect)
#    img.show()
