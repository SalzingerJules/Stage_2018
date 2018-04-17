from PIL import Image
import numpy as np
import os
import random as rd

def open_images(file_name,image_number=5000):
    os.chdir(file_name)
    folder_length = []
    folders = os.listdir()
    for i in folders:
        folder_length.append(len([k for k in os.listdir(i) if k.endswith('.jpg')]))
    samples = rd.sample([i for i in range(sum(folder_length))],image_number)
    array = np.zeros((image_number,64,64,1))
    for nb,i in enumerate(samples):
        path = ""
        for j in range(1,len(folders)+1):
            if i<sum(folder_length[:j]):
                path+=folders[j-1]+"/"+[k for k in os.listdir("./"+folders[j-1]) if k.endswith('.jpg')][i-sum(folder_length[:j-1])]
                break
        img = Image.open(path).convert('L')
        x1 = rd.randint(0,img.size[0]-65)
        y1 = rd.randint(0,img.size[1]-65)
        img = img.crop((x1,y1,x1+64,y1+64))
        array[nb] = np.reshape(np.asarray(img).astype(np.float32)/255,(64,64,1))
        if nb/(image_number//100) == nb//(image_number//100):
            print("Loading database... "+str(nb//(image_number//100))+"%.")
    return array

def add_noise(array,noise_type='gaussian',stdd=0.01,image_number=5000):
    if noise_type=='gaussian':
        return np.add(array,np.random.normal(0,stdd,(image_number,64,64,1)))


