import numpy as np
import matplotlib.pyplot as plt

import model
import data

file_name = "C:/Users/Antoine/Desktop/Scolaire/PTS7/FACES/FERET/colorferet/images"
number_images = 5000
stdd = 0.01

###CHARGEMENT BASE DE DONNEES ET BRUITAGE
db = data.open_images(file_name,number_images)
'''
print(db.shape)
test = db[2635]
test = np.reshape(test,(64,64))
plt.figure(1)
plt.imshow(test,cmap='gray',interpolation='nearest')
'''
db_noisy = data.add_noise(db,'gaussian',stdd,number_images)
'''
print(db_noisy.shape)
test_noisy = db_noisy[2635]
test_noisy = np.reshape(test_noisy,(64,64))
plt.figure(2)
plt.imshow(test_noisy,cmap='gray',interpolation='nearest')

plt.show()
'''
print(np.max(db))
###ENTRAINEMENT
model.train_clean_to_clean(db)
