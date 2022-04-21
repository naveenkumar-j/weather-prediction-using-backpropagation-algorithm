from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

# Loading Prediciton data
pre_data = np.loadtxt('predict.txt')
model = model_from_json(open('model_architecture.json').read())
model.load_weights('weights.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
pre_data = pre_data.astype('int32')
pre_data = pre_data.reshape(1,12)


print('\n \t \t \tInputs')
print('\n',pre_data)

ans = model.predict(pre_data)
np.set_printoptions(suppress=True)

# Data Representation
print('\n \tSUNNY \t RAINY \t CLOUDY \tCOLD')
print('\n',ans)
yy=["ThunderStorm","Rainy","Foggy","Sunny"]
xx=[i*100 for i in ans[0]] 

# Visualization
fig, ax = plt.subplots()
ax.barh(yy,xx)
ax.set(xlim=[0, 100], xlabel='Probability', ylabel='',title='Weather Prediction')
plt.show()