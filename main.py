%matplotlib inline
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from google.colab.patches import cv2_imshow
from keras.utils.vis_utils import plot_model
import math
import matplotlib.pyplot as plt
def preprocess(imgGray):
imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)
ret, imgBin = cv2.threshold(imgGray, 42, 255, cv2.THRESH_BINARY_INV)
print('Initial binary representation')
sf = 30
cv2_imshow(cv2.resize(imgBin, (int(imgBin.shape[1]*sf/100),
int(imgBin.shape[0]*sf/100)), interpolation = cv2.INTER_AREA))
imgDst = cv2.medianBlur(imgBin, 5)
kernel = np.ones((4, 4), np.uint8)
imgDst = cv2.morphologyEx(imgBin, cv2.MORPH_ERODE, kernel, iterations=1)
imgDst = cv2.morphologyEx(imgDst, cv2.MORPH_DILATE, kernel, iterations=1)
contours, hierarchy = cv2.findContours(imgDst, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
return contours, hierarchy, imgDst
def removeInvalidContours(imgBin, contours, areaMin, areaMax):
for contour in contours:
area = cv2.contourArea(contour)
if area < areaMin or area > areaMax:
m = cv2.moments(contour)
cx = int(m['m10']/m['m00'])
cy = int(m['m01']/m['m00'])
seedPoint = (cx, cy)
cv2.floodFill(imgBin, None, seedPoint, 0)
print('After preprocessing')
sf = 30
cv2_imshow(cv2.resize(imgBin, (int(imgBin.shape[1]*sf/100),
int(imgBin.shape[0]*sf/100)), interpolation = cv2.INTER_AREA))
def objectContrast(imgBin, imgGray, contour):
kernel = np.ones((5, 5), np.uint8)
imgBrdOut = cv2.morphologyEx(imgBin, cv2.MORPH_DILATE, kernel, iterations=1)
imgBrdOut = cv2.morphologyEx(imgBrdOut, cv2.MORPH_GRADIENT, kernel, iterations=1)
meanObj, stddevObj = cv2.meanStdDev(imgGray, mask=imgBin)
meanBrdOut, stddevBrdOut = cv2.meanStdDev(imgGray, mask=imgBrdOut)
return abs(meanObj[0][0] - meanBrdOut[0][0]) / (meanObj[0][0]+1)
def morphGradient(imgBin, imgGray):
kernel = np.ones((5, 5), np.uint8)
gradientImgBin = cv2.morphologyEx(imgBin, cv2.MORPH_DILATE, kernel, iterations=1)
gradientImgGray = cv2.morphologyEx(imgGray, cv2.MORPH_DILATE, kernel,
iterations=1)
meanImgGray, stdImgGray = cv2.meanStdDev(gradientImgGray, mask=imgBin)
meanImgBin, stdImgBin = cv2.meanStdDev(gradientImgBin, mask=imgBin)
if meanImgGray[0][0] < meanImgBin[0][0]:
img1 = meanImgGray / meanImgBin
else:
img1 = meanImgBin / meanImgGray
gradientC = cv2.bitwise_and(gradientImgGray, gradientImgBin)
meanGradientC, _ = cv2.meanStdDev(gradientC, mask=gradientImgBin)
img2 = 1 - abs(phi1 - meanGradientC[0][0] / 255)
return img1, img2
def borderContrast(imgBin, imgGray, contour):
kernel = np.ones((5, 5), np.uint8)
imgBrdOut = cv2.morphologyEx(imgBin, cv2.MORPH_DILATE, kernel, iterations=1)
imgBrdOut = cv2.morphologyEx(imgBrdOut, cv2.MORPH_GRADIENT, kernel, iterations=1)
cv2.imwrite('sample_data/imgBrdOut.png', imgBrdOut)
imgBrdIn = cv2.morphologyEx(imgBin, cv2.MORPH_ERODE, kernel, iterations=1)
imgBrdIn = cv2.morphologyEx(imgBrdIn, cv2.MORPH_GRADIENT, kernel, iterations=1)
cv2.imwrite('sample_data/imgBrdIn.png', imgBrdIn)
meanBrdOut, stddevBrdOut = cv2.meanStdDev(imgGray, mask=imgBrdOut)
meanBrdIn, stddevBrdIn = cv2.meanStdDev(imgGray, mask=imgBrdIn)
if meanBrdIn[0][0] > meanBrdOut[0][0]:
return abs(meanBrdIn[0][0] - meanBrdOut[0][0]) / (meanBrdIn[0][0]+1)
else:
return abs(meanBrdIn[0][0] - meanBrdOut[0][0]) / (meanBrdOut[0][0]+1)
def density(contour):
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)
return 2*math.sqrt(math.pi*area) / (perimeter+1)
def calculateFeatures(imgName):
imgGray = cv2.imread(imgName)
contours, hierarchy, imgBin = preprocess(imgGray)
removeInvalidContours(imgBin, contours, 50, 1000)
contours, hierarchy = cv2.findContours(imgBin, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
delta = 5
H, W, channels = imgGray.shape
features = []
for contour in contours:
x,y,w,h = cv2.boundingRect(contour)
if y > delta and y < H-2*delta-1 and x > delta and x < W-2*delta1:
imgGrayRoi = imgGray[y-delta:y+h+2*delta, xdelta:x+w+2*delta]
imgBinRoi = imgBin[y-delta:y+h+2*delta, xdelta:x+w+2*delta]
features.append([objectContrast(imgBinRoi, imgGrayRoi, contour),
borderContrast(imgBinRoi, imgGrayRoi, contour),
img3(imgBin, imgGray, contour)])
return features
def testClassifier(modelName, imgName):
model = load_model(modelName)
imgGray = cv2.imread(imgName)
imgMarked = imgGray
contours, hierarchy, imgBin = preprocess(imgGray)
removeInvalidContours(imgBin, contours, 50, 1000)
contours, hierarchy = cv2.findContours(imgBin, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
delta = 5
H, W, channels = imgGray.shape
contInd = 0
T_P = 0
F_P = 0
for contour in contours:
x,y,w,h = cv2.boundingRect(contour)
if y > delta and y < H-2*delta-1 and x > delta and x < W-2*delta1:
imgGrayRoi = imgGray[y-delta:y+h+2*delta, xdelta:x+w+2*delta]
imgBinRoi = imgBin[y-delta:y+h+2*delta, xdelta:x+w+2*delta]
feature = [objectContrast(imgBinRoi, imgGrayRoi, contour),
borderContrast(imgBinRoi, imgGrayRoi,contour),
img3(imgBin, imgGray, contour)]
res = model.predict(np.array([feature]))
if res[0,0] > res[0,1]:
T_P += 1
v2.drawContours(imgMarked, contours, contInd, color=(0,255,0),
thickness = 1, lineType = 0)
else:
F_P += 1
cv2.drawContours(imgMarked, contours, contInd, color=(0,0,255),
thickness = 1, lineType = 0)
contInd += 1
print('Marked')
cv2_imshow(imgMarked)
return T_P, F_P
def trainClassifier(modelName, features1, features2):
train = []
label = []
label1 = [1, 0]
label2 = [0, 1]
npa1 = np.array(features1)
npa2 = np.array(features2)
for i in range(len(npa1[:,0])):
train.append(npa1[i,:])
label.append(label1)
for i in range(len(npa2[:,0])):
train.append(npa2[i,:])
label.append(label2)
train = np.array(train)
label = np.array(label)
rng_state = np.random.get_state()
np.random.shuffle(train)
np.random.set_state(rng_state)
np.random.shuffle(label)
model = Sequential()
model.add(Dense(7, input_dim=3, activation='relu'))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(train, label, epochs=100, verbose=1)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
model.save(modelName)
print(model.get_weights())
def displayFeatures(features1, features2):
plt.subplot(2, 2, 1)
plt.title('ObjectContrast')
npa1 = np.array(features1)
npa2 = np.array(features2)
args = dict(histtype='stepfilled', alpha=0.7, bins=50)
plt.hist(npa1[:,0], **args)
plt.hist(npa2[:,0], **args)
plt.subplot(2, 2, 2)
plt.title('MorphGradient')
plt.hist(npa1[:,1], **args)
plt.hist(npa2[:,1], **args)
plt.subplot(2, 2, 3)
plt.title('Density')
plt.hist(npa1[:,2], **args)
plt.hist(npa2[:,2], **args)
plt.show()
features1 = []
features2 = []
features1 = calculateFeatures('sample_data/gr1.png')
features2 = calculateFeatures('sample_data/ov1.png')
displayFeatures(features1, features2)
modelName = 'classifier.h5'
trainClassifier(modelName, features1, features2)
T_P, F_P = testClassifier(modelName, 'sample_data/mi1.png')
recall_1 = (T_P/(T_P + F_P))*100
features1_1 = []
features2_1 = []
features1_1 = calculateFeatures('sample_data/gr2.png')
features2_1 = calculateFeatures('sample_data/ov2.png')
displayFeatures(features1_1, features2_1)
modelName = 'classifier.h5'
trainClassifier(modelName, features1_1, features2_1)
F_N, T_N = testClassifier(modelName, 'sample_data/mi2.png')
recall_2 = (F_N/(T_N + F_N))*100
accuracy_1 = ((T_P+T_N)/(T_P + F_P + T_N + F_N))*100
accuracy_2 = ((T_N+T_N)/(T_P + F_P + T_N + F_N))*100
print(recall_1,recall_2,accuracy_1,accuracy_2)