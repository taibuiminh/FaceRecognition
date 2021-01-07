import numpy as np
from numpy import linalg as la
import math
import imagefuncs as imf
import os
import sys
import time

SIZE = 250
MAX_SHADE = 255
BORDER = 20
PATH = 'duplicates/'
# PATH = 'library/'

#do nothing 
def test1(img):
    return img

#smoothing 
def test2(img):
    sigma = 1.5
    r = 3 
    gauss2D = imf.gaussian2D(sigma, r)
    img = imf.convolve2D(img, gauss2D)
    return img

#smoothing
def test3(img):
    sigma = 1.5
    r = 3 
    gauss2D = imf.gaussian2D(sigma, r)
    img = imf.convolve2D(img, gauss2D)
    return img

#smoothing
def test4(img):
    sigma = 3
    r = 3 
    gauss2D = imf.gaussian2D(sigma, r)
    img = imf.convolve2D(img, gauss2D)
    return img

#smoothing 
def test5(img):
    sigma = 6
    r = 3 
    gauss2D = imf.gaussian2D(sigma, r)
    img = imf.convolve2D(img, gauss2D)
    return img

#gradient only library
def test6(img):
    gradientX = imf.sobel_edge_detection_vertical(img)
    gradientY = imf.sobel_edge_detection_horizontal(img)
    img = imf.gradient_magnitude(gradientX, gradientY, MAX_SHADE)
    img = imf.convert_gradient_magnitude_2PGM(img)
    return img

#gradient 
def test7(img):
    gradientX = imf.sobel_edge_detection_vertical(img)
    gradientY = imf.sobel_edge_detection_horizontal(img)
    img = imf.gradient_magnitude(gradientX, gradientY, MAX_SHADE)
    img = imf.convert_gradient_magnitude_2PGM(img)
    return img

#gradient and smoothing
def test8(img):
    sigma = 6
    r = 3 
    gauss2D = imf.gaussian2D(sigma, r)
    img = imf.convolve2D(img, gauss2D)
    gradientX = imf.sobel_edge_detection_vertical(img)
    gradientY = imf.sobel_edge_detection_horizontal(img)
    img = imf.gradient_magnitude(gradientX, gradientY, MAX_SHADE)
    img = imf.convert_gradient_magnitude_2PGM(img)
    return img

#just cropping 
def test9(img):
    #crop image and leave 100 pixels in the middle
    img = imf.crop(img, 100)
    return img 

#cropping and gradient 
def test10(img):
    #crop image and leave 100 pixels in the middle
    img = imf.crop(img, 100)
    gradientX = imf.sobel_edge_detection_vertical(img)
    gradientY = imf.sobel_edge_detection_horizontal(img)
    img = imf.gradient_magnitude(gradientX, gradientY, MAX_SHADE)
    img = imf.convert_gradient_magnitude_2PGM(img)
    return img

def readFaces():
    file_names = os.listdir(PATH)
    d = np.zeros((SIZE**2, len(file_names)), dtype=np.int32)

    for i, filename in enumerate(file_names, start=0):
        if filename == ".DS_Store":
            continue

        img = imf.read_image(PATH + filename)

        # Image Manipulation
        if test == "test1":
            img = test1(img)

        if test == "test2":
            img = test2(img)

        if test == "test3":
            img = test3(img)

        if test == "test4":
            img = test4(img)

        if test == "test5":
            img = test5(img)

        if test == "test6":
            img = test6(img)

        if test == "test7":
            img = test7(img)

        if test == "test8":
            img = test8(img)

        if test == "test9":
            img = test9(img)

        if test == "test10":
            img = test10(img)
        # ================================

        max_shade, data = img
        reshaped = data.reshape(-1)
        d[:,i] = reshaped

    return d, file_names

def detectFace(filename):
    print("Reading faces...")
    d, file_names = readFaces()
    n = d.shape[1]
    print("Building library...")

    # Calculate the average column x
    mean = d.mean(axis=1)

    # Subtract x from every column of the d x n matrix
    # as a result we get a transpose of L
    LT = (d.transpose() - mean)

    # find L
    L = LT.transpose()

    # find LTL by matrix multiplication
    LTL = np.matmul(LT, L)

    # divide LTL by (n-1)
    multiplier = 1/(n-1)
    LTL = multiplier * LTL

    # find eigenfaces
    eigenfaces = findEigenFaces(LTL, L)


    # find weights
    weights = [0] * n

    for i in range(n):
        col_L = L[:,i]
        weights[i] = findWeight(eigenfaces, col_L)

    weights = np.array(weights)

    # Test an image
    print("Testing...\n")
    d, ind = testImage(eigenfaces, weights, mean, filename)

    print("The closest image is filename = " + file_names[ind])
    print(f"Distance is {d}")
    print("====================================")

# Step 2
# Find eigenvalues, eigenvectors, and corresponding eigenfaces
def findEigenFaces(covMatrix, L):

    # Now find the eigen values and the eigenvectors
    print("Calculating the eigen values and vectors...")

    eValues, eVectors = np.linalg.eig(covMatrix)
    np.round(eValues, 2)
    np.round(eVectors, 2)

    # Calculating the eigenvalues that make up 95 percent of the sum
    print("Calculating the eigenvalues that make up 95 percent of the sum...")
    total = sum(eValues)
    k = 0
    currentSum = 0
    for i in eValues:
        currentSum += i
        if currentSum/total < 0.95:
            k += 1
        else:
            break

    # Find eigenfaces
    eigenFaces = []

    for colNum in range(0, len(eVectors[0])):
        temp = eVectors[:, colNum:colNum+1]
        temp = np.matmul(L, temp)
        temp = temp / la.norm(temp)
        eigenFaces.append(temp)

    return np.array(eigenFaces)


# STEP 3 
# Finds weight vector for column Lj of matrix j
def findWeight(eigenFaces, Lj):
    wVector = [0] * len(eigenFaces)
    x = Lj.flatten()

    for i in range(0, len(eigenFaces)):
        #get column slice from eigenFaces and flatten
        vi = eigenFaces[i].flatten()
        wVector[i] = np.dot( x, vi )

    return np.array(wVector)

# STEP 4
# Read a test image, concatenate pixels
# Test if the image is a face
def testImage(eigenfaces, weights, mean, filename):
    img = imf.read_image(filename)

    # image Manipulation
    if test == "test1":
        img = test1(img)
    
    if test == "test3":
        img = test3(img)

    if test == "test4":
        img = test4(img)

    if test == "test5":
        img = test5(img)

    if test == "test7":
        img = test7(img)

    if test == "test8":
        img = test8(img)

    if test == "test9":
            img = test9(img)
    
    if test == "test10":
            img = test10(img)
    #=================================

    max_shade, data = img
    data = data.flatten()

    z = data - mean # subtract mean from image data

    w = findWeight(eigenfaces, z)

    distances = [0] * len(weights) # the distance vector

    for i in range(len(weights)):
        distances[i] = la.norm(weights[i] - w)

    d = np.amin(distances) # the minimal distance to a pic from library
    index = np.where(distances == d)[0][0]

    return d, index


file_to_test = "Ahmed_Chalabi_0002.pgm"
test = input("Please enter the test name that you want to see (for example: 'test9' ): ")

print("====================================")
print(f"Current test is {test}")
print("The file to test is:", file_to_test)
if len(sys.argv) > 1:
    file_to_test = sys.argv[1]

startTime = time.time()
detectFace(file_to_test)
