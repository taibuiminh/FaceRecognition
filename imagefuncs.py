import numpy as np
import math
from collections import namedtuple

# the only relevant file type is P2
FILETYPE = 'P2'

# structure for storing the data of a PGM file
PGMFile = namedtuple('PGMFile', 'max_shade, data')

# kernel K2 from class
K2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])


"""
The header of a PGM file has the form

P2
20 30
255

where P2 indicates a PGM file with ASCII encoding,
and the next three numbers are the number of columns in the image,
the number of rows in the image, and the maximum pixel value.

Comments are indicated by # and could occur anywhere in the file.
For simplicity, we assume that a # is preceded by whitespace.

The remaining entries are pixel values.

This function receives the name of a PGM file and returns the data in
the form of a PGMFile.
"""
def read_image(filename):
    
    with open(filename) as imagefile:
        # list to hold integer entries in the file
        entries = []
    
        firstword = True
        for line in imagefile:
            words = line.split()
            worditer = iter(words)
            comment = False
            endline = False
            while not comment and not endline:
                try:
                    word = next(worditer)
                    if not word.startswith('#'):
                        if not firstword:
                            # an entry that is not part of a comment and is not 
                            # the first word is an integer
                            entries.append(int(word))
                        else:
                            # the first word that is not a comment is the file type
                            assert word == FILETYPE, f"The only supported file type is P2."
                            firstword = False
                    else:
                        # this is a comment; drop the rest of the line
                        comment = True
                except StopIteration:
                    endline = True

    num_cols = entries[0] # number of columns in the image
    num_rows = entries[1] # number of rows in the image
    max_shade = entries[2] # maximum pixel value
        
    # all remaining integers are pixel values
    # arrange them in a numpy array using dimensions read from the header
    data = np.reshape(np.array(entries[3:]), (num_rows, num_cols))

    return PGMFile(max_shade, data)

def maxShade(image):
    return image.max_shade


"""
This function receives a file name and a PGMFile, and writes
a PGM file with the given data.

The pixel data must be in a NumPy array whose dimensions are the 
number of rows and number of columns in the image.  

Entries in the array will be cast to integers before being written
to the file, e.g. an entry of 9.9 will be written as 9.
"""
def write_image(filename, image):

    # read the dimensions of the image from the shape of the pixel data array
    num_rows, num_cols = image.data.shape

    # create the file header
    header = f"{FILETYPE}\n{num_cols} {num_rows}\n{image.max_shade}"

    # entries in the pixel data array are written to the file as integers
    np.savetxt(filename, image.data, fmt="%d", comments='', header=header)

    return


"""
Question 1 (a) and (b)

Given a standard deviation sigma, the formulas for the 1D and 2D Gaussian
distributions are

P(x) = [1/sqrt(2 pi sigma^2)] * exp[-x^2/(2 sigma^2)]

P(x,y) = [1/(2 pi sigma^2)] * exp[-(x^2+y^2)/(2 sigma^2)]
"""

# return the value of a 1D Gaussian at a point
def gaussianPt1D(x, sigma=1):
    A = 1/math.sqrt(2*math.pi*sigma*sigma)
    return math.exp(-(x*x)/(2*sigma*sigma))*A

# return the value of a 2D Gaussian at a point
def gaussianPt2D(x, y, sigma=1):
    A = 1/(2*math.pi*sigma*sigma)
    return math.exp(-(x*x+y*y)/(2*sigma*sigma))*A

# given the stddev sigma and the number of neighbors r, return a 1D Gaussian
# kernel of length 2r + 1, renormalized so the entries sum to 1
def gaussian1D(sigma=1, r=2):
    gaussArray = np.zeros(2*r+1)
    for i in range(r+1):
        # use symmetry of the Gaussian distribution under x -> -x
        gaussArray[r+i] = gaussArray[r-i] = gaussianPt1D(i,sigma)
    # rescale so the entries sum to 1
    return gaussArray/np.sum(gaussArray)

# given the stddev sigma and the number of neighbors r, return a 2D Gaussian
# kernel of size (2r + 1) x (2r + 1), renormalized so the entries sum to 1
def gaussian2D(sigma=1, r=2):
    gaussArray = np.zeros((2*r+1,2*r+1))
    for i in range(r+1):
        for j in range(r+1):
            # use symmetry of the 2D Gaussian distribution 
            # under x -> -x and y -> -y
            gaussArray[r+i,r+j] = gaussArray[r+i,r-j] = gaussArray[r-i,r+j] = gaussArray[r-i,r-j] = gaussianPt2D(i,j,sigma)
    # rescale so the entries sum to 1
    return gaussArray/np.sum(gaussArray)


"""
Question 1 (c), (d), (e)
We implement the 1D and 2D convolution operations with truncation
at the boundaries.
"""

"""
In the convolution operation, the current pixel is aligned with the center of
the kernel, which is assumed to have length 2r+1.  If the kernel extends past 
the boundary of the image, the kernel is truncated.  

current position
      |
   [p p p p] p p p ...
 k [k k k k]

The following functions return the left and right indices for the slice of 
the kernel and the slice of the pixel array.

In both functions, index refers to the current position within the pixel
array, maximum_index is length of the array, and margin is the value r
if the length of the kernel is 2r+1.
"""
def kernel_boundaries(index, maximum_index, margin):
    kernel_size = 2*margin+1
    # Indices for the slice of the kernel
    kl = max(0,margin-index)
    kr = kernel_size - max(index+margin+1-maximum_index,0)

    return kl, kr


def array_boundaries(index, maximum_index, margin):
    # Indices for the slice of the array
    pl = max(index-margin,0)
    pr = min(index+margin+1,maximum_index)

    return pl, pr


"""
This function implements the operation of convolving an array of pixel values
with a 1D kernel of length 2r+1, applied in the horizontal direction.

Position (j,k) in the array is always matched with the center of the kernel,
position r

If k is at least a distance r from the boundaries of the array, then the 
entire kernel can be matched to a 1 x (2r+1) slice of the array centered on
position (j,k).  We take this slice, multiply it component-wise by the 
kernel, and sum the results.  This number becomes entry (j,k) in the 
convolved image.
    
If position (j,k) is such that a 1 x (2r+1) slice of the array centered on 
this position would exceed the boundaries of the array, then we adjust the 
slice to stop at the boundaries, and we truncate the kernel to match.

For example, if (j,k) = (0,0), then the slice of the array would be
array[0, 0:r+1], and the corresponding slice of the kernel would be
kernel[r:2r+1].

At the moment, the only available boundary condition is to renormalize the 
truncated kernel to sum to 1.  However, this may change in future assignments!
"""
def convolve1D_horizontal(data, kernel, boundary_option="renormalize"):
    # Assumes the length of the kernel has the form 2r+1
    kernel_size = kernel.shape[0]
    margin = int((kernel_size-1)/2)

    # Get the dimensions of the pixel array
    numrows, numcols = data.shape

    # The result has the same dimensions as the pixel array
    conv_array = np.zeros((numrows, numcols))

    for ak in range(numcols):
        # Indices for the slices of the pixel array and the kernel
        pl, pr = array_boundaries(ak, numcols, margin)

        if margin <= ak and ak < numcols-margin:
            # no adjustments to the kernel are needed
            for aj in range(numrows):
                # Take the dot product of the slice of the array with the kernel
                conv_array[aj,ak] = np.dot(data[aj,pl:pr],kernel)
                
        else:
            kl, kr = kernel_boundaries(ak, numcols, margin)
            # various boundary options
            if boundary_option == "renormalize":
                # for use if kernel entries are positive and sum to 1
                # the truncated kernel is renormalized to sum to 1
                kernel_sum = np.sum(kernel[kl:kr])
                for aj in range(numrows):
                    conv_array[aj,ak] = np.dot(data[aj,pl:pr],kernel[kl:kr]/kernel_sum)
                
    return np.rint(conv_array)


"""
This function implements the operation of convolving an array of pixel values
with a 2D kernel of dimensions (2r+1) x (2r+1).

Position (j,k) in the array is always matched with the center of the kernel,
position (r,r).

If j and k are at least a distance r from the boundaries of the array, then
the entire kernel can be matched to a (2r+1) x (2r+1) slice of the array
centered on position (j,k).  We take this slice, multiply it component-wise
by the kernel, and sum the results.  This number becomes entry (j,k) in
the convolved image.
    
If position (j,k) is such that a (2r+1) x (2r+1) slice of the array
centered on this position would exceed the boundaries of the array, then we
adjust the slice to stop at the boundaries, and we truncate the kernel to 
match.

For example, if (j,k) = (0,0), then the slice of the array would be
array[0:r+1, 0:r+1], and the corresponding slice of the kernel would be
kernel[r:2r+1, r:2r+1].

Multiple options, determined by keyword, are available for the boundaries
of the image.

The resulting array is rounded to the nearest integer before it is returned.
"""
def convolve2D_array(data, kernel, boundary_option="renormalize"):
    # Assumes that the dimensions of the kernel have the form (2r+1) x (2r+1)
    kernel_size = kernel.shape[0]
    margin = int((kernel_size-1)/2)
    
    # Get the dimensions of the pixel array
    numrows, numcols = data.shape

    # The result has the same dimensions as the pixel array
    conv_array = np.zeros((numrows, numcols))

    # Iterate over the pixel array to perform the convolution
    for aj in range(numrows):
        # First indices for the slice of the pixel array
        pu, pd = array_boundaries(aj, numrows, margin)

        for ak in range(numcols):    
            # Second indices for the slice of the pixel array
            pl, pr = array_boundaries(ak, numcols, margin)

            if margin <= aj and aj < numrows-margin and margin <= ak and ak < numcols-margin:
                # no adjustments to the kernel are needed
                conv_array[aj,ak] = np.sum(data[pu:pd,pl:pr]*kernel)
            else:
                # indices for the slice of the kernel
                ku, kd = kernel_boundaries(aj, numrows, margin)
                kl, kr = kernel_boundaries(ak, numcols, margin)

                # various boundary options
                if boundary_option == "renormalize":
                    # for use if kernel entries are positive and sum to 1
                    # the truncated kernel is renormalized to sum to 1
                    kernel_sum = np.sum(kernel[ku:kd,kl:kr])
                    conv_array[aj,ak] = np.sum(data[pu:pd,pl:pr]*kernel[ku:kd,kl:kr]/kernel_sum)

                elif boundary_option == "center-shift":
                    # for use with kernels K1 and K2 from class
                    # the center entry of the kernel is shifted
                    # to preserve the kernel sum
                    temp_kernel = kernel.copy()
                    kernel_sum = np.sum(temp_kernel)
                    new_sum = np.sum(temp_kernel[ku:kd,kl:kr])
                    temp_kernel[margin, margin] -= (new_sum - kernel_sum)
                    conv_array[aj,ak] = np.sum(data[pu:pd,pl:pr]*temp_kernel[ku:kd,kl:kr])
              
    return np.rint(conv_array)
   

"""
Receives an image as a PGMFile and a (2r+1)x(2r+1) kernel as a NumPy array.
Returns the PGMFile containing the convolved result.

This operation is suitable for performing 2D weighted averages.  The entries
in the kernel must be positive and sum to 1.  At the boundaries, the kernel is
truncated and the remaining entries are renormalized to sum to 1.
"""
def convolve2D(image, kernel):
    newdata = convolve2D_array(image.data, kernel, boundary_option="renormalize")
    return PGMFile(image.max_shade, newdata)


"""
Receives an image as a PGMFile and a kernel of length 2r+1 as a NumPy array.
Returns the PGMFile containing the result of convolving the given image with
the kernel twice:  once in the horizontal direction and once in the vertical
direction.

This operation is suitable for performing weighted averages with kernels that
are separable.  At the boundaries, the kernel is truncated and the remaining 
entries are renormalized to sum to 1.
"""
def convolve2D_separable(image, kernel):
    step1 = convolve1D_horizontal(image.data,kernel)
    step2 = (convolve1D_horizontal(step1.T,kernel)).T
    return PGMFile(image.max_shade, step2)


"""
Question 2

Receives an image as a PGMFile and performs edge detection by convolving with 
the kernel K2 given in class.  See the function convolve2D_array() for the
appropriate boundary conditions.

After convolution, take the absolute value of  the array, then find the 
maximum value in the result.

Different options are possible for the rest of the image processing.  I have
found by trial and error that if the maximum value in the convolved array is
larger than the original maximum pixel value, then a better image of the edges
results if I clip the array.  On the other hand, if the maximum value in the 
convolved array is smaller than the original maximum pixel value, then the
maximum shade in the resulting image should be set to that smaller value.

You should do *at least one* of clipping the pixel array and resetting the
maximum pixel value.
"""
def edge_detect_K2(image):
    conv2array = convolve2D_array(image.data, K2, boundary_option="center-shift")
    conv2array = np.fabs(conv2array)
    new_max = int(np.amax(conv2array))
    if new_max > image.max_shade:
        conv2array = np.clip(conv2array, 0, image.max_shade)
        new_max = image.max_shade
    return PGMFile(new_max, conv2array)





sobelFilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

#apply sobel filter in vertical direction
def sobel_edge_detection_vertical(image):
    conv2array = convolve2D_array(image.data, sobelFilter, boundary_option="center-shift")
    conv2array = np.fabs(conv2array)
    new_max = int(np.amax(conv2array))
    if new_max > image.max_shade:
        conv2array = np.clip(conv2array, 0, image.max_shade)
        new_max = image.max_shade
    return PGMFile(new_max, conv2array)

#apply sobel filter in horizontal direction
def sobel_edge_detection_horizontal(image):
    conv2array = convolve2D_array(image.data, np.flip(sobelFilter.T, axis = 0), boundary_option="center-shift")
    conv2array = np.fabs(conv2array)
    new_max = int(np.amax(conv2array))
    if new_max > image.max_shade:
        conv2array = np.clip(conv2array, 0, image.max_shade)
        new_max = image.max_shade
    return PGMFile(new_max, conv2array)

#get gradient magnitude using the formula 
def gradient_magnitude(imageX, imageY, maxShade):
    gradient_magnitude = np.sqrt(np.square(imageX.data) + np.square(imageY.data))
    gradient_magnitude *= 255 / gradient_magnitude.max()
    return gradient_magnitude

#return image with gradient_magnitude data
def convert_gradient_magnitude_2PGM(gradient_magnitude):
    return PGMFile(255, gradient_magnitude)

# theta is the  angle of the  gradient  vector , between 0 and pi
def convert_to_degree(imageX, imageY):
    theta = np.arctan2(imageX.data, imageY.data)
    # theta = np.rad2deg(theta)
    theta += 180
    return theta

#Sort  theta  into  one of four  bins  and  perform  the  corresponding  test
def non_max_suppression(gradient_magnitude, theta, maxShade):
 
    image_r, image_c = gradient_magnitude.shape
 
    retMatrix = np.zeros(gradient_magnitude.shape)
 
    for r in range(1, image_r - 1):
        for c in range(1, image_c - 1):
            direction_of_gradient = theta[r, c]
            if  (9*np.pi/8 <= direction_of_gradient < 11*np.pi/8) or (np.pi/8 <= direction_of_gradient <3 * np.pi/8):
                first_pixel = gradient_magnitude[r+1, c-1]
                second_pixel = gradient_magnitude[r-1, c+1]
 
            elif (11*np.pi/8 <= direction_of_gradient < 13*np.pi/8) or (3*np.pi/8 <= direction_of_gradient < 5*np.pi/8):
                first_pixel = gradient_magnitude[r-1, c]
                second_pixel = gradient_magnitude[r+1, c]

            elif (15 * np.pi/8 <= direction_of_gradient <= 2*np.pi) or (0 <= direction_of_gradient < np.pi/8):
                first_pixel = gradient_magnitude[r, c-1]
                second_pixel = gradient_magnitude[r, c+1]
 
            else:
                first_pixel = gradient_magnitude[r-1, c-1]
                second_pixel = gradient_magnitude[r+1, c+1]
 
            if gradient_magnitude[r, c] >= first_pixel and gradient_magnitude[r, c] >= second_pixel:
                retMatrix[r, c] = gradient_magnitude[r, c]

    return PGMFile(maxShade, retMatrix)


#Apply double threshold to determine potential edges
def threshold(image, low_threshold, high_threshold, pixel, maxShade):
    retMatrix = np.zeros(image.data.shape)
    strong = maxShade
    strong_r, strong_c = np.where(image.data >= high_threshold)
    weak_r, weak_c = np.where((image.data <= high_threshold) & (image.data >= low_threshold))
    retMatrix[strong_r, strong_c] = strong
    retMatrix[weak_r, weak_c] = pixel
    return PGMFile(maxShade, retMatrix)

#Track edge by hysteresis: Finalize the detection of edges 
# by suppressing all the other edges that are weak and not connected to strong edges.
def hysteresis(image, weak, maxShade):
    image_r, image_c = image.data.shape
 
    cloneLeftRight = image.data.copy()
    for r in range(image_r - 1, 0, -1):
        for c in range(1, image_c):
            if cloneLeftRight[r, c] == weak:
                if cloneLeftRight[r, c+1] == maxShade or cloneLeftRight[r, c-1] == maxShade or cloneLeftRight[r-1, c] == maxShade or cloneLeftRight[
                    r+1, c] == maxShade or cloneLeftRight[
                    r-1, c-1] == maxShade or cloneLeftRight[r+1, c-1] == maxShade or cloneLeftRight[r-1, c+1] == maxShade or cloneLeftRight[
                    r+1, c+1] == maxShade:
                    cloneLeftRight[r, c] = maxShade
                else:
                    cloneLeftRight[r, c] = 0


    cloneRightLeft = image.data.copy()
    for r in range(1, image_r):
        for c in range(image_c -1, 0, -1):
            if cloneRightLeft[r, c] == weak:
                if cloneRightLeft[r, c+1] == maxShade or cloneRightLeft[r, c-1] == maxShade or cloneRightLeft[r-1, c] == maxShade or cloneRightLeft[r+1, c] == maxShade or cloneRightLeft[
                    r-1, c-1] == maxShade or cloneRightLeft[r+1, c-1] == maxShade or cloneRightLeft[r-1, c+1] == maxShade or cloneRightLeft[r+1, c+1] == maxShade:
                    cloneRightLeft[r, c] = maxShade
                else:
                    cloneRightLeft[r, c] = 0
 
    cloneDownUP = image.data.copy()
    for r in range(image_r - 1, 0, -1):
        for c in range(image_c - 1, 0, -1):
            if cloneDownUP[r, c] == weak:
                if cloneDownUP[r, c + 1] == maxShade or cloneDownUP[r, c-1] == maxShade or cloneDownUP[r-1, c] == maxShade or cloneDownUP[r+1, c] == maxShade or cloneDownUP[
                    r-1, c-1] == maxShade or cloneDownUP[r+1, c-1] == maxShade or cloneDownUP[r-1, c+1] == maxShade or cloneDownUP[r+1, c+1] == maxShade:
                    cloneDownUP[r, c] = maxShade
                else:
                    cloneDownUP[r, c] = 0
    
    cloneUPDown = image.data.copy()
    for r in range(1, image_r):
        for c in range(1, image_c):
            if cloneUPDown[r, c] == weak:
                if cloneUPDown[r, c+1] == maxShade or cloneUPDown[r, c-1] == maxShade or cloneUPDown[r-1, c] == maxShade or cloneUPDown[r+1, c] == maxShade or cloneUPDown[
                    r-1, c-1] == maxShade or cloneUPDown[r+1, c-1] == maxShade or cloneUPDown[r-1, c+1] == maxShade or cloneUPDown[r+1, c+1] == maxShade:
                    cloneUPDown[r, c] = maxShade
                else:
                    cloneUPDown[r, c] = 0
    
    hyster = cloneUPDown + cloneDownUP + cloneRightLeft + cloneLeftRight
    hyster[hyster > maxShade] = maxShade
 
    return PGMFile(maxShade, hyster)


def crop(image, size):
    new_data = np.zeros(image.data.shape)
    image_r, image_c = image.data.shape
    left_space = (image_r - size)/2  
    for i in range(0, image_r):
        for j in range(0, image_c):
            if( image_r - left_space > i > left_space and image_r - left_space > j > left_space and i):
                new_data[i][j] = image.data[i][j]
    # new_data.reshape()
    return PGMFile(image.max_shade, new_data)





