# Tests of Face recognition   
________________________________________________
            __December 9,2020__ 

In this report, I am going to apply different techniques from course SCI 2000 to improve the Face Recognition 
algorithm.
### Instrictions 
To run our code you will need __FaceRecognition.py__  __imagefuncs.py__  __library__  __duplicates__
And a test images 
Put test image file name in the code (line 273)
Run the program 
As input in terminal you can enter a test which you want to see {test1, test2, test3, test4, test5, test6, 
test7, test8, test9, test10}

The first step is to test the face recognition algorithm without any changes
## Test 1
* *Original Code* Usual conditions 
* Nothig was changed 
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Kristanna_Loken_0004.pgm__
* * Distance is __11242.608859289261__
* * Time is : __real    0m3.201s__

Right now we will try to apply smoothing only in the library of images. 
We will use kernel smoothing with sigma = 1.5 to convolve the image
## Test 2
* *Smoothing* only (library of images)
* __sigma = 1.5__   __r = 3__
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Kristanna_Loken_0004.pgm__
* * Distance is __10978.4480391684__
### Comments 
* Distance *decreased* by __264.1608201208619__ with original test 1
We can see that Distance is decreased, so it means that reducing noise in the image might help a lit bit 

Next try is to apply this technique both in the library and test image 
## Test 3
* *Smoothing* (library of images) && (Test image)
* __sigma = 1.5__   __r = 3__
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Kristanna_Loken_0004.pgm__
* * Distance is __10937.998919917623__
### Comments 
* Distance *decreased* by __304.60993937163767__ with original test 1
* Time complexity: to slow to process all images 
We can see that reducing noise in both sources help us to get better results 

In test 4 we will increase our sigma to get better results in reducing noise
## Test 4
* *Smoothing* (library of images) && (Test image)
* Increase sigma
* __sigma = 3__   __r = 3__
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Kristanna_Loken_0004.pgm__
* * Distance is __10747.203227966944__
### Comments 
* Distance *decreased* by __495.4056313223173__ with original test 1
* Time complexity: to slow to process all images
It worked, so can make a conclusion that noise reduction is a bit effective to modify the algorithm 


Test 5 is the same as test 4, it uses the same mechanism 
But we increase our sigma up to 6
## Test 5
* *Smoothing* (library of images) && (Test image)
* Increase sigma
* __sigma = 6__   __r = 3__
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Kristanna_Loken_0004.pgm__
* * Distance is __10721.366489957614__
### Comments 
* Distance *decreased* by __521.2423693316468__ with original test 1
* Time complexity: to slow to process all images __real    0m40.705s__
We could reduce distance by more than 500. 

In this step we will try to use different modify method which is called Gradient 
First of all try to apply it only with library of images and ignore test image manipulations
## Test 6
* Applying *Gradient* only (library of images)
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Gwendal_Peizerat_0001.pgm__
* * Distance is __19773.462847157403__
### Comments 
* Distance *increased* by __8530.853987868142__ with original test 1
* Time complexity: to slow to process all images 
As we can see, results are worse. It is because library right now has in gradient mod and test image is 
without any changes 

In test 7 we will apply the same technique but also with test image 
## Test 7
* Applying *Gradient* (library of images) && (Test image)
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Edwin_Edwards_0002.pgm__
* * Distance is __6589.427164615235__
### Comments 
* Distance *decreased* by __4653.181694674026__ with original test 1
* Time complexity: to slow to process all images __real    1m26.958s__
* Right now it is *the best* result, but to sloooooow
The problem of this method is time complexity. It takes a lot of time to procced all images 
However, we could reduce our distance by more than 4000 points 
Right now it is the best result 

We hope that applying both techniques from steps above will give us better results 
Gradient + smoothing 
## Test 8
* Applying *Gradient*  and *Smoothing* (library of images) && (Test image)
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Thabo_Mbeki_0003.pgm__
* * Distance is __7174.470337075414__
### Comments 
* Distance *decreased* by __4068.138522213847__ with original test 1
* Time complexity: to slow to process all images __real    2m0.559s__
* Worse than *Test 7*
Results are almost similar, but applying just gradient gave us better result 

## Conclusion 
After Tests using *Smoothing* and *Gradients* we can make a conclusion that __Test 7__ is the BEST option for
this time 

________________________________________________
            __December 11,2020__

After watching another group videos, we noted that cropping is a really good approach to get better results
in Face Recognition 

We decided to write a new function in imagefuncs.py to crop the image.
Right now we can detect our face in the middle of the image, but there are some minuses that we noted.
The problem is that we can only apply this technique to this library because all faces are in the middle. 
So taking photos from the internet would un efficient, because face could be in the corner for instance 
## Test 9
* Applying *Cropping* (library of images) && (Test image)
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Bernard_Lord_0002.pgm__
* * Distance is __2978.231144755921__
### Comments 
* Distance *decreased* by __8264.37771453334__ with original test 1
* Time complexity: right now it is pretty fast to process all images __real    0m4.278s__
* From now on it is *the best* result and in addition it runs faster then original 
So after applying this method we can conclude that cutting the useless space in the image is one of the most
important steps in a face recognition algorithm. To detect a face in the image we just need to have this face
without any details, like hair, clothes and other unnecessary stuff

We would like to notice that some faces have different angles so it makes our results worse

In the last test we would like to make a combination of test 7 and test 9, because they showed the best
results 
## Test 10
* Applying *Cropping* and *Gradient* (library of images) && (Test image)
* library of images in use: __dublicates/__
* The file to test is: __Ahmed_Chalabi_0002.pgm__
* * Results 
* * The closest image is filename = __Bernard_Lord_0002.pgm__
* * Distance is __3124.118358237748__
### Comments 
* Distance *decreased* by __8118.490501051513__ with original test 1
* Time complexity: right now it is not so slow to process all images __real    1m23.378s__
As we can see, the results is almost the same so we can make an assumption that this way is also actual 



# Tests using another images
* In this section we are trying to use our best approach *test 9* to test another images 
* We are using one image from __library/__ 
* Library to test is __dublicates/__

1. The file to test is: __Arantxa_Sanchez-Vicario_0002.pgm__
* * The closest image is filename = Michael_Chiklis_0003.pgm
* * Distance is __3580.313861061524__

2. The file to test is: __Jamir_Miller_0001.pgm__
* * The closest image is filename = Farouk_al-Sharaa_0003.pgm
* * Distance is __2709.566620047072__

3. The file to test is: __Erik_Morales_0003.pgm__
* * The closest image is filename = Tang_Jiaxuan_0011.pgm
* * Distance is __3054.640236265442__

4. The file to test is: __Hans-Christian_Schmid_0001.pgm__
* * The closest image is filename = Bob_Beauprez_0002.pgm
* * Distance is __3192.1368577824624__

5. The file to test is: __Ibrahim_Al-Marashi_0001.pgm__
* * The closest image is filename = Lynne_Cheney_0003.pgm
* * Distance is __3122.0529832193492__

6. The file to test is: __Bernard_Lord_0001.pgm__
* * The closest image is filename = Lynne_Cheney_0003.pgm
* * Distance is __3360.426299298347__

## Test results 
* Mean after all these six tests is equal to __3169.856142945699__

# Conclusion 
After all tests, we can conclude that cropping is the best way to get a better result. It helps us to cut
unnecessary stuff and make our program faster. We can also note that adding the gradient technique can make
the algorithm in some way effective because results showed a similar distance. However, implementing kernel
smoothing does not help detect the face. The minus of this approach is that distance was too large and time
complexity "can explode your computer".  