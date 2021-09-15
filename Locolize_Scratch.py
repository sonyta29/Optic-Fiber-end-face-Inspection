import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image 
import imutils
import glob 
## (1) Read
img1 = cv2.imread("15.png")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

## (2) Threshold
th, threshed = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

## (3) Find the first contour that greate than 100, locate in centeral region
## Adjust the parameter when necessary
cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key=cv2.contourArea)
H,W = img1.shape[:2]
for cnt in cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > 100 and (0.7 < w/h < 1.3) and(W/4 < x + w//2 < W*3/4) and (H/4 < y + h//2 < H*3/4):
        break

## (4) Create mask and do bitwise-op
mask = np.zeros(img1.shape[:2],np.uint8)
cv2.drawContours(mask, [cnt],-1, 255, -1)
dst = cv2.bitwise_and(img1, img1, mask=mask)


gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# threshold input image using otsu thresholding as mask and refine
#with morphology
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
kernel = np.ones((9,9), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# put thresh into 
result = dst.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask


gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 200)
blurred = cv2.GaussianBlur(edged, (5, 5), 0)
thresh_1 = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


# find contours in the thresholded image
cnts = cv2.findContours(thresh_1.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)


# loop over the contours
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the contour and center of the shape on the image
	cv2.drawContours(result, [c], -1, (0, 255, 0), 3)
	cv2.circle(result, (cX, cY), 7, (255, 255, 255), -1)
	cv2.putText(result, "center", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	#print( "the center's coordinates x and y are ",cX,"and", cY)
	

sobelX= cv2.Sobel(gray, cv2.CV_64F, 1,0)
sobelY= cv2.Sobel(gray, cv2.CV_64F, 0,1)

sobelX=np.uint8(np.absolute(sobelX))
sobelY=np.uint8(np.absolute(sobelY))

sobelCombined=cv2.bitwise_or(sobelX,sobelY)

gblur = cv2.GaussianBlur(sobelCombined, (1,1),0)

_, th1= cv2.threshold(sobelCombined,50,255, cv2.THRESH_BINARY)
cv2.imwrite('th1.png',th1)


img = cv2.imread('th1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)



rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 120# minimum number of pixels making up a line
max_line_gap =20   # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on the image 

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
          cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),2,4)
        

# Draw the lines on the  image
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)


line_image2 = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(line_image2, cv2.COLOR_BGR2HSV)
lower_range = np.array([110,50,50])
upper_range = np.array([130,255,255])
mask3 = cv2.inRange(hsv, lower_range, upper_range)


img=cv2.circle(line_image,(cX,cY), 13, (255,0,0), 2)
img=cv2.circle(line_image,(cX,cY), 58, (0,255,0), 2)
img=cv2.circle(line_image,(cX,cY), 68, (100,150,200), 2)


line_image3 = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
mask33 = cv2.cvtColor(mask3, cv2.COLOR_BGR2RGB)
ig1= cv2.resize(line_image3,(300,229))

ig2= cv2.resize(mask33,(300,229))

# compute difference
difference = cv2.subtract(ig1, ig2)

# color the mask red
Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
difference[mask != 255] = [0, 0, 255]
i= cv2.resize(difference,(229,229))
cv2.imwrite('difference.png',i)


src = cv2.imread('difference.png', cv2.IMREAD_COLOR)

#Transform source image to gray if it is not already
if len(src.shape) != 2:
    gray66 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
else:
    gray66 = src

_, th11= cv2.threshold(gray66,50,255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(th11, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]

for i, c in enumerate(contours):
        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0 and cv2.arcLength(c,False)> 300 :
                 (x,y),radius = cv2.minEnclosingCircle(c)
                 center =  (int(x),int(y))
                 radius = int(radius)
                 img = cv2.circle(src,center,radius,(0,255,0),2)
      
        elif  hierarchy[i][2] < 0 and hierarchy[i][3] < 0 and cv2.arcLength(c,False)<60:
                (x,y),radius = cv2.minEnclosingCircle(c)
                center = (int(x),int(y))
                radius = int(radius)
                img = cv2.circle(src,center,radius,(0,255,0),2)
                
        elif cv2.arcLength(c,True)> 300 or  cv2.arcLength(c,True)<100 :
             
             (x,y),radius = cv2.minEnclosingCircle(c)
             center = (int(x),int(y))
             radius = int(radius)
             img = cv2.circle(src,center,radius,(255,255,0),2)
             
              


                

src2 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
hsv2 = cv2.cvtColor(src2, cv2.COLOR_BGR2HSV)
lower_range = np.array([22, 93, 0])
upper_range = np.array([45, 255, 255])
mask44 = cv2.inRange(hsv2, lower_range, upper_range)
kernal = np.ones((3,3), np.uint8)
closing = cv2.morphologyEx(mask44, cv2.MORPH_CLOSE, kernal)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernal)
contours, hierarchy = cv2.findContours(opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
        
        perimeter = cv2.arcLength(c,True)
 
        if  70 <perimeter <200 or  360 <perimeter <400 :
             print('zone A with or without B IS SAFE  ')
             print('\n')
             print ('**********PASS*************')
             break
        elif 70 <perimeter <200 or  410 <perimeter <490 :
             print ('zone A and zone C  are safe ')
             print('\n')
             print ('***********PASS************')
             break
        elif 360 <perimeter <490 or  410 <perimeter <490 :
             print ('zone A is not safe')
             print('\n')
             print ('***********FAIL*************')
             break
        else :
             print ('***********FAIL*************')
             break

if cv2.countNonZero(opening) == 0:
    print('************FAIL***************')
           
titles=['original image','crop with identifying the center','filtring',
        'thresholding','scratch','mask','difference','src','mask44']
images=[img1 ,result,sobelCombined,th1,line_image,mask33,difference,src,opening]

for i in range(9) :
    plt.subplot(3,3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


 
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()