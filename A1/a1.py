import cv2
import numpy as np
from matplotlib import pyplot as plt



def logTransformed(imgPath, imgName):
    img = cv2.imread(imgPath+"/"+imgName)
    dummyValue = 0.00000001
    c = 255/(np.log(1 + dummyValue + np.max(img)))
    imgTransformed = c * np.log(1 + dummyValue + img)
    imgTransformedConst = np.log(1 + dummyValue + img)
    imgTransformed = np.array(imgTransformed, dtype = np.uint8)
    # imgTransformedConst = np.array(imgTransformedConst, dtype = np.uint8)
    imgPathOut = imgPath + "/logTransformed.tif"
    # imgPathOutConst = imgPath + "/logTransformedConst.tif"
    cv2.imwrite(imgPathOut, imgTransformed)
    # cv2.imwrite(imgPathOutConst, imgTransformedConst)

def powerTransformed(imgPath, imgName):
    img = cv2.imread(imgPath+"/"+imgName)
    c = 1
    # values from the slide
    for gamma in [0.3, 0.4, 0.6]:
        gammaCorrected = np.array(255*(img/255) ** gamma, dtype = 'uint8')
        imgPathOut = imgPath + "/gamma" + str(gamma)[-1] + ".tif"
        cv2.imwrite(imgPathOut, gammaCorrected)


# Reference: 
def historgramEqualized(imgPath, imgName):
    img = cv2.imread(imgPath+"/"+imgName, 0)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    # plot of original histogram
    plt.figure(0)
    # plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.title("Histogram of original image")

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

    # plot of transformation function
    plt.figure(1)
    plt.plot(cdf_m, color = 'b')
    plt.xlim([0,256])
    plt.title("Transformation Function")
    plt.xlabel("Input Intensity")
    plt.ylabel("Output Intensity")
    cdf = np.ma.filled(cdf_m,0).astype('uint8')


    # plot of enhanced image histogram
    img2 = cdf[img]
    hist,bins = np.histogram(img2.flatten(),256,[0,256])
    cdfImg2 = hist.cumsum()
    # cdfImg2_normalized = cdfImg2 * hist.max()/ cdfImg2.max()
    plt.figure(2)
    # plt.plot(cdfImg2_normalized, color = 'b')
    plt.hist(img2.flatten(),256,[0,256], color = 'r')
    plt.title("Histogram of equalized image")
    plt.xlim([0,256])


    imgPathOut = imgPath + "/historgramEqualized.tif"
    cv2.imwrite(imgPathOut, img2)


if __name__ == "__main__":
    imgPath = "/Users/zhesu/Documents/suzhe/playground"
    imgName = "308.tif"
    logTransformed(imgPath, imgName)
    powerTransformed(imgPath, imgName)
    historgramEqualized(imgPath, imgName)
    plt.show()
