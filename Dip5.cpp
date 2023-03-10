//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip5.h"


namespace dip5 {
    

/**
* @brief Generates gaussian filter kernel of given size
* @param kSize Kernel size (used to calculate standard deviation)
* @returns The generated filter kernel
*/
cv::Mat_<float> createGaussianKernel1D(float sigma)
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
    cv::Mat kernel = cv::Mat::ones(1, kSize, CV_32FC1);
    int mean = kSize / 2;

    for (int i = 0; i < kSize; i++)
    {
        kernel.at<float>(0, i) = 1 / (2 * sigma * CV_PI) * exp(-0.5 * ((i - mean) * (i - mean) / (sigma * sigma)));
    }

    kernel *= (1 / sum(kernel)[0]);

    return kernel;
}

/**
* @brief Convolution in spatial domain
* @param src Input image
* @param kernel Filter kernel
* @returns Convolution result
*/
cv::Mat_<float> spatialConvolution(const cv::Mat_<float> &src, const cv::Mat_<float> &kernel)
{
    int offsetRow = kernel.rows / 2;
    int offsetCol = kernel.cols / 2;

    cv::Mat flippedKernel;
    flip(kernel, flippedKernel, 0);
    flip(kernel, flippedKernel, 1);

    cv::Mat expandedImg;
    cv::copyMakeBorder(src, expandedImg, offsetRow, offsetRow, offsetCol, offsetCol, cv::BORDER_REPLICATE, 0);

    cv::Mat imgClone = src.clone();

    for (int i = offsetRow; i < expandedImg.rows - offsetRow; i++)
    {
        for (int j = offsetCol; j < expandedImg.cols - offsetCol; j++)
        {
            cv::Mat expandedImgFraction = expandedImg.rowRange(i - offsetRow, i + offsetRow + 1).colRange(j - offsetCol, j + offsetCol + 1).clone();

            expandedImgFraction = expandedImgFraction.mul(flippedKernel);

            imgClone.at<float>(i - offsetRow, j - offsetCol) = sum(expandedImgFraction)[0];
        }
    }

    return imgClone;
}
        

/**
* @brief Convolution in spatial domain by seperable filters
* @param src Input image
* @param size Size of filter kernel
* @returns Convolution result
*/
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernelX, const cv::Mat_<float>& kernelY)
{
    // Hopefully already DONE, copy from last homework
    // But do mind that this one gets two different kernels for horizontal and vertical convolutions.
    cv::Mat imgClone = spatialConvolution(src, kernelX);
    transpose(imgClone, imgClone);

    imgClone = spatialConvolution(imgClone, kernelY);
    transpose(imgClone, imgClone);

    return imgClone;
}

    
/**
 * @brief Creates kernel representing fst derivative of a Gaussian kernel (1-dimensional)
 * @param sigma standard deviation of the Gaussian kernel
 * @returns the calculated kernel
 */
cv::Mat_<float> createFstDevKernel1D(float sigma) 
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);

    cv::Mat kernel = cv::getGaussianKernel(kSize, sigma);

    cv::Mat gaussian_first_deriv_x = cv::Mat::zeros(1, kSize, CV_64FC1);

    int half_kernel_size = kSize / 2;
    for (int i = 0; i < kSize; i++) {
        int x = -half_kernel_size + i;
        double factor = -x / (sigma * sigma);
        gaussian_first_deriv_x.at<double>(i) = kernel.at<double>(i) * factor;
    }

    return gaussian_first_deriv_x;
}


/**
 * @brief Calculates the directional gradients through convolution
 * @param img The input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param gradX Matrix through which to return the x component of the directional gradients
 * @param gradY Matrix through which to return the y component of the directional gradients
 */
void calculateDirectionalGradients(const cv::Mat_<float>& img, float sigmaGrad,
                            cv::Mat_<float>& gradX, cv::Mat_<float>& gradY)
{
    // TO DO !!!
    gradX.create(img.rows, img.cols);
    gradY.create(img.rows, img.cols);
    //to calculate derivative of Gaussian and normal Gaussian
    cv::Mat_<float>kernel_norm = createGaussianKernel1D(sigmaGrad);
    cv::Mat_<float>kernel_dev = createFstDevKernel1D(sigmaGrad);
    //computes directional gradient via separable convolution
    gradX = separableFilter(img, kernel_dev, kernel_norm);
    gradY = separableFilter(img, kernel_norm, kernel_dev);
}

/**
 * @brief Calculates the structure tensors (per pixel)
 * @param gradX The x component of the directional gradients
 * @param gradY The y component of the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation".
 * @param A00 Matrix through which to return the A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 Matrix through which to return the A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 Matrix through which to return the A_{1,1} elements of the structure tensor of each pixel.
 */
void calculateStructureTensor(const cv::Mat_<float>& gradX, const cv::Mat_<float>& gradY, float sigmaNeighborhood,
                            cv::Mat_<float>& A00, cv::Mat_<float>& A01, cv::Mat_<float>& A11)
{
    cv::Mat_<float> gaussianKernel = createGaussianKernel1D(sigmaNeighborhood);

	A00	= separableFilter(gradX.mul(gradX), gaussianKernel, gaussianKernel);
	A11	= separableFilter(gradY.mul(gradY), gaussianKernel, gaussianKernel);
	A01	= separableFilter(gradX.mul(gradY), gaussianKernel, gaussianKernel);
}

/**
 * @brief Calculates the feature point weight and isotropy from the structure tensors.
 * @param A00 The A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 The A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 The A_{1,1} elements of the structure tensor of each pixel.
 * @param weight Matrix through which to return the weights of each pixel.
 * @param isotropy Matrix through which to return the isotropy of each pixel.
 */
void calculateFoerstnerWeightIsotropy(const cv::Mat_<float>& A00, const cv::Mat_<float>& A01, const cv::Mat_<float>& A11,
                                    cv::Mat_<float>& weight, cv::Mat_<float>& isotropy)
{
    // TO DO !!!
    weight.create(A00.rows, A00.cols);
    isotropy.create(A00.rows, A00.cols);

    for (int i = 0; i < A00.rows; i++) {
        for (int j = 0; j < A00.cols; j++) {
            double a00 = A00(i, j);
            double a01 = A01(i, j);
            double a11 = A11(i, j);
            double det = a00 * a11 - a01 * a01;
            double trace = a00 + a11;
            if (trace == 0)  {
                weight(i, j) = 0; 
                isotropy(i, j) = 0;
            } else {
                weight(i, j) = det / trace;
                isotropy(i, j) = 4 * det / (trace * trace);
            }                     
        }
    }
}

/**
 * @brief Finds Foerstner interest points in an image and returns their location.
 * @param img The greyscale input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation" of the structure tensor.
 * @param fractionalMinWeight Threshold on the weight as a fraction of the mean of all locally maximal weights.
 * @param minIsotropy Threshold on the isotropy of interest points.
 * @returns List of interest point locations.
 */
std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float>& img, float sigmaGrad, float sigmaNeighborhood, float fractionalMinWeight, float minIsotropy)
{
    // TO DO !!!
    cv::Mat_<float> gradX, gradY;
    calculateDirectionalGradients(img, sigmaGrad, gradX, gradY);
    cv::Mat_<float> A00, A01, A11;
    calculateStructureTensor(gradX, gradY, sigmaNeighborhood, A00, A01, A11);
    cv::Mat_<float> weight, isotropy;
    calculateFoerstnerWeightIsotropy(A00, A01, A11, weight, isotropy);

    // Find local maxima using isLocalMaximum
    std::vector<cv::Vec2i> interestPoints;
    for (int i = 0; i < weight.rows; i++) {
        for (int j = 0; j < weight.cols; j++) {    
            float meanWeight = cv::mean(weight)[0];

            if (isLocalMaximum(weight, j, i) && weight(i, j) > fractionalMinWeight * meanWeight && isotropy(i, j) > minIsotropy) {
                interestPoints.push_back(cv::Vec2i(j, i));
            }
        }
    }

    return interestPoints;
}



/* *****************************
  GIVEN FUNCTIONS
***************************** */


// Use this to compute kernel sizes so that the unit tests can simply hard checks for correctness.
unsigned getOddKernelSizeForSigma(float sigma)
{
    unsigned kSize = (unsigned) std::ceil(5.0f * sigma) | 1;
    if (kSize < 3) kSize = 3;
    return kSize;
}

bool isLocalMaximum(const cv::Mat_<float>& weight, int x, int y)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            int x_ = std::min(std::max(x+j, 0), weight.cols-1);
            int y_ = std::min(std::max(y+i, 0), weight.rows-1);
            if (weight(y_, x_) > weight(y, x))
                return false;
        }
    return true;
}

}
