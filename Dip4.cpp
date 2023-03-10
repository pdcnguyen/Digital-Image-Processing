//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip4.h"

namespace dip4
{

    using namespace std::complex_literals;
    using namespace std;
    using namespace cv;

    /*

    ===== std::complex cheat sheet =====

    Initialization:

    std::complex<float> a(1.0f, 2.0f);
    std::complex<float> a = 1.0f + 2.0if;

    Common Operations:

    std::complex<float> a, b, c;

    a = b + c;
    a = b - c;
    a = b * c;
    a = b / c;

    std::sin, std::cos, std::tan, std::sqrt, std::pow, std::exp, .... all work as expected

    Access & Specific Operations:

    std::complex<float> a = ...;

    float real = a.real();
    float imag = a.imag();
    float phase = std::arg(a);
    float magnitude = std::abs(a);
    float squared_magnitude = std::norm(a);

    std::complex<float> complex_conjugate_a = std::conj(a);

    */

    /**
     * @brief Computes the complex valued forward DFT of a real valued input
     * @param input real valued input
     * @return Complex valued output, each pixel storing real and imaginary parts
     */
    cv::Mat_<std::complex<float>> DFTReal2Complex(const cv::Mat_<float> &input)
    {
        cv::Mat_<std::complex<float>> result(input.rows, input.cols);
        dft(input, result, cv::DFT_COMPLEX_OUTPUT);
        return result;
    }

    /**
     * @brief Computes the real valued inverse DFT of a complex valued input
     * @param input Complex valued input, each pixel storing real and imaginary parts
     * @return Real valued output
     */
    cv::Mat_<float> IDFTComplex2Real(const cv::Mat_<std::complex<float>> &input)
    {
        cv::Mat_<float> result(input.rows, input.cols);
        idft(input, result, cv::DFT_COMPLEX_INPUT | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        return result;
    }

    /**
     * @brief Performes a circular shift in (dx,dy) direction
     * @param in Input matrix
     * @param dx Shift in x-direction
     * @param dy Shift in y-direction
     * @return Circular shifted matrix
     */
    cv::Mat_<float> circShift(const cv::Mat_<float> &in, int dx, int dy)
    {
        // 2 times % to get positive remainder number dx
        dx = ((dx % in.rows) + in.rows) % in.rows;
        // 2 times % to get positive remainder number dy
        dy = ((dy % in.cols) + in.cols) % in.cols;
        // new image dst
        cv::Mat dst = cv::Mat::zeros(in.size(), in.type());
        // transfer the row
        for (int i = 0; i < in.rows; i++)
        {
            int newrow = (i + dx) % in.rows;
            // transfer the col
            for (int j = 0; j < in.cols; j++)
            {
                int newcol = (j + dy) % in.cols;
                // circle shift
                dst.at<float>(newrow, newcol) = in.at<float>(i, j);
            }
        }
        // get the result image
        dst.copyTo(in);
        return in;
    }

    /**
     * @brief Computes the thresholded inverse filter
     * @param input Blur filter in frequency domain (complex valued)
     * @param eps Factor to compute the threshold (relative to the max amplitude)
     * @return The inverse filter in frequency domain (complex valued)
     */
    cv::Mat_<std::complex<float>> computeInverseFilter(const cv::Mat_<std::complex<float>> &input, const float eps)
    {
        float threshold, max = 0;
        cv::Mat Qu = input.clone();
        // get max amplitude
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                max = std::max(max, std::abs(input.at<std::complex<float>>(i, j)));
            }
        }
        // get threshold
        threshold = eps * max;
        // clipped inverse filter(frequencies define)
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                auto amp = std::abs(input.at<std::complex<float>>(i, j));
                if (amp < threshold)
                {
                    Qu.at<std::complex<float>>(i, j) = 1 / threshold;
                }
                else
                {
                    Qu.at<std::complex<float>>(i, j) = float(1) / input.at<std::complex<float>>(i, j);
                }
            }
        }
        return Qu;
    }

    /**
     * @brief Applies a filter (in frequency domain)
     * @param input Image in frequency domain (complex valued)
     * @param filter Filter in frequency domain (complex valued), same size as input
     * @return The filtered image, complex valued, in frequency domain
     */
    cv::Mat_<std::complex<float>> applyFilter(const cv::Mat_<std::complex<float>> &input, const cv::Mat_<std::complex<float>> &filter)
    {
        cv::Mat after = input.clone();
        // use mulSpectrums
        cv::mulSpectrums(input, filter, after, 0);
        return after;
    }

    /**
     * @brief Function applies the inverse filter to restorate a degraded image
     * @param degraded Degraded input image
     * @param filter Filter which caused degradation
     * @param eps Factor to compute the threshold (relative to the max amplitude)
     * @return Restorated output image
     */
    cv::Mat_<float> inverseFilter(const cv::Mat_<float> &degraded, const cv::Mat_<float> &filter, const float eps)
    {
        cv::Mat_<std::complex<float>> RestoredImage = cv::Mat::zeros(degraded.size(), degraded.type());
        RestoredImage = DFTReal2Complex(degraded);

        cv::Mat_<float> inverseFilter = cv::Mat::zeros(degraded.size(), degraded.type());
        filter.copyTo(inverseFilter(cv::Rect(0, 0, filter.cols, filter.rows)));

        inverseFilter = circShift(inverseFilter, -filter.rows / 2, -filter.cols / 2);
        inverseFilter = computeInverseFilter(DFTReal2Complex(inverseFilter), eps);
        RestoredImage = applyFilter(RestoredImage, inverseFilter);
        RestoredImage = IDFTComplex2Real(RestoredImage);
        return RestoredImage;
    }

    /**
     * @brief Computes the Wiener filter
     * @param input Blur filter in frequency domain (complex valued)
     * @param snr Signal to noise ratio
     * @return The wiener filter in frequency domain (complex valued)
     */
    cv::Mat_<std::complex<float>> computeWienerFilter(const cv::Mat_<std::complex<float>> &input, const float snr)
    {

        Mat_<std::complex<float>> filter = cv::Mat::zeros(input.size(), input.type());

        float a = 1.0f / snr;

        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                std::complex<float> val = input(i, j);
                float bottom = std::pow(std::abs(val), 2) + a * a;
                filter(i, j) = conj(val) / bottom;
            }
        }

        return filter;
    }

    /**
     * @brief Function applies the wiener filter to restore a degraded image
     * @param degraded Degraded input image
     * @param filter Filter which caused degradation
     * @param snr Signal to noise ratio of the input image
     * @return Restored output image
     */
    cv::Mat_<float> wienerFilter(const cv::Mat_<float> &degraded, const cv::Mat_<float> &filter, float snr)
    {
        cv::Mat_<std::complex<float>> RestoredImage = cv::Mat::zeros(degraded.size(), degraded.type());

        cv::Mat_<float> wienerFilter = cv::Mat::zeros(degraded.size(), degraded.type());
        filter.copyTo(wienerFilter(cv::Rect(0, 0, filter.cols, filter.rows)));

        wienerFilter = circShift(wienerFilter, -filter.rows / 2, -filter.rows / 2);
        wienerFilter = computeWienerFilter(DFTReal2Complex(wienerFilter), snr);

        RestoredImage = applyFilter(DFTReal2Complex(degraded), wienerFilter);

        return IDFTComplex2Real(RestoredImage);
    }

    /* *****************************
      GIVEN FUNCTIONS
    ***************************** */

    /**
     * function degrades the given image with gaussian blur and additive gaussian noise
     * @param img Input image
     * @param degradedImg Degraded output image
     * @param filterDev Standard deviation of kernel for gaussian blur
     * @param snr Signal to noise ratio for additive gaussian noise
     * @return The used gaussian kernel
     */
    cv::Mat_<float> degradeImage(const cv::Mat_<float> &img, cv::Mat_<float> &degradedImg, float filterDev, float snr)
    {

        int kSize = round(filterDev * 3) * 2 - 1;

        cv::Mat gaussKernel = cv::getGaussianKernel(kSize, filterDev, CV_32FC1);
        gaussKernel = gaussKernel * gaussKernel.t();

        cv::Mat imgs = img.clone();
        cv::dft(imgs, imgs, img.rows);
        cv::Mat kernels = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
        int dx, dy;
        dx = dy = (kSize - 1) / 2.;
        for (int i = 0; i < kSize; i++)
            for (int j = 0; j < kSize; j++)
                kernels.at<float>((i - dy + img.rows) % img.rows, (j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i, j);
        cv::dft(kernels, kernels);
        cv::mulSpectrums(imgs, kernels, imgs, 0);
        cv::dft(imgs, degradedImg, cv::DFT_INVERSE + cv::DFT_SCALE, img.rows);

        cv::Mat mean, stddev;
        cv::meanStdDev(img, mean, stddev);

        cv::Mat noise = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
        cv::randn(noise, 0, stddev.at<double>(0) / snr);
        degradedImg = degradedImg + noise;
        cv::threshold(degradedImg, degradedImg, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(degradedImg, degradedImg, 0, 0, cv::THRESH_TOZERO);

        return gaussKernel;
    }

}
