/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 2: Content-Based Image Retrieval

  feature extraction implementations.
*/

#include "features.h"

int computeBaseline(const cv::Mat &src, std::vector<float> &feat)
{
    if (src.empty() || src.rows < 7 || src.cols < 7)
    {
        return -1;
    }

    // compute center coordinates of the 7x7 patch
    int centerRow = src.rows / 2;
    int centerCol = src.cols / 2;
    int startRow = centerRow - 3; // 7x7 patch spans -3..+3 from center
    int startCol = centerCol - 3;

    // output: 7 * 7 * 3 = 147 floats, in row-major order
    feat.clear();
    feat.reserve(147);

    for (int r = 0; r < 7; r++)
    {
        const cv::Vec3b *row = src.ptr<cv::Vec3b>(startRow + r);
        for (int c = 0; c < 7; c++)
        {
            cv::Vec3b pixel = row[startCol + c];
            feat.push_back((float)pixel[0]); // B
            feat.push_back((float)pixel[1]); // G
            feat.push_back((float)pixel[2]); // R
        }
    }

    return 0;
}

int computeRGBHist(const cv::Mat &src, std::vector<float> &feat, int bins)
{
    if (src.empty())
        return -1;

    // allocate the flattened 3D histogram (length = bins^3, all zeros)
    int totalBins = bins * bins * bins;
    feat.assign(totalBins, 0.0f);

    // each channel value 0..255 maps to a bin index 0..bins-1.
    // bins = 8 means each bin covers a 32-value range (256 / 8).
    // use integer division: binIdx = pixelValue * bins / 256.
    //(not pixelValue / (256 / bins) - that breaks when bins doesn't divide 256.)
    for (int r = 0; r < src.rows; r++)
    {
        const cv::Vec3b *row = src.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; c++)
        {
            int bIdx = row[c][0] * bins / 256; // B
            int gIdx = row[c][1] * bins / 256; // G
            int rIdx = row[c][2] * bins / 256; // R

            // flatten the 3D index into a 1D position.
            // convention: feat[bIdx * bins*bins + gIdx * bins + rIdx]
            int flatIdx = bIdx * bins * bins + gIdx * bins + rIdx;
            feat[flatIdx] += 1.0f;
        }
    }

    return 0;
}

int computeTopBottomHist(const cv::Mat &src, std::vector<float> &feat, int bins)
{
    if (src.empty())
        return -1;

    int totalBins = bins * bins * bins;
    int splitRow = src.rows / 2; // top half = rows [0, splitRow); bottom = [splitRow, rows)

    // allocate both histograms in one flat vector. top first, bottom second.
    feat.assign(2 * totalBins, 0.0f);

    for (int r = 0; r < src.rows; r++)
    {
        const cv::Vec3b *row = src.ptr<cv::Vec3b>(r);

        // decide whether this row belongs to the top or bottom histogram,
        // by picking the right offset into the flat vector.
        int offset = (r < splitRow) ? 0 : totalBins;

        for (int c = 0; c < src.cols; c++)
        {
            int bIdx = row[c][0] * bins / 256;
            int gIdx = row[c][1] * bins / 256;
            int rIdx = row[c][2] * bins / 256;
            int flatIdx = bIdx * bins * bins + gIdx * bins + rIdx;
            feat[offset + flatIdx] += 1.0f;
        }
    }

    return 0;
}

int computeColorTexture(const cv::Mat &src, std::vector<float> &feat,
                        int bins, int textureBins)
{
    if (src.empty())
        return -1;

    int colorSize = bins * bins * bins;
    feat.assign(colorSize + textureBins, 0.0f);

    // color histogram
    for (int r = 0; r < src.rows; r++)
    {
        const cv::Vec3b *row = src.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; c++)
        {
            int bIdx = row[c][0] * bins / 256;
            int gIdx = row[c][1] * bins / 256;
            int rIdx = row[c][2] * bins / 256;
            int flatIdx = bIdx * bins * bins + gIdx * bins + rIdx;
            feat[flatIdx] += 1.0f;
        }
    }

    // texture histogram from Sobel gradient magnitudes
    // convert to grayscale first so we get a single-channel magnitude image
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Sobel X and Y (signed 16-bit to hold negative gradient values)
    cv::Mat sx, sy;
    cv::Sobel(gray, sx, CV_16S, 1, 0, 3); // dx=1, dy=0 -> horizontal gradient
    cv::Sobel(gray, sy, CV_16S, 0, 1, 3); // dx=0, dy=1 -> vertical gradient

    // magnitude = sqrt(sx^2 + sy^2). cv::magnitude needs CV_32F inputs
    cv::Mat sxF, syF, mag;
    sx.convertTo(sxF, CV_32F);
    sy.convertTo(syF, CV_32F);
    cv::magnitude(sxF, syF, mag); // mag is CV_32F

    // bin the magnitudes
    for (int r = 0; r < mag.rows; r++)
    {
        const float *row = mag.ptr<float>(r);
        for (int c = 0; c < mag.cols; c++)
        {
            int v = (int)row[c];
            if (v < 0)
                v = 0;
            if (v > 255)
                v = 255;
            int idx = v * textureBins / 256;
            feat[colorSize + idx] += 1.0f;
        }
    }

    return 0;
}

int computeCustomDumpster(const cv::Mat &src, std::vector<float> &feat,
                          int bins, int textureBins)
{
    if (src.empty())
        return -1;

    // green ratio in HSV space
    // hue 35-85 (out of 180) covers saturated greens
    // saturation > 60 filters out near-gray pixels
    // value > 30 filters out near-black pixels
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    int greenPixels = 0;
    int totalPixels = hsv.rows * hsv.cols;
    for (int r = 0; r < hsv.rows; r++)
    {
        const cv::Vec3b *row = hsv.ptr<cv::Vec3b>(r);
        for (int c = 0; c < hsv.cols; c++)
        {
            uchar h = row[c][0];
            uchar s = row[c][1];
            uchar v = row[c][2];
            if (h >= 35 && h <= 85 && s > 60 && v > 30)
            {
                greenPixels++;
            }
        }
    }
    float greenRatio = (float)greenPixels / totalPixels;

    // compute color+texture histogram
    std::vector<float> ctFeat;
    computeColorTexture(src, ctFeat, bins, textureBins);

    // concatenate
    feat.clear();
    feat.reserve(1 + ctFeat.size());
    feat.push_back(greenRatio);
    for (float v : ctFeat)
        feat.push_back(v);

    return 0;
}