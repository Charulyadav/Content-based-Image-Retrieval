/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 2: Content-Based Image Retrieval

  walks a directory of images, computes features for each, and writes
  them to a CSV. The feature type is selected via command-line argument.

  usage:
    computeFeatures <image_dir> <feature_type> <output_csv>

  feature_type options:
    baseline   - 7x7 center patch
*/

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "features.h"
#include "csv_util.h"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0]
                  << " <image_dir> <feature_type> <output_csv>\n"
                  << "  feature_type: baseline, rgbhist, topbottom, colortexture, custom\n";
        return -1;
    }

    std::string imageDir = argv[1];
    std::string featureType = argv[2];
    std::string outputCsv = argv[3];

    if (featureType != "baseline" && featureType != "rgbhist" && featureType != "topbottom" &&
        featureType != "colortexture" && featureType != "custom")
    {
        std::cout << "Unknown feature type: " << featureType << "\n";
        return -1;
    }

    // iterate the directory
    namespace fs = std::filesystem;
    if (!fs::exists(imageDir))
    {
        std::cout << "Directory does not exist: " << imageDir << "\n";
        return -1;
    }

    int resetFile = 1; // first write clears the CSV; subsequent writes append
    int processed = 0;
    int failed = 0;

    for (const auto &entry : fs::directory_iterator(imageDir))
    {
        if (!entry.is_regular_file())
            continue;

        std::string ext = entry.path().extension().string();
        // only process common image types
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
            ext != ".JPG" && ext != ".JPEG" && ext != ".PNG")
            continue;

        std::string imagePath = entry.path().string();
        std::string filename = entry.path().filename().string();

        cv::Mat img = cv::imread(imagePath);
        if (img.empty())
        {
            std::cout << "Could not read " << imagePath << ", skipping\n";
            failed++;
            continue;
        }

        // compute the chosen feature
        std::vector<float> feat;
        int result = 0;
        if (featureType == "baseline")
        {
            result = computeBaseline(img, feat);
        }
        else if (featureType == "rgbhist")
        {
            result = computeRGBHist(img, feat, 8); // 8 bins per channel = 512 features
        }
        else if (featureType == "topbottom")
        {
            result = computeTopBottomHist(img, feat, 8);
        }
        else if (featureType == "colortexture")
        {
            result = computeColorTexture(img, feat, 8, 16);
        }
        else if (featureType == "custom")
        {
            result = computeCustomDumpster(img, feat, 8, 16);
        }

        if (result != 0)
        {
            std::cout << "Feature extraction failed for " << filename << "\n";
            failed++;
            continue;
        }

        // append_image_data_csv takes char* (not std::string), so cast
        char *csvName = const_cast<char *>(outputCsv.c_str());
        char *imgName = const_cast<char *>(filename.c_str());
        append_image_data_csv(csvName, imgName, feat, resetFile);
        resetFile = 0; // only reset on first write

        processed++;
        if (processed % 100 == 0)
        {
            std::cout << "Processed " << processed << " images\n";
        }
    }

    std::cout << "\nDone. Processed " << processed
              << " images, " << failed << " failed.\n"
              << "Output: " << outputCsv << "\n";
    return 0;
}