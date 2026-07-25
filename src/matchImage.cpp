/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 2: Content-Based Image Retrieval

  given a target image and a precomputed feature CSV, ranks all
  database images by distance to the target and prints the top N matches.

  usage:
    matchImage <target_image> <feature_type> <distance_type> <csv_file> <N>

  feature_type options:
    baseline
  distance_type options:
    ssd
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include "features.h"
#include "distances.h"
#include "csv_util.h"

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        std::cout << "Usage: " << argv[0]
                  << " <target_image> <feature_type> <distance_type> <csv_file> <N>\n";
        return -1;
    }

    std::string targetPath = argv[1];
    std::string featureType = argv[2];
    std::string distanceType = argv[3];
    std::string csvPath = argv[4];
    int N = std::atoi(argv[5]);

    // load the precomputed database features from CSV
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    char *csvName = const_cast<char *>(csvPath.c_str());
    if (read_image_data_csv(csvName, filenames, data, 0) != 0)
    {
        std::cout << "Could not read CSV: " << csvPath << "\n";
        return -1;
    }

    if (featureType == "custom")
    {
        // need a second CSV with ResNet features
        if (argc < 7)
        {
            std::cout << "Usage for custom: <target> custom custom_dist <custom_csv> <N> <resnet_csv>\n";
            return -1;
        }
        std::string resnetCsvPath = argv[6];

        // read the ResNet CSV
        std::vector<char *> resnetNames;
        std::vector<std::vector<float>> resnetData;
        char *resnetCsv = const_cast<char *>(resnetCsvPath.c_str());
        if (read_image_data_csv(resnetCsv, resnetNames, resnetData, 0) != 0)
        {
            std::cout << "Could not read ResNet CSV\n";
            return -1;
        }

        // build a lookup table from filename -> ResNet vector
        std::map<std::string, std::vector<float>> resnetMap;
        for (size_t i = 0; i < resnetNames.size(); i++)
        {
            resnetMap[std::string(resnetNames[i])] = resnetData[i];
        }

        // compute the target's custom feature
        cv::Mat target = cv::imread(targetPath);
        if (target.empty())
        {
            std::cout << "Could not read target image\n";
            return -1;
        }
        std::vector<float> targetCustom;
        if (computeCustomDumpster(target, targetCustom, 8, 16) != 0)
        {
            std::cout << "Could not compute custom feature for target\n";
            return -1;
        }

        // look up the target's ResNet vector
        namespace fs = std::filesystem;
        std::string targetName = fs::path(targetPath).filename().string();
        if (resnetMap.find(targetName) == resnetMap.end())
        {
            std::cout << "Target " << targetName << " not found in ResNet CSV\n";
            return -1;
        }
        std::vector<float> targetResnet = resnetMap[targetName];

        // score against every database image
        std::vector<std::pair<float, int>> distances;
        for (int i = 0; i < (int)data.size(); i++)
        {
            std::string dbName(filenames[i]);
            if (resnetMap.find(dbName) == resnetMap.end())
                continue; // skip if not in ResNet CSV

            float d = customDumpsterDistance(
                targetCustom, data[i],
                targetResnet, resnetMap[dbName],
                8 * 8 * 8 // colorSize
            );
            if (d < 0)
                continue;
            distances.push_back({d, i});
        }

        std::sort(distances.begin(), distances.end());

        std::cout << "\nTop " << N << " matches for " << targetPath << ":\n";
        int limit = std::min(N, (int)distances.size());
        for (int i = 0; i < limit; i++)
        {
            std::cout << "  " << (i + 1) << ". " << filenames[distances[i].second]
                      << "  (distance = " << distances[i].first << ")\n";
        }

        // cleeanup
        for (char *p : filenames)
            delete[] p;
        for (char *p : resnetNames)
            delete[] p;
        return 0;
    }

    std::vector<float> targetFeat;
    namespace fs = std::filesystem;
    std::string targetName = fs::path(targetPath).filename().string();

    if (featureType == "resnet")
    {
        bool found = false;
        for (size_t i = 0; i < filenames.size(); i++)
        {
            if (targetName == filenames[i])
            {
                targetFeat = data[i];
                found = true;
                break;
            }
        }
        if (!found)
        {
            std::cout << "Target " << targetName << " not found in CSV\n";
            return -1;
        }
    }
    else
    {
        // load target image and compute its feature
        cv::Mat target = cv::imread(targetPath);
        if (target.empty())
        {
            std::cout << "Could not read target image: " << targetPath << "\n";
            return -1;
        }

        if (featureType == "baseline")
        {
            if (computeBaseline(target, targetFeat) != 0)
            {
                std::cout << "Could not compute baseline feature for target\n";
                return -1;
            }
        }
        else if (featureType == "rgbhist")
        {
            if (computeRGBHist(target, targetFeat, 8) != 0)
            {
                std::cout << "Could not compute RGB histogram for target\n";
                return -1;
            }
        }
        else if (featureType == "topbottom")
        {
            if (computeTopBottomHist(target, targetFeat, 8) != 0)
            {
                std::cout << "Could not compute top-bottom histogram for target\n";
                return -1;
            }
        }
        else if (featureType == "colortexture")
        {
            if (computeColorTexture(target, targetFeat, 8, 16) != 0)
            {
                std::cout << "Could not compute color+texture feature for target\n";
                return -1;
            }
        }
        else
        {
            std::cout << "Unknown feature type: " << featureType << "\n";
            return -1;
        }
    }

    // compute distance from target to every database image
    // use a vector of (distance, index) pairs so we can sort and recover filenames
    std::vector<std::pair<float, int>> distances;
    distances.reserve(data.size());

    for (int i = 0; i < (int)data.size(); i++)
    {
        float d = -1.0f;
        if (distanceType == "ssd")
        {
            d = ssd(targetFeat, data[i]);
        }
        else if (distanceType == "hist_intersect")
        {
            d = histIntersection(targetFeat, data[i]);
        }
        else if (distanceType == "hist_intersect_multi")
        {
            d = histIntersectionMulti(targetFeat, data[i]);
        }
        else if (distanceType == "colortexture_dist")
        {
            d = colorTextureDistance(targetFeat, data[i], 8 * 8 * 8);
        }
        else if (distanceType == "cosine")
        {
            d = cosineDistance(targetFeat, data[i]);
        }
        else
        {
            std::cout << "Unknown distance type: " << distanceType << "\n";
            return -1;
        }
        if (d < 0)
            continue; // skip failed comparisons
        distances.push_back({d, i});
    }

    // sort ascending - smallest distance = best match
    std::sort(distances.begin(), distances.end());

    // print top-N (note: index 0 will typically be the target itself
    // with distance 0 if the target is in the database)
    std::cout << "\nTop " << N << " matches for " << targetPath << ":\n";
    int limit = std::min(N, (int)distances.size());
    for (int i = 0; i < limit; i++)
    {
        std::cout << "  " << (i + 1) << ". " << filenames[distances[i].second]
                  << "  (distance = " << distances[i].first << ")\n";
    }

    // free the filename C-strings allocated by read_image_data_csv
    for (char *p : filenames)
        delete[] p;

    return 0;
}