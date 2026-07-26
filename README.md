CS 5330 - Project 2: Content-Based Image Retrieval
NAME: Charul

DEVELOPMENT SETUP

Operating System:   Windows
IDE:                Visual Studio Code
Compiler:           MSVC (Visual Studio Build Tools 2026, amd64)
Build System:       CMake
OpenCV:             4.x (prebuilt for Windows, vc16)
C++ Standard:       C++17

EXECUTABLES BUILT

After building, bin/Debug/ contains two executables:

    computeFeatures.exe  - scans a directory of images, computes the
                           requested feature for each, and writes the
                           results to a CSV file.
    matchImage.exe       - given a target image and a feature CSV, ranks
                           database images by distance and prints top N.

BUILD INSTRUCTIONS

1. Open the project2 folder in VS Code.
2. Open the Command Palette (Ctrl+Shift+P).
3. Run "CMake: Configure" and select the
   "Visual Studio Build Tools 2026 - amd64" kit.
4. Run "CMake: Build" (or press F7).
5. Copy OpenCV debug DLLs into bin/Debug if not already there:
       copy C:\Users\Charul\Downloads\opencv\build\x64\vc16\bin\opencv_world*d.dll bin\Debug\

RUNNING THE EXECUTABLES

Step 1: Compute features for the database.

    .\bin\Debug\computeFeatures.exe <image_dir> <feature_type> <output_csv>

  feature_type options:
      baseline       - 7x7 center patch (Task 1)
      rgbhist        - 8-bin RGB histogram (Task 2)
      topbottom      - top-half + bottom-half RGB histograms (Task 3)
      colortexture   - color hist + Sobel gradient texture hist (Task 4)
      custom         - green-ratio + colortexture (Task 7)

  Example:
    .\bin\Debug\computeFeatures.exe data\olympus rgbhist data\features_rgbhist.csv


Step 2: Run a query.

    .\bin\Debug\matchImage.exe <target_image> <feature_type> <distance_type> <csv_file> <N> [extra_csv]

  feature_type options:   same as above, plus "resnet" (Task 5)
  distance_type options:
      ssd                    - sum of squared differences (Task 1)
      hist_intersect         - histogram intersection (Tasks 2, 4)
      hist_intersect_multi   - average intersection over two halves (Task 3)
      colortexture_dist      - average of color and texture intersections (Task 4)
      cosine                 - cosine distance (Task 5)
      custom_dist            - weighted combo of all three (Task 7)

  The 6th argument <extra_csv> is required ONLY for the "custom" feature
  type - pass the ResNet18 embeddings CSV file path.

  Examples:

    Task 1:  .\bin\Debug\matchImage.exe data\olympus\pic.1016.jpg baseline ssd data\features_baseline.csv 4

    Task 2:  .\bin\Debug\matchImage.exe data\olympus\pic.0164.jpg rgbhist hist_intersect data\features_rgbhist.csv 4

    Task 3:  .\bin\Debug\matchImage.exe data\olympus\pic.0274.jpg topbottom hist_intersect_multi data\features_topbottom.csv 4

    Task 4:  .\bin\Debug\matchImage.exe data\olympus\pic.0535.jpg colortexture colortexture_dist data\features_colortexture.csv 4

    Task 5:  .\bin\Debug\matchImage.exe data\olympus\pic.0893.jpg resnet cosine data\ResNet18_olym.csv 4

    Task 7:  .\bin\Debug\matchImage.exe data\olympus\pic.0746.jpg custom custom_dist data\features_custom.csv 5 data\ResNet18_olym.csv

NOTES

- All feature CSVs are precomputed once and reused across queries.
- The image database (data/olympus) and the ResNet18 embeddings CSV
  (data/ResNet18_olym.csv) are NOT included in the submission since
  they were provided with the course materials.
- For Task 1, the self-match (target itself, distance 0) is shown as
  the first result. The next 3 are the actual top matches, which
  match the prof's reference list exactly.
- Built and tested in Debug mode.
