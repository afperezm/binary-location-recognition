/*
 * FeatureSelect.cpp
 *
 *  Created on: Oct 8, 2013
 *      Author: andresf
 */

#include <dirent.h>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
#include <opencv2/core/internal.hpp>
#include <opencv2/extensions/features2d.hpp>
#include <opencv2/flann/logger.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
*/

struct QueryImage {

	std::string imgName;
	float topLeftX;
	float topLeftY;
	float bottomRightX;
	float bottomRightY;

	QueryImage(std::string& line) {
			std::vector<std::string> tokens;
			FunctionUtils::split(line, ' ', tokens);
			imgName = tokens[0] + ".jpg";
			topLeftX = atof(tokens[1].c_str());
			topLeftY = atof(tokens[2].c_str());
			bottomRightX = atof(tokens[3].c_str());
			bottomRightY = atof(tokens[4].c_str());
	}

};

void readClipSave(QueryImage& qimg, std::string& outFolder);

int main(int argc, char **argv) {

	if (argc != 3 ) {
		printf(
				"\nUsage:\n"
						"\t%s <in.list.imgs.with.coords> <out.clipped.imgs.folder>\n\n",
				argv[0]);
		return EXIT_FAILURE;
	}

	char* list_in = argv[1];
	std::string out_folder = std::string(argv[2]);

    // Step 1: read list of images with coordinates
	printf("-- Reading file [%s] with list of images with coordinates\n", list_in);
    std::vector<QueryImage> keysFilenames;
    std::ifstream keysList(list_in, std::fstream::in);

    if (keysList.is_open() == false) {
            fprintf(stderr, "Error opening file [%s] for reading\n", list_in);
            return EXIT_FAILURE;
    }

    // Loading file names in list into a vector
    std::string line;
    while (getline(keysList, line)) {
			QueryImage qimg(line);
            struct stat buffer;
            // Checking if file exist, if not print error and exit
            if (stat(qimg.imgName.c_str(), &buffer) == 0) {
                    keysFilenames.push_back(qimg);
            } else {
                    fprintf(stderr, "Keypoints file [%s] doesn't exist\n",
                                    qimg.imgName.c_str());
                    return EXIT_FAILURE;
            }
    }
    // Close file
    keysList.close();

	// Step 2: read, clip and save images in list
	printf("-- Read, clip and save images\n");

	for(QueryImage qimg : keysFilenames){
		printf("  %s\n", qimg.imgName.c_str());
		readClipSave(qimg, out_folder);
	}

	return EXIT_SUCCESS;
}

void readClipSave(QueryImage& qimg, std::string& outFolder) {

	std::vector<std::string> tokens;
	FunctionUtils::split(qimg.imgName, '/', tokens);

	std::string imgPath = tokens[0];
	std::string imgName = tokens[1];

	cv::Mat img = cv::imread(imgPath + std::string("/") + imgName,
			CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data) {
		throw std::runtime_error(
				"Error reading image [" + imgPath + std::string("/") + imgName
						+ "]");
	}

	cv::Rect zone(qimg.topLeftX, qimg.topLeftY, qimg.bottomRightX-qimg.topLeftX, qimg.bottomRightY-qimg.topLeftY);
	cv::Mat clippedImg;
	img(zone).copyTo(clippedImg);
	cv::imwrite(outFolder + "/" + imgName, clippedImg);

	img.release();
	clippedImg.release();

}

