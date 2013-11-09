/*
 * SelectDescriptors.cpp
 *
 *  Created on: Nov 7, 2013
 *      Author: andresf
 */

#include <ctime>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <vector>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/random.h>

static const size_t DESC_CHUNK = 4000;

int main(int argc, char **argv) {

	if (argc != 4) {
		printf("\nUsage:\n"
				"\t%s <in.list> <in.percentage> <out.folder>\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	const char *list_in = argv[1];
	double percentage = atof(argv[2]);

	const char *folder_out = argv[3];

	if (percentage <= 0.0 || percentage >= 100.0) {
		fprintf(stderr,
				"<in.percentage> must be a number between 0 and 100 excluding limits\n");
		return EXIT_FAILURE;
	} else {
		percentage = percentage / 100.0;
	}

	// Step 1: read list of key files
	printf("-- Loading list of keypoint files [%s]\n", list_in);
	std::vector<std::string> keysFilenames;
	std::ifstream keysList(list_in, std::fstream::in);

	if (keysList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n", list_in);
		return EXIT_FAILURE;
	}

	// Loading file names in list into a vector
	std::string line;
	while (getline(keysList, line)) {
		struct stat buffer;
		// Checking if file exist, if not print error and exit
		if (stat(line.c_str(), &buffer) == 0) {
			keysFilenames.push_back(line);
		} else {
			fprintf(stderr, "Keypoints file [%s] doesn't exist\n",
					line.c_str());
			return EXIT_FAILURE;
		}
	}
	// Close file
	keysList.close();

	// Step 2: read key files
	printf("-- Reading keypoint files\n");
	std::vector<image> descriptorsIndices;

	int descCount = 0, descLen = 0, descType = -1, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {

		printf("   %04d/%04lu %s\n", imgIdx + 1, keysFilenames.size(),
				keyFileName.c_str());

		// Declare variables for holding keypoints and descriptors
		std::vector<cv::KeyPoint> imgKeypoints;
		cv::Mat imgDescriptors = cv::Mat();

		// Load keypoints and descriptors
		FileUtils::loadFeatures(keyFileName, imgKeypoints, imgDescriptors);

		// Check that keypoints and descriptors have same length
		CV_Assert((int )imgKeypoints.size() == imgDescriptors.rows);

		if (imgDescriptors.empty() == false) {

			for (size_t i = 0; int(i) < imgDescriptors.rows; i++) {
				image img;
				img.imgIdx = imgIdx;
				img.startIdx = descCount;
				descriptorsIndices.push_back(img);
			}

			// Increase descriptors counter
			descCount += imgDescriptors.rows;

			// If initialized check descriptors length
			// Recall all the descriptors must be the same length
			if (descLen != 0) {
				CV_Assert(descLen == imgDescriptors.cols);
			} else {
				descLen = imgDescriptors.cols;
			}

			// If initialized check descriptors type
			// Recall all the descriptors must be the same type
			if (descType != -1) {
				CV_Assert(descType == imgDescriptors.type());
			} else {
				descType = imgDescriptors.type();
			}
		}
		imgDescriptors.release();
		// Increase images counter
		imgIdx++;
	}

	DynamicMat mergedDescriptors(descriptorsIndices, keysFilenames, descCount,
			descLen, descType);

	// Step 3: randomly select a percentage of the descriptors

	printf("-- Selecting [%f] of descriptors randomly, hence [%d] of [%d]\n",
			percentage, int(mergedDescriptors.rows * percentage),
			mergedDescriptors.rows);

	cvflann::seed_random(unsigned(std::time(0)));
	cvflann::UniqueRandom randGen(mergedDescriptors.rows);

	std::vector<int> indices(int(mergedDescriptors.rows * percentage));
	for (size_t i = 0; i < indices.size(); i++) {
		indices[i] = randGen.next();
	}

	// Sort the array of indices
	std::sort(indices.begin(), indices.end());

	// Step 4: iterate over the loaded descriptors and save to files in chunks
	printf("-- Accessing chosen descriptor and saving them into [%s]\n",
			folder_out);

	// Declare variables for holding keypoints and descriptors
	std::vector<cv::KeyPoint> imgKeypoints;
	cv::Mat imgDescriptors = cv::Mat::zeros(DESC_CHUNK, descLen, descType);

	for (size_t i = 0; i < indices.size(); i++) {

		// If it is a starting descriptor or the last one then save
		if (i > 0 && ((i % DESC_CHUNK) == 0 || i + 1 == indices.size())) {
			// Prepare filename
			char buffer[50];

			sprintf(buffer, "descriptors_%04d.yaml.gz",
					int(
							floor(i / DESC_CHUNK)
									+ ((i % DESC_CHUNK == 0) ? 0 : 1)));

			printf("   %02d/%02d %s\n",
					int(
							floor(i / DESC_CHUNK)
									+ ((i % DESC_CHUNK == 0) ? 0 : 1)),
					int(
							floor((indices.size() - 1) / DESC_CHUNK)
									+ (((indices.size() - 1) % DESC_CHUNK == 0) ?
											0 : 1)), buffer);

			// Prepare dummy list of keypoints
			imgKeypoints.resize(imgDescriptors.rows, cv::KeyPoint());
			CV_Assert(imgKeypoints.size() == DESC_CHUNK);
			CV_Assert(imgDescriptors.rows == (int )DESC_CHUNK);

			// Save features
			FileUtils::saveFeatures(
					std::string(folder_out) + "/" + std::string(buffer),
					imgKeypoints, imgDescriptors);

			// Clean descriptors matrix and keypoints vector
			imgDescriptors.release();
			imgDescriptors = cv::Mat::zeros(DESC_CHUNK, descLen, descType);
			imgKeypoints.clear();
		}

		cv::Mat submat = imgDescriptors.rowRange(i % DESC_CHUNK,
				(i % DESC_CHUNK) + 1);
		mergedDescriptors.row(indices[i]).copyTo(submat);

	}

}
