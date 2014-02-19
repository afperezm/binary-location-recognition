/*
 * OxDataReader.cpp
 *
 *  Created on: Jan 5, 2014
 *      Author: andresf
 */

#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>

#include <FileUtils.hpp>

#define NUMBER_OF_DESCRIPTORS 16334970

#define TRAILING_BYTES 12

/**
 * Load number of features.
 *
 * @param keypointsFolder
 * @param readingOrder
 * @param numFeaturesMap
 */
int loadNumberOfFeatures(std::string& keypointsFolder,
		std::vector<std::string>& readingOrder,
		std::map<std::string, int>& numFeaturesMap);

/**
 * Read original descriptors from a binary file, then transform and save them
 * in a suitable format.
 *
 * @param descriptorsFile - Path to the descriptors file
 * @param readingOrder - List holding order in which descriptors are written
 * @param descriptorsFolder - Path to the folder where to save descriptors
 * @param keypointsFolder - Path to the folder of key-point files
 * @param numFeaturesMap
 */
int readOriginalDescriptors(std::string& descriptorsFile,
		std::vector<std::string>& readingOrder, std::string& descriptorsFolder,
		std::string& keypointsFolder,
		std::map<std::string, int>& numFeaturesMap);

int main(int argc, char **argv) {

	if (argc != 5) {
		printf("\nUsage:\n"
				"\tOxDataReader <in.bin_descriptors_file> <in.list_imgs_order> "
				"<in.key_points_folder> <out.features_folder>\n\n");

		return EXIT_FAILURE;
	}

	// Program arguments
	std::string in_descriptorsFile = argv[1];
	std::string in_imagesReadingOrder = argv[2];
	std::string in_keypointsFolder = argv[3];
	std::string out_descriptorsFolder = argv[4];

	// Load reading order list
	std::vector<std::string> readingOrderList;
	FileUtils::loadList(in_imagesReadingOrder, readingOrderList);

	printf("-- Loading number of features per image\n");

	std::map<std::string, int> numFeaturesMap;
	int totalNumberFeatures = loadNumberOfFeatures(in_keypointsFolder,
			readingOrderList, numFeaturesMap);

	printf("   Total number of features is [%d]\n", totalNumberFeatures);

	printf("-- Reading descriptors file\n");

	int readBytes = readOriginalDescriptors(in_descriptorsFile,
			readingOrderList, out_descriptorsFolder, in_keypointsFolder,
			numFeaturesMap);

	printf("   Read [%d] Bytes\n", readBytes);

	return EXIT_SUCCESS;
}

int loadNumberOfFeatures(std::string& keypointsFolder,
		std::vector<std::string>& readingOrder,
		std::map<std::string, int>& numFeaturesMap) {

	numFeaturesMap.clear();

	std::ifstream keypointsFile;

	// Declare variables used inside the cycle
	std::string line, annoyingPrefix = "oxc1_";
	int numFeatures = -1;
	int totFeat = 0;

	for (std::string& keypointsFilename : readingOrder) {
		std::string keypointsFilepath = keypointsFolder + "/"
				+ keypointsFilename + ".txt";

		// Open key-points file
		keypointsFile.open(keypointsFilepath.c_str(), std::fstream::in);

		CV_Assert(keypointsFile.good() == true);

		// Skip first line
		keypointsFile >> line;
		// Second line contains number of key-points
		keypointsFile >> numFeatures;

		keypointsFilename.replace(0, annoyingPrefix.length(), "").c_str();
		numFeaturesMap.insert(std::make_pair(keypointsFilename, numFeatures));

		totFeat += numFeatures;

		// Close key-points file
		keypointsFile.close();
	}

	CV_Assert(totFeat == NUMBER_OF_DESCRIPTORS);

	return totFeat;
}

int readOriginalDescriptors(std::string& descriptorsFile,
		std::vector<std::string>& readingOrder, std::string& descriptorsFolder,
		std::string& keypointsFolder,
		std::map<std::string, int>& numFeaturesMap) {

	/* Readings descriptors from binary file */

	std::ifstream inputFileStream;

	// Open file
	inputFileStream.open(descriptorsFile.c_str(),
			std::fstream::in | std::fstream::binary);

	// Check file
	if (inputFileStream.good() == false) {
		throw std::runtime_error(
				"Unable to open file [" + descriptorsFile + "] for reading");
	}

	// Computing file size
	inputFileStream.seekg(0, std::ifstream::end);
	std::streampos size = inputFileStream.tellg();

	// Reading descriptors file by chunks of 128 Bytes and ignoring last 12 Bytes
	inputFileStream.seekg(0, std::ifstream::beg);

	int cumSum = 0;
	int descIdx = 0;
	int readBytes = 0;
	int descriptorSize = 128;
	unsigned char* data = new unsigned char[descriptorSize];

	cv::Mat descriptors;

	std::map<std::string, int>::iterator it = numFeaturesMap.begin();

	descriptors = cv::Mat(it->second, descriptorSize, CV_32F);

	std::string descriptorFileName;

	while (inputFileStream.read((char*) data, descriptorSize)) {
		readBytes += descriptorSize;
		for (int k = 0; k < descriptorSize; ++k) {
			CV_Assert(float(data[k]) >= 0.0);
			CV_Assert(
					descIdx - cumSum >= 0
							&& descIdx - cumSum <= descriptors.rows - 1);
			descriptors.at<float>(descIdx - cumSum, k) = float(data[k]);
		}
		if (descIdx == cumSum + it->second - 1) {
			descriptorFileName = descriptorsFolder + "/" + it->first
					+ ".bin";
			printf("-- Saving feature descriptors to [%s]\n",
					descriptorFileName.c_str());
//			FileUtils::saveDescriptors(descriptorFileName, descriptors);
		}
		if (descIdx + 1 == cumSum + it->second) {
			cumSum += it->second;
			++it;
			if (it != numFeaturesMap.end()) {
				descriptors.release();
				descriptors = cv::Mat(it->second, descriptorSize, CV_32F);
			}
		}
		++descIdx;
	}

	CV_Assert(cumSum == NUMBER_OF_DESCRIPTORS);

	CV_Assert(readBytes + TRAILING_BYTES == int(size));

	delete[] data;

	// Close file
	inputFileStream.close();

	return readBytes;

}
