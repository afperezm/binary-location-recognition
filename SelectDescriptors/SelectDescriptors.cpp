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

#include <DynamicMat.hpp>
#include <FileUtils.hpp>
#include <FunctionUtils.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/random.h>

static const size_t DESC_CHUNK = 4000;

int main(int argc, char **argv) {

	if (argc != 4) {
		printf(
				"\nUsage:\n"
						"\tSelectDescriptors <in.descriptors.folder> <in.percentage> <out.sampled.descriptors.folder>\n\n");
		return EXIT_FAILURE;
	}

	std::string in_descs_folder = argv[1];
	double in_percentage = atof(argv[2]);
	std::string out_folder = argv[3];

	if (in_percentage <= 0.0 || in_percentage >= 100.0) {
		fprintf(stderr,
				"<in.percentage> must be a number between 0 and 100 excluding limits\n");
		return EXIT_FAILURE;
	} else {
		in_percentage = in_percentage / 100.0;
	}

	// Step 1: read descriptors list
	printf("-- Reading files in folder [%s]\n", in_descs_folder.c_str());
	std::vector<std::string> descriptorsFilenames;
	FileUtils::readFolder(in_descs_folder.c_str(), descriptorsFilenames);
	printf("   Done, got [%lu] entries\n", descriptorsFilenames.size());

	for (std::string& descriptors : descriptorsFilenames) {
		descriptors = in_descs_folder + "/" + descriptors;
	}

	// Step 2: read descriptors files
	printf("-- Reading descriptors files\n");

	vlr::Mat mergedDescriptors(descriptorsFilenames);

	// Step 3: randomly select a percentage of the descriptors

	printf("-- Selecting randomly [%f] of descriptors, hence [%d] of [%d]\n",
			in_percentage, int(mergedDescriptors.rows * in_percentage),
			mergedDescriptors.rows);

	cvflann::seed_random(unsigned(std::time(0)));
	cvflann::UniqueRandom randGen(mergedDescriptors.rows);

	std::vector<int> indices(int(mergedDescriptors.rows * in_percentage));
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = randGen.next();
	}

	// Sort the array of indices
	std::sort(indices.begin(), indices.end());

	// Step 4: iterate over the loaded descriptors and save to files in chunks
	printf("-- Accessing chosen descriptor and saving them into [%s]\n",
			out_folder.c_str());

	cv::Mat imgDescriptors = cv::Mat::zeros(DESC_CHUNK, mergedDescriptors.cols,
			mergedDescriptors.type());

	for (size_t i = 0; i < indices.size(); ++i) {

		// If it is a starting descriptor or the last one then save
		if (i > 0 && ((i % DESC_CHUNK) == 0 || i + 1 == indices.size())) {
			// Prepare filename
			char buffer[50];

			sprintf(buffer, "descriptors_%04d.bin",
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

			CV_Assert(imgDescriptors.rows == (int )DESC_CHUNK);

			// Save features
			FileUtils::saveDescriptors(out_folder + "/" + std::string(buffer),
					imgDescriptors);

			// Clean descriptors matrix and key-points vector
			imgDescriptors.release();
			imgDescriptors = cv::Mat::zeros(DESC_CHUNK, mergedDescriptors.cols,
					mergedDescriptors.type());
		}

		cv::Mat submat = imgDescriptors.rowRange(i % DESC_CHUNK,
				(i % DESC_CHUNK) + 1);
		mergedDescriptors.row(indices[i]).copyTo(submat);

	}

}
