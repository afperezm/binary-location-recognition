/*
 * VocabMatch.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: andresf
 */

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <boost/regex.hpp>

#include <opencv2/core/core.hpp>

#include <VocabTree.h>

#include <FileUtils.hpp>

double mytime;

int main(int argc, char **argv) {

	if (argc < 6 || argc > 7) {
		printf("\nUsage:\n"
				"\t%s <in.tree> <in.db.list> <in.query.list>"
				" <num_nbrs> <matches.out> [results.html]\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	char *tree_in = argv[1];
	char *db_list_in = argv[2];
	char *query_list_in = argv[3];
	int num_nbrs = atoi(argv[4]);
	char *matches_out = argv[5];
	const char *output_html = "results.html";

	if (argc >= 7)
		output_html = argv[6];

	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(std::string(tree_in), expression) == false) {
		fprintf(stderr,
				"Input tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: load tree

	cvflann::VocabTree tree;

	printf("-- Reading tree from [%s]\n", tree_in);

	mytime = cv::getTickCount();
	tree.load(std::string(tree_in));
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Tree loaded in [%lf] ms, got [%lu] words \n", mytime,
			tree.size());

	// Step 2/4: read the database keyfiles
	printf("-- Loading DB keyfiles names and landmark id's\n");
	std::vector<std::string> db_filenames;
	std::vector<int> db_landmarks;
	std::ifstream keysList(db_list_in, std::fstream::in);

	if (keysList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n", db_list_in);
		return EXIT_FAILURE;
	}

	// Loading file names in list into a vector
	std::string line;
	while (getline(keysList, line)) {

		if (boost::regex_match(line, boost::regex("^(.+)\\s(.+)$")) == false) {
			fprintf(stderr,
					"Error while parsing DB list file [%s], line [%s] should be: <key.file> <landmark.id>\n",
					db_list_in, line.c_str());
			return EXIT_FAILURE;
		}

		char filename[256];
		int landmark;
		sscanf(line.c_str(), "%s %d", filename, &landmark);

		struct stat buffer;

		// Checking if file exist, if not print error and exit
		if (stat(filename, &buffer) != 0) {
			fprintf(stderr, "Keypoints file [%s] doesn't exist\n", filename);
			return EXIT_FAILURE;
		}

		// Checking that line refers to a compressed yaml or xml file
		if (boost::regex_match(std::string(filename), expression) == false) {
			fprintf(stderr,
					"Keypoints file [%s] must have the extension .yaml.gz or .xml.gz\n",
					filename);
			return EXIT_FAILURE;
		}

		db_filenames.push_back(std::string(filename));
		db_landmarks.push_back(landmark);
	}
	// Close file
	keysList.close();

	// Step 3/4: read the query keyfiles
	printf("-- Loading query keyfiles names\n");
	std::vector<std::string> query_filenames;
	keysList.open(query_list_in, std::fstream::in);

	if (keysList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n", db_list_in);
		return EXIT_FAILURE;
	}

	// Loading file names in list into a vector
	while (getline(keysList, line)) {

		struct stat buffer;

		// Checking if file exist, if not print error and exit
		if (stat(line.c_str(), &buffer) != 0) {
			fprintf(stderr, "Keypoints file [%s] doesn't exist\n",
					line.c_str());
			return EXIT_FAILURE;
		}

		// Checking that line refers to a compressed yaml or xml file
		if (boost::regex_match(line, expression) == false) {
			fprintf(stderr,
					"Keypoints file [%s] must have the extension .yaml.gz or .xml.gz\n",
					line.c_str());
			return EXIT_FAILURE;
		}

		query_filenames.push_back(line);
	}
	// Close file
	keysList.close();

	// Step 4/4: score each query keyfile

	int normType = cv::NORM_L1;

	printf("-- Scoring [%lu] query images against [%lu] DB images using [%s]\n",
			query_filenames.size(), db_filenames.size(),
			normType == cv::NORM_L1 ? "L1-norm" :
			normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");

	std::vector<cv::KeyPoint> imgKeypoints;
	cv::Mat imgDescriptors;
	cv::Mat scores;

	for (size_t i = 0; i < query_filenames.size(); i++) {
		// Initialize keypoints and descriptors
		imgKeypoints.clear();
		imgDescriptors = cv::Mat();
		// Load query keypoints and descriptors
		FileUtils::loadFeatures(query_filenames[i], imgKeypoints,
				imgDescriptors);

		mytime = cv::getTickCount();
		try {
			tree.scoreQuery(imgDescriptors, scores, db_filenames.size(),
					cv::NORM_L1);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}

		mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
				* 1000;

		for (size_t j = 0; (int) j < scores.cols; j++) {
			printf(
					"   Match score between [%lu] query image and [%lu] DB image: %f\n",
					i, j, scores.at<float>(0, j));
		}
	}

	return EXIT_SUCCESS;
}

