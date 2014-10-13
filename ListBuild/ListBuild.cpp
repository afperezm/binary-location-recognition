/*
 * ListBuild.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: andresf
 */

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <FunctionUtils.hpp>

#define KEYPOINT_FILE_EXTENSION ".bin"

void createListDbTxt(const char* folderName,
		const std::vector<std::string>& geometryFiles,
		std::string& keypointsFilename,
		const std::vector<std::string>& queryKeypointFiles,
		bool appendLandmarkId = false);

std::vector<std::string> createListQueriesTxt(const char* folderName,
		const std::vector<std::string>& geometryFiles,
		std::string& keypointsFilename, bool appendLandmarkId = false);

int readFolder(const char* folderPath, std::vector<std::string>& files);

int main(int argc, char **argv) {

	if (argc != 4) {
		printf(
				"\nUsage: %s [-lists|-gt] <in.ground.truth.folder> <out.lists.folder>\n\nOptions:\n"
						"\t-lists:\tcreate list of SIFT feature files of query and database images.\n"
						"\t-gt:\tcreate list of SIFT feature files of query and database images with corresponding landmark ID.\n\n",
				argv[0]);
		return EXIT_FAILURE;
	}

	std::vector<std::string> folderFiles;
	int result = readFolder(argv[2], folderFiles);
	if (result == EXIT_FAILURE) {
		return result;
	}

	if (std::string(argv[1]).compare("-lists") == 0) {

		std::string keypointsFilename = std::string(argv[3])
				+ "/list_queries.txt";
		std::vector<std::string> queryKeypointFiles = createListQueriesTxt(
				argv[2], folderFiles, keypointsFilename);

		keypointsFilename = std::string(argv[3]) + "/list_db.txt";
		createListDbTxt(argv[2], folderFiles, keypointsFilename,
				queryKeypointFiles);

	} else if (std::string(argv[1]).compare("-gt") == 0) {

		std::string keypointsFilename = std::string(argv[3]) + "/list_gt.txt";
		std::vector<std::string> queryKeypointFiles = createListQueriesTxt(
				argv[2], folderFiles, keypointsFilename, true);

		keypointsFilename = std::string(argv[3]) + "/list_db_ld.txt";
		createListDbTxt(argv[2], folderFiles, keypointsFilename,
				queryKeypointFiles, true);

	}

	return EXIT_SUCCESS;
}

int readFolder(const char* folderPath, std::vector<std::string>& files) {

	DIR *dir;
	struct dirent *ent;
	// Try opening folder
	if ((dir = opendir(folderPath)) != NULL) {
		fprintf(stdout, "Opening directory [%s]\n", folderPath);
		// Save all true directory names into a vector of strings
		while ((ent = readdir(dir)) != NULL) {
			// Ignore . and .. as valid folder names
			std::string name = std::string(ent->d_name);
			if (name.compare(".") != 0 && name.compare("..") != 0) {
				files.push_back(std::string(ent->d_name));
			}
		}
		closedir(dir);
		// Sort alphabetically vector of folder names
		sort(files.begin(), files.end());
		fprintf(stdout, "  Found [%d] files\n", (int) files.size());
	} else {
		fprintf(stderr, "  Could not open directory [%s]", folderPath);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void createListDbTxt(const char* folderName,
		const std::vector<std::string>& geometryFiles,
		std::string& keypointsFilename,
		const std::vector<std::string>& queryKeypointFiles,
		bool appendLandmarkId) {

	std::vector<std::string> dbKeypointFiles;

	std::ofstream keypointsFile;
	keypointsFile.open(keypointsFilename.c_str(),
			std::ios::out | std::ios::trunc);

	std::vector<std::string> landmarks;

	for (std::vector<std::string>::const_iterator fileName =
			geometryFiles.begin(); fileName != geometryFiles.end();
			++fileName) {
		if ((*fileName).find("1") != std::string::npos
				&& (*fileName).find("query") == std::string::npos) {
			// Open file
			fprintf(stdout, "Reading file [%s]\n", (*fileName).c_str());

			std::string landmarkName = FunctionUtils::parseLandmarkName(
					fileName);

			if (std::find(landmarks.begin(), landmarks.end(), landmarkName)
					== landmarks.end()) {
				// If landmarkName wasn't found then add it
				landmarks.push_back(landmarkName);
			}

			std::ifstream infile(
					(std::string(folderName) + "/" + *fileName).c_str());

			// Extract data from file
			std::string line;

			while (std::getline(infile, line)) {
				if (std::find(queryKeypointFiles.begin(),
						queryKeypointFiles.end(), line)
						== queryKeypointFiles.end()) {

					std::string imageName = "db/" + std::string(line.c_str())
							+ std::string(KEYPOINT_FILE_EXTENSION);

					if (appendLandmarkId == true) {
						// Position of the landmarkName in the vector of landmarks
						std::ostringstream temp;
						temp << ((int) landmarks.size()) - 1;
						imageName += " " + temp.str();
					}

					if (std::find(dbKeypointFiles.begin(),
							dbKeypointFiles.end(), line)
							== dbKeypointFiles.end()
							|| appendLandmarkId == true) {
						dbKeypointFiles.push_back(line);
						keypointsFile << imageName << std::endl;
					}
				}
			}

			//Close file
			infile.close();
		}
	}

	keypointsFile.close();
}

std::vector<std::string> createListQueriesTxt(const char* folderName,
		const std::vector<std::string>& geometryFiles,
		std::string& keypointsFilename, bool appendLandmarkId) {

	std::vector<std::string> queryKeypointFiles;

	std::ofstream keypointsFile;
	keypointsFile.open(keypointsFilename.c_str(),
			std::ios::out | std::ios::trunc);

	std::vector<std::string> landmarks;

	for (std::vector<std::string>::const_iterator fileName =
			geometryFiles.begin(); fileName != geometryFiles.end();
			++fileName) {
		if ((*fileName).find("query") != std::string::npos) {
			// Open file
			fprintf(stdout, "Reading file [%s]\n", (*fileName).c_str());

			std::string landmarkName = FunctionUtils::parseLandmarkName(
					fileName);
			if (std::find(landmarks.begin(), landmarks.end(), landmarkName)
					== landmarks.end()) {
				// If landmarkName wasn't found then add it
				landmarks.push_back(landmarkName);
			}

			std::ifstream infile(
					(std::string(folderName) + "/" + *fileName).c_str());

			// Extract data from file
			std::string line;

			while (std::getline(infile, line)) {
				std::vector<std::string> lineSplitted = FunctionUtils::split(
						line, ' ');
				std::string qName = "queries/" + lineSplitted[0].substr(5)
						+ std::string(KEYPOINT_FILE_EXTENSION);

				if (appendLandmarkId == true) {
					// Position of the landmarkName in the vector of landmarks
					std::ostringstream temp;
					temp << ((int) landmarks.size()) - 1;
					qName += " " + temp.str();
				}

				keypointsFile << qName << std::endl;

				queryKeypointFiles.push_back(lineSplitted[0].substr(5));
			}

			//Close file
			infile.close();
		}
	}

	return queryKeypointFiles;
}
