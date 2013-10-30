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

#include <FunctionUtils.hpp>

double mytime;

int main(int argc, char **argv) {

	if (argc < 5 || argc > 10) {
		printf(
				"\nUsage:\n"
						"\t%s <in.tree> <in.db.list> <in.db.gt.list> <in.query.list>"
						" <in.num.neighbors> [in.type.binary:1] [out.matches:matches.txt]"
						" [out.results:results.html] [out.candidates:candidates.txt]\n\n",
				argv[0]);
		return EXIT_FAILURE;
	}

	char *tree_in = argv[1];
	char *db_list_in = argv[2];
	char *db_gt_list_in = argv[3];
	char *query_list_in = argv[4];
	uint num_nbrs = atoi(argv[5]);
	bool isDescriptorBinary = true;
	char *matches_out = const_cast<char*>("matches.txt");
	const char *output_html = const_cast<char*>("results.html");
	const char *candidates_out = const_cast<char*>("candidates.txt");

	if (argc >= 7) {
		isDescriptorBinary = atoi(argv[6]);
	}

	if (argc >= 8) {
		matches_out = argv[7];
	}

	if (argc >= 9) {
		output_html = argv[8];
	}

	if (argc >= 10) {
		candidates_out = argv[9];
	}

	// Verifying input parameters
	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(std::string(tree_in), expression) == false) {
		fprintf(stderr,
				"Input tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: load tree
	cv::Ptr<cvflann::VocabTreeBase> tree;

	if (isDescriptorBinary == true) {
		tree = new cvflann::VocabTree<uchar, cv::Hamming>();
	} else {
		tree = new cvflann::VocabTree<float, cvflann::L2<float> >();
	}

	printf("-- Reading tree from [%s]\n", tree_in);

	mytime = cv::getTickCount();
	tree->load(std::string(tree_in));
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Tree loaded in [%lf] ms, got [%lu] words \n", mytime,
			tree->size());

	// Step 2a/4: read the file with the list of DB ground truth information
	printf("-- Loading file of DB images ground truth\n");
	std::map<std::string, std::vector<int> > db_gt;
	std::ifstream keysList(db_gt_list_in, std::fstream::in);
	int max_ld = -1;

	if (keysList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n", db_gt_list_in);
		return EXIT_FAILURE;
	}

	// Loading db keypoint filename and landmark id into a set
	std::string line;
	while (getline(keysList, line)) {

		// Verifying line format
		if (boost::regex_match(line, boost::regex("^(.+)\\s(.+)$")) == false) {
			fprintf(stderr,
					"Line [%s] should be formatted as: <key.file> <landmark.id>\n",
					 line.c_str());
			return EXIT_FAILURE;
		}

		// Extracting filename and landmark from line
		char filename[256];
		int landmarkId;
		sscanf(line.c_str(), "%s %d", filename, &landmarkId);

		// Checking that filename refers to a compressed yaml or xml file
		if (boost::regex_match(std::string(filename), expression) == false) {
			fprintf(stderr,
					"Keypoints file [%s] must have the extension .yaml.gz or .xml.gz\n",
					filename);
			return EXIT_FAILURE;
		}

		// Check that landmark id is valid
		if (landmarkId < 0) {
			fprintf(stderr,
					"Landmark id [%d] extracted from line [%s] should be greater than or equal to zero\n",
					landmarkId, line.c_str());
			return EXIT_FAILURE;
		}

		// Compute maximum landmark id
		if (max_ld < landmarkId) {
			max_ld = landmarkId;
		}

		// Check if key exists and insert extracted data into map
		std::map<std::string, std::vector<int> >::iterator it = db_gt.find(std::string(filename));
		if (it == db_gt.end()) {
			// key doesn't exist
			std::vector<int> landmarks;
			landmarks.push_back(landmarkId);
			db_gt.insert(
					std::pair<std::string, std::vector<int> >(
							std::string(filename), landmarks));
		} else {
			// key exists hence the vector of landmarks shouldn't be empty
			CV_Assert(it->second.empty() == false);
			it->second.push_back(landmarkId);
		}
	}
	// Close file
	keysList.close();

	// Step 2b/4: read the file with the list of DB keypoint filenames
	printf("-- Loading file of DB images keypoints filenames\n");
	std::vector<std::string> db_keys_filenames;
	keysList.open(db_list_in, std::fstream::in);

	if (keysList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n", db_list_in);
		return EXIT_FAILURE;
	}

	// Loading db keypoints filenames into a vector
	while (getline(keysList, line)) {

		// Checking file extension to be compressed yaml or xml
		if (boost::regex_match(line, expression) == false) {
			fprintf(stderr,
					"Keypoints file [%s] must have the extension .yaml.gz or .xml.gz\n",
					line.c_str());
			return EXIT_FAILURE;
		}

		db_keys_filenames.push_back(line);
	}
	// Close file
	keysList.close();

	// Step 3/4: read the query keyfiles
	printf("-- Loading query keyfiles names\n");
	std::vector<std::string> query_filenames;
	keysList.open(query_list_in, std::fstream::in);

	if (keysList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n", db_gt_list_in);
		return EXIT_FAILURE;
	}

	// Loading file names in list into a vector
	while (getline(keysList, line)) {

		// Checking that file exists, if not print error and exit
		struct stat buffer;
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
			query_filenames.size(), db_keys_filenames.size(),
			normType == cv::NORM_L1 ? "L1-norm" :
			normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");

	std::vector<cv::KeyPoint> imgKeypoints;
	cv::Mat imgDescriptors;
	cv::Mat scores;

	// Compute the number of candidates
	int top = MIN (num_nbrs, db_keys_filenames.size());

	FILE *f_match = fopen(matches_out, "w");
	if (f_match == NULL) {
		fprintf(stderr, "Error opening file [%s] for writing\n",
				candidates_out);
		return EXIT_FAILURE;
	}

	FILE *f_candidates = fopen(candidates_out, "w");
	if (f_candidates == NULL) {
		fprintf(stderr, "Error opening file [%s] for writing\n",
				candidates_out);
		return EXIT_FAILURE;
	}

	FILE *f_html = fopen(output_html, "w");
	HtmlResultsWriter::getInstance().writeHeader(f_html, top);
	if (f_html == NULL) {
		fprintf(stderr, "Error opening file [%s] for writing\n",
				candidates_out);
		return EXIT_FAILURE;
	}

	for (size_t i = 0; i < query_filenames.size(); i++) {
		// Initialize keypoints and descriptors
		imgKeypoints.clear();
		imgDescriptors = cv::Mat();

		// Load query keypoints and descriptors
		FileUtils::loadFeatures(query_filenames[i], imgKeypoints,
				imgDescriptors);

		// Check type of descriptors
		if ((imgDescriptors.type() == CV_8U) != isDescriptorBinary) {
			fprintf(stderr,
					"Descriptor type doesn't coincide, it is said to be [%s] while it is [%s]\n",
					isDescriptorBinary == true ? "binary" : "non-binary",
					imgDescriptors.type() == CV_8U ? "binary" : "real");
			return EXIT_FAILURE;
		}

		// Score query bow vector against DB images bow vectors
		mytime = cv::getTickCount();
		try {
			tree->scoreQuery(imgDescriptors, scores, db_keys_filenames.size(),
					cv::NORM_L1);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}
		mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
				* 1000;

		// Print to standard output the matching scores between
		// the query bow vector and the DB images bow vectors
		for (size_t j = 0; (int) j < scores.cols; j++) {
//			cv::Mat dbBowVector;
//			tree->getDbBowVector(j, dbBowVector);
//			std::cout << j << ") DB BoW vector:\n" << dbBowVector << std::endl;
//			printf(
//					"   Match score between [%s] query image and [%lu] DB image: %f\n",
//					query_filenames[i].c_str(), j, scores.at<float>(0, j));
//			getchar();
			printf(
					"   Match score between [%lu] query image and [%lu] DB image: %f\n",
					i, j, scores.at<float>(0, j));
		}

		// Obtain indices of ordered scores
		cv::Mat perm;
		// Note: recall that the index of the images in the inverted file corresponds
		// to the zero-based line number in the file used to build the DB.
		// Hence scores matrix and db_landmarks and db_filenames vectors.
		// are equally ordered.
		// Also the images in list_db and list_db_ld must be equally ordered,
		// that implies same number of elements.
		//
		// list_db      list_db_ld
		//   img1  --->  img1 ld1
		//   img2  --->  img2 ld1
		//   img3  --->  img3 ld1
		//   img4  --->  img4 ld2
		//   img5  --->  img5 ld2
		//   img6  --->  img6 ld2
		cv::sortIdx(scores, perm, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

		// Initialize votes vector
		// Note: size is maximum landmark id plus one because landmark index is zero-based
		std::vector<int> votes(max_ld + 1, 0);

		// Accumulating landmark votes for the top scored images
		// Note: recall that images might refer to the same landmark
		for (size_t j = 0; (int) j < top; j++) {
			std::map<std::string, std::vector<int> >::iterator it = db_gt.find(
					db_keys_filenames[perm.at<int>(0, j)]);
			if (it == db_gt.end()) {
				fprintf(stderr,
						"Error, file of DB images ground truth [%s] refers to different keypoint files"
								" than file of DB images keypoints filenames [%s]\n",
						std::string(db_gt_list_in).c_str(), std::string(db_list_in).c_str());
				return EXIT_FAILURE;
			}
			for (int landmark : it->second) {
				votes[landmark]++;
			}
		}

		// Finding max voted landmark and the number of votes it got
		int max_votes = 0;
		int max_landmark = -1;
		for (int j = 0; j < max_ld + 1; j++) {
			if (votes[j] > max_votes) {
				max_votes = votes[j];
				max_landmark = j;
			}
		}

		// Print to a file the ranked list of candidates ordered by score
		fprintf(f_candidates, "%s", query_filenames[i].c_str());
		for (int j = 0; j < top; j++) {
			std::string d_base = db_keys_filenames[perm.at<int>(0, j)];
			fprintf(f_candidates, " %s", d_base.c_str());
		}
		fprintf(f_candidates, "\n");
		fflush(f_candidates);

		std::stringstream ranked_list_fname;
		ranked_list_fname << "query_" << i << "_ranked.txt";

		FILE *f_ranked_list = fopen(ranked_list_fname.str().c_str(), "w");
		if (f_ranked_list == NULL) {
			fprintf(stderr, "Error opening file [%s] for writing\n",
					ranked_list_fname.str().c_str());
			return EXIT_FAILURE;
		}
		for (int j = 0; j < top; j++) {
			std::string d_base = db_keys_filenames[perm.at<int>(0, j)];
			fprintf(f_ranked_list, "%s\n", d_base.c_str());
		}
		fclose(f_ranked_list);

		// Print to a file the max voted landmark information
		fprintf(f_match, "%lu %d %d\n", i, max_landmark, max_votes);
		fflush(f_match);
		fflush(stdout);

		// Print to a file the ranked list of candidates ordered by score in HTML format
		HtmlResultsWriter::getInstance().writeRow(f_html, query_filenames[i],
				scores, perm, top, db_keys_filenames);
	}

	fclose(f_candidates);
	fclose(f_match);
	HtmlResultsWriter::getInstance().writeFooter(f_html);
	fclose(f_html);

	return EXIT_SUCCESS;
}
