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
#include <HtmlResultsWriter.hpp>

double mytime;

const static boost::regex DESCRIPTOR_REGEX("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

int main(int argc, char **argv) {

	if (argc < 6 || argc > 8) {
		printf(
				"\nUsage:\n"
						"\t%s <in.db> <in.db.list> <in.query.list> <out.ranked.files.folder>"
						" <in.num.neighbors> [in.type.binary:1] [out.results:results.html]\n\n",
				argv[0]);
		return EXIT_FAILURE;
	}

	char *db_filepath = argv[1];
	char *db_gt_list_filepath = argv[2];
	char *query_list_filepath = argv[3];
	char *ranked_files_folderpath = argv[4];
	uint num_nbrs = atoi(argv[5]);
	bool is_binary = true;
	const char *output_html = const_cast<char*>("results.html");

	if (argc >= 7) {
		is_binary = atoi(argv[6]);
	}

	if (argc >= 8) {
		output_html = argv[7];
	}

	// Checking that DB filename refers to a compressed yaml or xml file
	if (boost::regex_match(std::string(db_filepath), DESCRIPTOR_REGEX)
			== false) {
		fprintf(stderr,
				"Input tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: load tree
	cv::Ptr<cvflann::VocabTreeBase> tree;

	if (is_binary == true) {
		tree = new cvflann::VocabTree<uchar, cv::Hamming>();
	} else {
		tree = new cvflann::VocabTree<float, cvflann::L2<float> >();
	}

	printf("-- Reading tree from [%s]\n", db_filepath);

	mytime = cv::getTickCount();
	tree->load(std::string(db_filepath));
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Tree loaded in [%lf] ms, got [%lu] words \n", mytime,
			tree->size());

	// Step 2/4: read the database keyfiles
	printf("-- Loading file of DB images ground truth\n");
	std::vector<std::string> db_filenames;
	std::ifstream keys_list(db_gt_list_filepath, std::fstream::in);

	if (keys_list.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n",
				db_gt_list_filepath);
		return EXIT_FAILURE;
	}

	// Loading db keypoint filename and landmark id into a set
	std::string line;
	while (getline(keys_list, line)) {

		// Verifying line format
		if (boost::regex_match(line, boost::regex("^(.+)\\s(.+)$")) == false) {
			fprintf(stderr,
					"Line [%s] should be formatted as: <key.file> <landmark.id>\n",
					line.c_str());
			return EXIT_FAILURE;
		}

		// Extracting DB image keypoints filename
		char filename[256];

		sscanf(line.c_str(), "%s", filename);

		// Checking that filename refers to a compressed yaml or xml file
		if (boost::regex_match(std::string(filename), DESCRIPTOR_REGEX)
				== false) {
			fprintf(stderr,
					"Keypoints file [%s] must have the extension .yaml.gz or .xml.gz\n",
					filename);
			return EXIT_FAILURE;
		}

		db_filenames.push_back(std::string(filename));
	}
	// Close file
	keys_list.close();

	// Step 3/4: read the query keyfiles
	printf("-- Loading query keyfiles names\n");
	std::vector<std::string> query_filenames;
	keys_list.open(query_list_filepath, std::fstream::in);

	if (keys_list.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n",
				query_list_filepath);
		return EXIT_FAILURE;
	}

	// Loading file names in list into a vector
	while (getline(keys_list, line)) {

		// Checking that file exists, if not print error and exit
		struct stat buffer;
		if (stat(line.c_str(), &buffer) != 0) {
			fprintf(stderr, "Keypoints file [%s] doesn't exist\n",
					line.c_str());
			return EXIT_FAILURE;
		}

		// Checking that line refers to a compressed yaml or xml file
		if (boost::regex_match(line, DESCRIPTOR_REGEX) == false) {
			fprintf(stderr,
					"Keypoints file [%s] must have the extension .yaml.gz or .xml.gz\n",
					line.c_str());
			return EXIT_FAILURE;
		}

		query_filenames.push_back(line);
	}
	// Close file
	keys_list.close();

	// Step 4/4: score each query keyfile
	int normType = cv::NORM_L1;

	printf("-- Scoring [%lu] query images against [%lu] DB images using [%s]\n",
			query_filenames.size(), db_filenames.size(),
			normType == cv::NORM_L1 ? "L1-norm" :
			normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");

	cv::Mat imgDescriptors;
	cv::Mat scores;

	// Compute the number of candidates
	int top = MIN (num_nbrs, db_filenames.size());

	FILE *f_html = fopen(output_html, "w");
	HtmlResultsWriter::getInstance().writeHeader(f_html, top);
	if (f_html == NULL) {
		fprintf(stderr, "Error opening file [%s] for writing\n", output_html);
		return EXIT_FAILURE;
	}

	for (size_t i = 0; i < query_filenames.size(); i++) {
		// Initialize keypoints and descriptors
		imgDescriptors = cv::Mat();

		// Load query keypoints and descriptors
		FileUtils::loadDescriptors(query_filenames[i], imgDescriptors);

		// Check type of descriptors
		if ((imgDescriptors.type() == CV_8U) != is_binary) {
			fprintf(stderr,
					"Descriptor type doesn't coincide, it is said to be [%s] while it is [%s]\n",
					is_binary == true ? "binary" : "non-binary",
					imgDescriptors.type() == CV_8U ? "binary" : "real");
			return EXIT_FAILURE;
		}

		// Score query bow vector against DB images bow vectors
		mytime = cv::getTickCount();
		try {
			tree->scoreQuery(imgDescriptors, scores, cv::NORM_L1);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}
		mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
				* 1000;
		imgDescriptors.release();

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

		std::stringstream ranked_list_fname;
		printf("%lu) %s\n", i, query_filenames[i].c_str());
		ranked_list_fname << ranked_files_folderpath << "/query_" << i
				<< "_ranked.txt";

		FILE *f_ranked_list = fopen(ranked_list_fname.str().c_str(), "w");
		if (f_ranked_list == NULL) {
			fprintf(stderr, "Error opening file [%s] for writing\n",
					ranked_list_fname.str().c_str());
			return EXIT_FAILURE;
		}
		for (int j = 0; j < top; j++) {
			std::string d_base = db_filenames[perm.at<int>(0, j)];
			fprintf(f_ranked_list, "%s\n",
					d_base.substr(0, d_base.size() - 8).substr(3).c_str());
		}
		fclose(f_ranked_list);

		// Print to a file the ranked list of candidates ordered by score in HTML format
		HtmlResultsWriter::getInstance().writeRow(f_html, query_filenames[i],
				scores, perm, top, db_filenames);
		scores.release();
		perm.release();
	}

	HtmlResultsWriter::getInstance().writeFooter(f_html);
	fclose(f_html);

	return EXIT_SUCCESS;
}
