/*
 * GeomVerify.cpp
 *
 *  Created on: Nov 20, 2013
 *      Author: andresf
 */

#include <fstream>

#include <boost/regex.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <VocabTree.h>

double mytime;

int loadList(const std::string& list_fpath, std::vector<std::string>& list);

void saveList(const std::string& list_fpath,
		const std::vector<std::string>& list);

void matchKeypoints(const cv::Ptr<bfeat::DirectIndex> directIndex1, int id1,
		const std::vector<cv::KeyPoint>& keypoints1,
		cv::vector<cv::Point2f>& matchedPoints1,
		const cv::Ptr<bfeat::DirectIndex> directIndex2, int id2,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::Point2f>& matchedPoints2,
		std::vector<cv::DMatch>& matches1to2);

// For each query
//	 - Load its keys
//	 - Load the list of its ranked candidates
//	 - For each ranked candidate
//		 - Load its keys
//		 - Find correspondences between query and candidate by intersecting direct index results
//		 - Match together all keypoints in same node and maybe apply some criteria for matching results
//		 - Compute a projective transformation between query and ranked file using direct index
//		 - Obtain number of inliers
//	 - Re-order list of candidates by its number of inliers

int main(int argc, char **argv) {

	if (argc < 11 || argc > 14) {
		printf(
				"\nUsage:\n"
						"\tGeomVerify <in.tree> <in.direct.index> "
						"<in.ranked.files.folder> <in.ranked.files.prefix> "
						"<in.db.desc.list> <in.db.keys.folder> <in.queries.desc.list> <in.queries.keys.list> "
						"<out.geom.ranked.files.folder> <in.top.results> "
						"[in.type.binary:1] [in.ransac.thr:10]"
						"\n\n");
		return EXIT_FAILURE;
	}

	std::string in_tree = argv[1];
	std::string in_direct_index = argv[2];
	std::string in_ranked_lists_folder = argv[3];
	std::string in_prefix = argv[4];
	std::string in_db_desc_list = argv[5]; // Used to obtain the mapping dbimg.candidate -> dbimg.id
	std::string in_db_keys_folder = argv[6]; // Used to load query ranked candidates
	std::string in_queries_desc_list = argv[7]; // Quantized in tree to match keypoints represented by the same word
	std::string in_queries_keys_list = argv[8];
	std::string out_ranked_lists_folder = argv[9];

	int topResults = atoi(argv[10]);
	bool isBinary = argc >= 12 ? atoi(argv[11]) : true;
	double ransacThreshold = argc >= 13 ? atof(argv[12]) : 10.0;

	// Step 1/4: load tree + direct index
	cv::Ptr<bfeat::VocabTreeBase> tree;
	cv::Ptr<bfeat::DirectIndex> directIndexQueries = new bfeat::DirectIndex();
	cv::Ptr<bfeat::DirectIndex> directIndexCandidates =
			new bfeat::DirectIndex();

	if (isBinary == true) {
		tree = new bfeat::VocabTree<uchar, cv::Hamming>();
	} else {
		tree = new bfeat::VocabTree<float, cvflann::L2<float> >();
	}

	printf("-- Reading tree from [%s]\n", in_tree.c_str());

	mytime = cv::getTickCount();
	tree->load(in_tree);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Tree loaded in [%lf] ms, got [%lu] words \n", mytime,
			tree->size());

	printf("-- Reading direct index [%s]\n", in_direct_index.c_str());

	mytime = cv::getTickCount();
	directIndexCandidates->load(in_direct_index);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Direct index loaded in [%lf] ms, got [%lu] images\n", mytime,
			directIndexCandidates->size());

	// Step 2a: load list of queries descriptors
	printf("-- Load list of queries descriptors\n");
	std::vector<std::string> queries_desc_list;
	loadList(in_queries_desc_list, queries_desc_list);
	printf("   Done, got [%lu] entries\n", queries_desc_list.size());

	// Step 2b: load queries descriptors

	uint wordIdx; // unused
	double wordWeight; // unused
	int nodeAtL, imgIdx = 0;
	cv::Mat queryDescriptors;

	printf(
			"-- Loading queries descriptors and adding them to the direct index\n");

	// Loop over list of queries descriptors
	for (std::string descriptorsFilename : queries_desc_list) {
		// Load query descriptors
		FileUtils::loadDescriptors(descriptorsFilename, queryDescriptors);
		// Loop over query descriptors
		for (int i = 0; i < queryDescriptors.rows; ++i) {
			// Propagate descriptor down tree
			tree->quantize(queryDescriptors.row(i), wordIdx, wordWeight,
					nodeAtL);
			// Add it to a new image to the direct index
			directIndexQueries->addFeature(imgIdx, nodeAtL, i);
		}
		++imgIdx;
	}

	printf("   Done, got [%lu] words\n", directIndexCandidates->size());

	// Step 3a: load list of queries keypoints
	printf("-- Loading list of queries keypoints\n");
	std::vector<std::string> queries_keys_list;
	loadList(in_queries_keys_list, queries_keys_list);
	printf("   Done, got [%lu] entries\n", queries_keys_list.size());

	if (queries_desc_list.size() != queries_keys_list.size()) {
		fprintf(stderr,
				"Different number of entries in queries keypoints and descriptors lists\n");
		return EXIT_FAILURE;
	}

	// Step 3b: load names of database files
	printf("-- Loading names of database files\n");
	std::vector<std::string> db_desc_list;
	loadList(in_db_desc_list, db_desc_list);

	// Step 4/4: load and process queries keypoints
	printf("-- Loading and processing queries keypoints\n");
	std::vector<cv::KeyPoint> queryKeypoints;

	std::vector<std::string> ranked_candidates_list,
			geom_ranked_candidates_list;
	std::vector<cv::KeyPoint> candidateKeypoints;
	std::stringstream ranked_list_fname;

	std::vector<cv::DMatch> matchesCandidateToQuery;
	std::vector<cv::Point2f> matchedCandidatePoints;
	std::vector<cv::Point2f> matchedQueryPoints;
	int candidateImgId, queryImgId;

	cv::Mat inliers_idx;
	cv::Mat candidates_inliers;
	cv::Mat candidates_inliers_idx;
	cv::Mat H;

	// Loop over list of queries keypoints
	for (size_t i = 0; i < queries_keys_list.size(); ++i) {
		printf("-- Processing query [%lu]\n", i);
		queryImgId = i;

		// Step 4a: load query keypoints
		printf("   Loading keypoints\n");
		FileUtils::loadKeypoints(queries_keys_list[i], queryKeypoints);
		printf("   Got [%lu] keypoints\n", queryKeypoints.size());

		// Step 4b: load list of query ranked candidates
		printf("   Loading list of ranked candidates\n");
		// Load list of query ranked candidates
		// Note: recall that elements in the lists of queries keypoints and descriptors
		// follow the same order and hence using query keypoints filename position to build
		// its ranked candidates filename its legal
		ranked_list_fname.str("");
		ranked_list_fname << in_ranked_lists_folder << "/query_" << i
				<< "_ranked.txt";
		loadList(ranked_list_fname.str(), ranked_candidates_list);
		printf("   Got [%lu] candidates\n", ranked_candidates_list.size());

		if (int(ranked_candidates_list.size()) < topResults) {
			// Caused by one of two:
			// * Input argument topResults is too high.
			// * Ranked list is too short.s
			std::stringstream ss;
			ss << "Not enough candidates for spatial re-ranking, "
					"need at least [" << topResults << "] "
					"but there are only ["
					<< ranked_candidates_list.size() + "]";
			throw std::runtime_error(ss.str());
		}

		candidates_inliers = cv::Mat::zeros(1, topResults,
				cv::DataType<int>::type);

		// Step 4c: load query ranked candidates
		for (size_t j = 0; int(j) < topResults; ++j) {

			// Load keypoints of jth ranked candidate
			printf("   Loading keypoints of candidate [%lu]\n", j);
			FileUtils::loadKeypoints(
					in_db_keys_folder + "/" + ranked_candidates_list[j]
							+ "_kpt.yaml.gz", candidateKeypoints);
			printf("   Got [%lu] keypoints\n", candidateKeypoints.size());

			// Searching putative matches
			printf("   Matching keypoints of query [%lu] to candidate [%lu]\n",
					i, j);

			// Id of database image
			std::vector<std::string>::iterator it = std::find(
					db_desc_list.begin(), db_desc_list.end(),
					"db/" + ranked_candidates_list[j] + ".yaml.gz");

			if (it == db_desc_list.end()) {
				throw std::runtime_error(
						"Candidate [%s] not found in list of database filenames");
			}

			candidateImgId = std::distance(db_desc_list.begin(), it);
			// printf("candidate name=[%s] id=[%d]\n", (*it).c_str(), candidateImgId);

			mytime = cv::getTickCount();

			matchKeypoints(directIndexQueries, queryImgId, queryKeypoints,
					matchedQueryPoints, directIndexCandidates, candidateImgId,
					candidateKeypoints, matchedCandidatePoints,
					matchesCandidateToQuery);

			mytime = (double(cv::getTickCount()) - mytime)
					/ cv::getTickFrequency() * 1000;

			printf("   Found [%d] putative matches in [%lf] ms\n",
					int(matchesCandidateToQuery.size()), mytime);

			if ((int(matchesCandidateToQuery.size())) < 4) {
				fprintf(stderr, "Error while matching keypoints, need at least "
						"4 putative matches for homography computation\n");
				return EXIT_FAILURE;
			}

			// Compute a projective transformation between query and ranked file using direct index
			printf("   Computing projective transformation "
					"between query [%lu] and candidate [%lu]\n", i, j);

			inliers_idx = cv::Mat();

			mytime = cv::getTickCount();
			H = cv::findHomography(matchedCandidatePoints, matchedQueryPoints,
					CV_RANSAC, ransacThreshold, inliers_idx);
			mytime = (double(cv::getTickCount()) - mytime)
					/ cv::getTickFrequency() * 1000;

			// Obtain number of inliers
			candidates_inliers.at<int>(0, j) = int(sum(inliers_idx)[0]);

			printf("   Computed homography in [%0.3fs], found [%d] inliers\n",
					mytime, candidates_inliers.at<int>(0, j));

			/**** Drawing inlier matches ****/

			std::vector<cv::DMatch> inliers;
			for (int i = 0; i < inliers_idx.rows; ++i) {
				if ((int) inliers_idx.at<uchar>(i) == 1) {
					inliers.push_back(matchesCandidateToQuery.at(i));
				}
			}

			cv::Mat imgMatches;
			cv::Mat queryImg, candidateImg;

			std::string queryBase = queries_keys_list[i].substr(8,
					queries_keys_list[i].length() - 20);
			std::string candidateBase = ranked_candidates_list[j];

			queryImg = cv::imread("oxbuild_images/" + queryBase + ".jpg",
					CV_LOAD_IMAGE_GRAYSCALE);
			candidateImg = cv::imread(
					"oxbuild_images/" + candidateBase + ".jpg",
					CV_LOAD_IMAGE_GRAYSCALE);

			cv::drawMatches(queryImg, queryKeypoints, candidateImg,
					candidateKeypoints, inliers, imgMatches,
					cv::Scalar::all(-1), cv::Scalar::all(-1),
					std::vector<char>(),
					cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			cv::imwrite(
					"oxbuild_images/match_" + queryBase + "_"
							+ ranked_candidates_list[j] + ".jpg", imgMatches);

			/********************************/

		}

		// Re-order list of candidates by its inlier number
		sortIdx(candidates_inliers, candidates_inliers_idx, CV_SORT_DESCENDING);

		// Copying re-ranked candidates
		geom_ranked_candidates_list.clear();
		for (size_t j = 0; int(j) < topResults; ++j) {
			geom_ranked_candidates_list.push_back(
					ranked_candidates_list[candidates_inliers_idx.at<int>(0, j)]);
		}

#if GVVERBOSE
		printf("Original ranked candidates list:\n");
		for (std::string candidate : ranked_candidates_list) {
			printf("%s, ", candidate.c_str());
		}
		printf("\n");
#endif

#if GVVERBOSE
		printf("Re-ranked candidates list:\n");
		for (std::string candidate : geom_ranked_candidates_list) {
			printf("%s, ", candidate.c_str());
		}
		printf("\n");
#endif

		// Copying non re-ranked candidates
		geom_ranked_candidates_list.insert(geom_ranked_candidates_list.end(),
				ranked_candidates_list.begin() + topResults,
				ranked_candidates_list.end());

#if GVVERBOSE
		printf("Full re-ranked candidates list:\n");
		for (std::string candidate : geom_ranked_candidates_list) {
			printf("%s, ", candidate.c_str());
		}
		printf("\n");
#endif

		ranked_list_fname.str("");
		ranked_list_fname << out_ranked_lists_folder << "/query_" << i
				<< "_ranked.txt";
		saveList(ranked_list_fname.str(), geom_ranked_candidates_list);
	}

}

int loadList(const std::string& list_fpath, std::vector<std::string>& list) {

	list.clear();

	if (FileUtils::checkFileExist(list_fpath) == false) {
		fprintf(stderr, "File [%s] doesn't exist\n", list_fpath.c_str());
		return EXIT_FAILURE;
	}

	std::ifstream inputFileStream;
	std::string line;

	// Open file
	inputFileStream.open(list_fpath.c_str(), std::fstream::in);

	// Check file
	if (inputFileStream.good() == false) {
		fprintf(stderr, "Error while opening file [%s] for reading\n",
				list_fpath.c_str());
		return EXIT_FAILURE;
	}

	// Load list from file
	while (getline(inputFileStream, line)) {
		list.push_back(line);
	}

	// Close file
	inputFileStream.close();

	return EXIT_SUCCESS;
}

void saveList(const std::string& list_fpath,
		const std::vector<std::string>& list) {

	// Open file
	std::ofstream outputFileStream(list_fpath.c_str(), std::fstream::out);

	// Check file
	if (outputFileStream.good() == false) {
		throw std::runtime_error(
				"Error while opening file [" + list_fpath + "] for writing\n");
	}

	// Save list to file
	for (std::string line : list) {
		outputFileStream << line << std::endl;
	}

	// Close file
	outputFileStream.close();

}

void matchKeypoints(const cv::Ptr<bfeat::DirectIndex> directIndex1, int id1,
		const std::vector<cv::KeyPoint>& keypoints1,
		cv::vector<cv::Point2f>& matchedPoints1,
		const cv::Ptr<bfeat::DirectIndex> directIndex2, int id2,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::Point2f>& matchedPoints2,
		std::vector<cv::DMatch>& matches1to2) {

	// Clean up variables received as arguments
	matchedPoints1.clear();
	matchedPoints2.clear();
	matches1to2.clear();

	// Lookup query and database images in the index and get nodes list
	bfeat::TreeNode nodes1 = directIndex1->lookUpImg(id1);
	bfeat::TreeNode nodes2 = directIndex2->lookUpImg(id2);

	// Declare and initialize iterators to the maps to intersect
	typename bfeat::TreeNode::const_iterator it1 = nodes1.begin();
	typename bfeat::TreeNode::const_iterator it2 = nodes2.begin();

	// Intersect nodes maps, solution taken from:
	// http://stackoverflow.com/questions/3772664/intersection-of-two-stl-maps
	while (it1 != nodes1.end() && it2 != nodes2.end()) {
		if (it1->first < it2->first) {
			++it1;
		} else if (it2->first < it1->first) {
			++it2;
		} else {
			// Match together all query keypoints vs candidate keypoints of an intersected node
			for (int i1 : it1->second) {
				for (int i2 : it2->second) {

					// Assert that feature ids stored in the direct index are in range
					CV_Assert(
							i1 >= 0
									&& i1
											< static_cast<int>(keypoints1.size()));
					CV_Assert(
							i2 >= 0
									&& i2
											< static_cast<int>(keypoints2.size()));

					// Extract points
					cv::Point2f point1 = keypoints1[i1].pt;
					cv::Point2f point2 = keypoints2[i2].pt;

					double dist = cv::norm(
							cv::Point(point1.x, point1.y)
									- cv::Point(point2.x, point2.y));

					// Apply a proximity threshold
					if (dist <= 40.0) {
						// Add points to vectors of matched
						matchedPoints1.push_back(point1);
						matchedPoints2.push_back(point2);

						cv::DMatch match = cv::DMatch(i1, i2,
								cv::norm(point1 - point2));
						// Set pair as a match
						matches1to2.push_back(match);
					}
				}
			}
			++it1, ++it2;
		}
	}

}

