/*
 * GeomVerify.cpp
 *
 *  Created on: Nov 20, 2013
 *      Author: andresf
 */

#include <boost/regex.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <VocabTree.h>

double mytime;

void matchKeypoints(const cv::Ptr<bfeat::DirectIndex> directIndex1, int id1,
		const std::vector<cv::KeyPoint>& keypoints1,
		cv::vector<cv::Point2f>& matchedPoints1, cv::Mat& img1,
		const cv::Ptr<bfeat::DirectIndex> directIndex2, int id2,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::Point2f>& matchedPoints2, cv::Mat& img2,
		std::vector<cv::DMatch>& matches1to2, double proximityThreshold,
		double similarityThreshold);

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

	if (argc < 11 || argc > 15) {
		printf(
				"\nUsage:\n"
						"\tGeomVerify <in.tree> <in.direct.index> "
						"<in.ranked.files.folder> <in.ranked.files.prefix> "
						"<in.db.descriptors.list> <in.db.keypoints.folder> <in.queries.descriptors.list> <in.queries.keypoints.list> "
						"<out.re-ranked.files.folder> <in.top.results> "
						"[in.type.binary:1] [in.ransac.thr:10] [in.proximity.thr:40] [in.similarity.thr:0.4]"
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
	double proximityThreshold = argc >= 14 ? atof(argv[13]) : 40.0;
	double similarityThreshold = argc >= 15 ? atof(argv[14]) : 0.4;

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

	printf("-- Running spatial verification using proxThr=[%f] simThr=[%f] "
			"ransacThr=[%f]\n", proximityThreshold, similarityThreshold,
			ransacThreshold);

	printf("-- Loading tree from [%s]\n", in_tree.c_str());
	mytime = cv::getTickCount();
	tree->load(in_tree);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Loaded in [%lf] ms, got [%lu] words \n", mytime, tree->size());

	printf("-- Loading direct index [%s]\n", in_direct_index.c_str());
	mytime = cv::getTickCount();
	directIndexCandidates->load(in_direct_index);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Loaded in [%lf] ms, got [%lu] images\n", mytime,
			directIndexCandidates->size());

	// Step 2a: load list of queries descriptors
	printf("-- Loading list of queries descriptors\n");
	std::vector<std::string> queries_desc_list;
	FileUtils::loadList(in_queries_desc_list, queries_desc_list);
	printf("   Loaded, got [%lu] entries\n", queries_desc_list.size());

	// Step 2b: load queries descriptors

	uint wordIdx; // unused
	double wordWeight; // unused
	int nodeAtL, imgIdx = 0;
	cv::Mat queryDescriptors;

	tree->setDirectIndexLevel(
			tree->getDepth() - 1 - directIndexCandidates->getLevel());

	printf(
			"-- Loading queries descriptors and adding them to the direct index, level=[%d]\n",
			tree->getDirectIndexLevel());
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
	printf("   Loaded and added descriptors of [%lu] images\n",
			directIndexCandidates->size());

	// Step 3a: load list of queries keypoints
	printf("-- Loading list of queries keypoints\n");
	std::vector<std::string> queries_keys_list;
	FileUtils::loadList(in_queries_keys_list, queries_keys_list);
	printf("   Loaded, got [%lu] entries\n", queries_keys_list.size());

	if (queries_desc_list.size() != queries_keys_list.size()) {
		fprintf(stderr, "Different number of entries between "
				"lists of queries descriptors and keypoints\n");
		return EXIT_FAILURE;
	}

	// Step 3b: load names of database files
	printf("-- Loading names of database files\n");
	std::vector<std::string> db_desc_list;
	FileUtils::loadList(in_db_desc_list, db_desc_list);
	printf("   Loaded, got [%lu] entries\n", db_desc_list.size());

	// Step 4/4: load and process queries keypoints
	printf("-- Loading and processing queries keypoints\n");
	std::vector<cv::KeyPoint> queryKeypoints;

	std::vector<std::string> ranked_candidates_list,
			geom_ranked_candidates_list;
	std::vector<cv::KeyPoint> candidateKeypoints;
	std::stringstream ranked_list_fname;

	std::vector<cv::DMatch> matchesCandidateToQuery;
	std::vector<cv::Point2f> matchedCandidatePoints, matchedQueryPoints;
	int candidateImgId, queryImgId, top = -1;

	cv::Mat inliers_idx, candidates_inliers, candidates_inliers_idx, H;

	cv::Mat imgMatches;
	cv::Mat queryImg, candidateImg;

	std::string queryBase, candidateBase;

	// Loop over list of queries keypoints
	for (size_t i = 0; i < queries_keys_list.size(); ++i) {
		printf("-- Processing query [%lu]\n", i);
		queryImgId = i;

		// Step 4a: load query keypoints
		printf("   Loading keypoints\n");
		FileUtils::loadKeypoints(queries_keys_list[i], queryKeypoints);
		printf("   Loaded, got [%lu] keypoints\n", queryKeypoints.size());

		// Step 4b: load list of query ranked candidates
		printf("   Loading list of ranked candidates\n");
		// Load list of query ranked candidates
		// Note: recall that elements in the lists of queries keypoints and descriptors
		// follow the same order and hence using query keypoints filename position to build
		// its ranked candidates filename its legal
		ranked_list_fname.str("");
		ranked_list_fname << in_ranked_lists_folder << "/query_" << i
				<< "_ranked.txt";
		FileUtils::loadList(ranked_list_fname.str(), ranked_candidates_list);
		printf("   Loaded, got [%lu] candidates\n",
				ranked_candidates_list.size());

		top = MIN(int(ranked_candidates_list.size()), topResults);

		candidates_inliers = cv::Mat::zeros(1, top, cv::DataType<int>::type);

		// Step 4c: load query ranked candidates
		for (size_t j = 0; int(j) < top; ++j) {

			// Load keypoints of jth ranked candidate
			printf("   Loading keypoints of candidate [%lu]\n", j);
			FileUtils::loadKeypoints(
					in_db_keys_folder + "/" + ranked_candidates_list[j]
							+ "_kpt.yaml.gz", candidateKeypoints);
			printf("   Loaded, got [%lu] keypoints\n",
					candidateKeypoints.size());

			// Searching putative matches
			printf("   Matching keypoints of query [%lu] "
					"against candidate [%lu]\n", i, j);

			// Id of database image
			std::vector<std::string>::iterator it = std::find(
					db_desc_list.begin(), db_desc_list.end(),
					"db/" + ranked_candidates_list[j] + ".yaml.gz");

			if (it == db_desc_list.end()) {
				throw std::runtime_error("Candidate [%s] not found "
						"in list of database filenames");
			}

			candidateImgId = std::distance(db_desc_list.begin(), it);

			queryBase = queries_keys_list[i].substr(8,
					queries_keys_list[i].length() - 20);
			candidateBase = ranked_candidates_list[j];

			queryImg = cv::imread("oxbuild_images/" + queryBase + ".jpg",
					CV_LOAD_IMAGE_GRAYSCALE);
			candidateImg = cv::imread(
					"oxbuild_images/" + candidateBase + ".jpg",
					CV_LOAD_IMAGE_GRAYSCALE);

			mytime = cv::getTickCount();

			matchKeypoints(directIndexQueries, queryImgId, queryKeypoints,
					matchedQueryPoints, queryImg, directIndexCandidates,
					candidateImgId, candidateKeypoints, matchedCandidatePoints,
					candidateImg, matchesCandidateToQuery, proximityThreshold,
					similarityThreshold);

			mytime = (double(cv::getTickCount()) - mytime)
					/ cv::getTickFrequency() * 1000;

			printf("   Found [%d] putative matches in [%lf] ms\n",
					int(matchesCandidateToQuery.size()), mytime);

			if ((int(matchesCandidateToQuery.size())) < 4) {
				fprintf(stderr, "   Cannot compute homography, "
						"at least 4 putative matches are needed\n");
			} else {
				// Compute a projective transformation between query and ranked file using direct index
				printf("   Computing projective transformation "
						"between query [%lu] and candidate [%lu]\n", i, j);

				inliers_idx = cv::Mat();

				mytime = cv::getTickCount();
				H = cv::findHomography(matchedCandidatePoints,
						matchedQueryPoints, CV_RANSAC, ransacThreshold,
						inliers_idx);
				mytime = (double(cv::getTickCount()) - mytime)
						/ cv::getTickFrequency() * 1000;

				// Obtain number of inliers
				candidates_inliers.at<int>(0, j) = int(sum(inliers_idx)[0]);

				printf(
						"   Computed homography in [%0.3fs], found [%d] inliers\n",
						mytime, candidates_inliers.at<int>(0, j));

				/**** Drawing inlier matches ****/
				std::vector<cv::DMatch> inliers;
				for (int i = 0; i < inliers_idx.rows; ++i) {
					if ((int) inliers_idx.at<uchar>(i) == 1) {
						inliers.push_back(matchesCandidateToQuery.at(i));
					}
				}

				cv::drawMatches(queryImg, queryKeypoints, candidateImg,
						candidateKeypoints, inliers, imgMatches,
						cv::Scalar::all(-1), cv::Scalar::all(-1),
						std::vector<char>(),
						cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				cv::imwrite(
						"oxbuild_images/match_" + queryBase + "_"
								+ ranked_candidates_list[j] + ".jpg",
						imgMatches);
				/********************************/

			}

		}

		printf("-- Re-ranking candidates list\n");

		// Re-order list of candidates by its inlier number
		sortIdx(candidates_inliers, candidates_inliers_idx, CV_SORT_DESCENDING);

		// Copying re-ranked candidates
		geom_ranked_candidates_list.clear();
		for (size_t j = 0; int(j) < top; ++j) {
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
				ranked_candidates_list.begin() + top,
				ranked_candidates_list.end());

		printf("   Done, re-ranked top [%d] candidates out of [%lu]\n", top,
				geom_ranked_candidates_list.size());

#if GVVERBOSE
		printf("Full re-ranked candidates list:\n");
		for (std::string candidate : geom_ranked_candidates_list) {
			printf("%s, ", candidate.c_str());
		}
		printf("\n");
#endif

		printf("-- Saving list of re-ranked candidates\n");

		ranked_list_fname.str("");
		ranked_list_fname << out_ranked_lists_folder << "/query_" << i
				<< "_ranked.txt";
		FileUtils::saveList(ranked_list_fname.str(),
				geom_ranked_candidates_list);

		printf("   Done, saved [%lu] entries\n",
				geom_ranked_candidates_list.size());
	}

}

void matchKeypoints(const cv::Ptr<bfeat::DirectIndex> directIndex1, int id1,
		const std::vector<cv::KeyPoint>& keypoints1,
		cv::vector<cv::Point2f>& matchedPoints1, cv::Mat& img1,
		const cv::Ptr<bfeat::DirectIndex> directIndex2, int id2,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::Point2f>& matchedPoints2, cv::Mat& img2,
		std::vector<cv::DMatch>& matches1to2, double proximityThreshold,
		double similarityThreshold) {

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

	cv::Point2f point1, point2;

	double dist, ncc;

	int windowHalfLength = 10, windowSize = 2 * windowHalfLength + 1;

	cv::Mat A, B, NCC;
	cv::Scalar meanB, stdDevB;
	cv::Scalar meanA, stdDevA;

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

					// Extract point
					point1 = keypoints1[i1].pt;
					point2 = keypoints2[i2].pt;

					dist = cv::norm(
							cv::Point(point1.x, point1.y)
									- cv::Point(point2.x, point2.y));
					// Apply a proximity threshold
					if (dist > proximityThreshold) {
						continue;
					}

					// Ignore features close to the border since they don't have enough support
					if (point1.x - windowHalfLength < 0
							|| point1.y - windowHalfLength < 0
							|| point1.x + windowHalfLength + 1 > img1.cols
							|| point1.y + windowHalfLength + 1 > img1.rows) {
						continue;
					}

					// Ignore features close to the border since they don't have enough support
					if (point2.x - windowHalfLength < 0
							|| point2.y - windowHalfLength < 0
							|| point2.x + windowHalfLength + 1 > img2.cols
							|| point2.y + windowHalfLength + 1 > img2.rows) {
						continue;
					}

					// Extract patches
					A = cv::Mat(), B = cv::Mat();
					img1(
							cv::Range(int(point1.y) - windowHalfLength,
									int(point1.y) + windowHalfLength + 1),
							cv::Range(int(point1.x) - windowHalfLength,
									int(point1.x) + windowHalfLength + 1)).convertTo(
							A, CV_32F);
					img2(
							cv::Range(int(point2.y) - windowHalfLength,
									int(point2.y) + windowHalfLength + 1),
							cv::Range(int(point2.x) - windowHalfLength,
									int(point2.x) + windowHalfLength + 1)).convertTo(
							B, CV_32F);
					meanStdDev(A, meanA, stdDevA);
					meanStdDev(B, meanB, stdDevB);

					// Computing normalized cross correlation for the patches A and B
					subtract(A, meanA, A);
					subtract(B, meanB, B);
					multiply(A, B, NCC);
					divide(NCC, stdDevA, NCC);
					divide(NCC, stdDevB, NCC);

					ncc = sum(NCC)[0] / std::pow(double(windowSize), 2);

					if (ncc < similarityThreshold) {
						continue;
					}

					// Add points to vectors of matched
					matchedPoints1.push_back(point1);
					matchedPoints2.push_back(point2);

					cv::DMatch match = cv::DMatch(i1, i2,
							cv::norm(point1 - point2));
					// Set pair as a match
					matches1to2.push_back(match);
				}
			}
			++it1, ++it2;
		}
	}

}

