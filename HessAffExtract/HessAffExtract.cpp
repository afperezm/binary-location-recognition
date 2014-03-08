/*
 * Test.cpp
 *
 *  Created on: Nov 20, 2013
 *      Author: andresf
 */

#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <FileUtils.hpp>

int main(int argc, char **argv) {

	if (argc != 4) {
		printf(
				"\nUsage:\n"
						"\tExtractMser <in.descriptors.folder> <out.descriptors.folder> <out.keypoints.folder>\n\n");
		return EXIT_FAILURE;
	}

	std::string in_descriptorsFolder = argv[1];
	std::string out_descriptorsFolder = argv[2];
	std::string out_keypointsFolder = argv[3];

	std::vector<std::string> files;
	printf("   Opening directory [%s]\n", in_descriptorsFolder.c_str());
	FileUtils::readFolder(in_descriptorsFolder.c_str(), files);
	printf("   Found [%lu] files\n", files.size());

	std::ifstream is;
	int rows, cols, idx, row, col;
	float x, y, a, b, c, elem;
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

	for (std::string& file : files) {

		is.open((in_descriptorsFolder + "/" + file).c_str(), std::fstream::in);

		if (is.good() == false) {
			fprintf(stderr, "Error while opening file [%s] for reading\n",
					file.c_str());
			return EXIT_FAILURE;
		}

		idx = 0;

		is >> cols;
		is >> rows;

		descriptors.release();
		descriptors = cv::Mat(rows, cols, CV_32F);

		keypoints.clear();

		cols += 5;

		while (is) {
			row = int(idx / cols);
			col = idx % cols;
			if (col == 0) {
				is >> x;
			} else if (col == 1) {
				is >> y;
				keypoints.push_back(cv::KeyPoint(cv::Point2f(x, y), 0.0));
			} else if (col == 2) {
				is >> a;
			} else if (col == 3) {
				is >> b;
			} else if (col == 4) {
				is >> c;
			} else {
				col = col - 5;
				is >> elem;
				CV_Assert(row < rows);
				CV_Assert(col < cols);
				descriptors.at<float>(row, col) = elem;
			}
			++idx;
		}

		is.close();

		CV_Assert(descriptors.rows == rows);
		CV_Assert(descriptors.cols == cols - 5);

		std::string descriptorFileName = (out_descriptorsFolder + "/"
				+ file.substr(0, file.length() - 16) + ".bin");

		printf("-- Saving feature descriptors to [%s]\n",
				descriptorFileName.c_str());

		try {
			FileUtils::saveDescriptors(descriptorFileName, descriptors);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}

		std::string keypointsFileName = (out_keypointsFolder + "/"
				+ file.substr(0, file.length() - 16) + ".yaml.gz");

		printf("-- Saving feature key-points to [%s]\n",
				keypointsFileName.c_str());

		try {
			FileUtils::saveKeypoints(keypointsFileName, keypoints);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}

	}

	return EXIT_SUCCESS;
}

///* This sample code was originally provided by Liu Liu
// * Copyright (C) 2009, Liu Liu All rights reserved.
// */
//
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include <iostream>
//#include <stdio.h>
//
//using namespace cv;
//using namespace std;
//
//static void help() {
//	cout
//			<< "\nThis program demonstrates the Maximal Extremal Region interest point detector.\n"
//					"It finds the most stable (in size) dark and white regions as a threshold is increased.\n"
//					"\nCall:\n"
//					"./mser_sample <path_and_image_filename, Default is 'puzzle.png'>\n\n";
//}
//
//static const Vec3b bcolors[] = { Vec3b(0, 0, 255), Vec3b(0, 128, 255), Vec3b(0,
//		255, 255), Vec3b(0, 255, 0), Vec3b(255, 128, 0), Vec3b(255, 255, 0),
//		Vec3b(255, 0, 0), Vec3b(255, 0, 255), Vec3b(255, 255, 255) };
//
//int main(int argc, char** argv) {
//	string path;
//	Mat img0, img, yuv, gray, ellipses;
//	help();
//
//	img0 = imread(argc != 2 ? "puzzle.png" : argv[1], 1);
//	if (img0.empty()) {
//		if (argc != 2)
//			cout << "\nUsage: mser_sample <path_to_image>\n";
//		else
//			cout << "Unable to load image " << argv[1] << endl;
//		return 0;
//	}
//
//	cvtColor(img0, yuv, COLOR_BGR2YCrCb);
//	cvtColor(img0, gray, COLOR_BGR2GRAY);
//	cvtColor(gray, img, COLOR_GRAY2BGR);
//	img.copyTo(ellipses);
//
//	vector<vector<Point> > contours;
//	double t = (double) getTickCount();
//	MSER()(yuv, contours);
//	t = (double) getTickCount() - t;
//	printf("MSER extracted %d contours in %g ms.\n", (int) contours.size(),
//			t * 1000. / getTickFrequency());
//
//	// draw mser's with different colors
//	for (int i = (int) contours.size() - 1; i >= 0; i--) {
//		const vector<Point>& r = contours[i];
//		for (int j = 0; j < (int) r.size(); j++) {
//			Point pt = r[j];
//			img.at<Vec3b>(pt) = bcolors[i % 9];
//		}
//
//		// find ellipse (it seems cvfitellipse2 have error or sth?)
//		RotatedRect box = fitEllipse(r);
//
//		box.angle = (float) CV_PI / 2 - box.angle;
//		cv::ellipse(ellipses, box, Scalar(196, 255, 255), 2);
//
////		cv::Mat patch(gray,
////				cv::Rect(box.center.x, box.center.y, box.size.width,
////						box.size.height));
////		imshow("patch", patch);
//
////		// Get the rotation matrix
////		cv::Mat T = cv::getRotationMatrix2D(box.center, box.angle, 1);
////		cv::Mat warpedPatch;
////		// Perform the affine transformation
////		cv::warpAffine(gray, warpedPatch, T, gray.size());
////		imshow("warpedPatch", warpedPatch);
////		// Extracting sub-pixels from warped I
////		// Always extracting from canonical position
////		cv::Mat dst;
////		cv::getRectSubPix(warpedPatch, box.size, box.center, dst);
//	}
//
//	imshow("original", img0);
//	imshow("response", img);
//	imshow("ellipses", ellipses);
//
//	waitKey(0);
//}
