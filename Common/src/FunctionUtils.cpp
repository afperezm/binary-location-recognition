/*
 * FunctionUtils.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: andresf
 */

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>

#include <bitset>
#include <stdio.h>
#include <string>
#include <stdexcept>

void FunctionUtils::printKeypoints(std::vector<cv::KeyPoint>& keypoints) {

	for (size_t i = 0; i < keypoints.size(); i++) {
		cv::KeyPoint k = keypoints[i];
		printf(
				"angle=[%f] octave=[%d] response=[%f] size=[%f] x=[%f] y=[%f] class_id=[%d]\n",
				k.angle, k.octave, k.response, k.size, k.pt.x, k.pt.y,
				k.class_id);
	}
}

// --------------------------------------------------------------------------

void FunctionUtils::printDescriptors(const cv::Mat& descriptors) {
	for (int i = 0; i < descriptors.rows; i++) {
		printf("[");
		for (int j = 0; j < descriptors.cols; j++) {
			if (descriptors.type() == CV_8U) {
				std::bitset<8> byte(descriptors.at<uchar>(i, j));
				printf("%s,", byte.to_string().c_str());
			} else {
				printf("%f,", (float) descriptors.at<float>(i, j));
			}
		}
//		int decimal = BinToDec(descriptors.row(i));
//		if (descriptors.type() == CV_8U) {
//			printf(" = %ld (%d)", decimal, NumberOfSetBits(decimal));
//		}
		printf("]\n");
	}
}

// --------------------------------------------------------------------------

void FunctionUtils::printParams(cv::Ptr<cv::Algorithm> algorithm) {
	std::vector<std::string> parameters;
	algorithm->getParams(parameters);

	for (int i = 0; i < (int) parameters.size(); i++) {
		std::string param = parameters[i];
		int type = algorithm->paramType(param);
		std::string helpText = algorithm->paramHelp(param);
		std::string typeText;

		switch (type) {
		case cv::Param::BOOLEAN:
			typeText = "bool";
			break;
		case cv::Param::INT:
			typeText = "int";
			break;
		case cv::Param::REAL:
			typeText = "real (double)";
			break;
		case cv::Param::STRING:
			typeText = "string";
			break;
		case cv::Param::MAT:
			typeText = "Mat";
			break;
		case cv::Param::ALGORITHM:
			typeText = "Algorithm";
			break;
		case cv::Param::MAT_VECTOR:
			typeText = "Mat vector";
			break;
		}

		printf("Parameter name=[%s] type=[%s] help=[%s]\n", param.c_str(),
				typeText.c_str(), helpText.c_str());
	}
}

// --------------------------------------------------------------------------

int FunctionUtils::NumberOfSetBits(int i) {
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

// --------------------------------------------------------------------------

int FunctionUtils::BinToDec(const cv::Mat& binRow) {
	if (binRow.type() != CV_8U) {
		throw std::invalid_argument(
				"BinToDec: error, received matrix is not binary");
	}
	if (binRow.rows != 1) {
		throw std::invalid_argument(
				"BinToDec: error, received matrix must have only one row");
	}
	int decimal = 0;
	for (int i = 0; i < binRow.cols; i++) {
		decimal = decimal * 2 + ((bool) binRow.at<uchar>(0, i));
	}
	return decimal;
}

// --------------------------------------------------------------------------

std::vector<std::string> FunctionUtils::split(const std::string &ss, char delim) {
	std::vector<std::string> elems;
	FunctionUtils::split(ss, delim, elems);
	return elems;
}

void FunctionUtils::split(const std::string &s, char delim,
		std::vector<std::string> &tokens) {

	std::vector<std::string>().swap(tokens);

	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		tokens.push_back(item);
	}

}

// --------------------------------------------------------------------------

std::string FunctionUtils::basify(const std::string& s) {
	char folderDelim = '/';
	char extensionDelim = '.';

	std::vector<std::string> tokens;

	split(s, folderDelim, tokens);
	if (tokens.empty()) {
		throw std::runtime_error(
				"[FunctionUtils::basify] [" + s
						+ "] does not contain folder separator [" + folderDelim
						+ "]");
	}

	std::string basename = tokens.back();
	tokens.clear();
	split(basename, extensionDelim, tokens);
	if (tokens.empty()) {
		throw std::runtime_error(
				"[FunctionUtils::basify] [" + s
						+ "] does not contain extension separator ["
						+ extensionDelim + "]");
	}

	return *tokens.begin();
}

std::string FunctionUtils::parseLandmarkName(std::vector<std::string>::const_iterator fileName) {
	std::string landmarkName("");
	std::vector<std::string> fileNameSplitted = split((*fileName), '_');
	landmarkName = std::string(fileNameSplitted[0]);
	for (int var = 1; var < (int) fileNameSplitted.size() - 2; ++var) {
		landmarkName = landmarkName + "_" + fileNameSplitted[var];
	}
	return landmarkName;
}

///**
// * Transforms a key filename to an image filename
// *
// * @param keyFilename String holding the path to a keyfile since the data set root folder.
// * @return Parsed image name
// */
//string parseImgFilename(const string keyFilename, string prefix) {
//	string imgFilename = split(keyFilename.c_str(), '/').back();
//	imgFilename.resize(imgFilename.size() - 4);
//	return imgFilename + (!prefix.empty() ? prefix : "") + IMAGE_FILE_EXTENSION;
//}
