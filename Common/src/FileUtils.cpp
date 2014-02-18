#include <FileUtils.hpp>

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <sys/stat.h>

void FileUtils::readFolder(const char* folderPath,
		std::vector<std::string>& files) {
	DIR *dir;
	struct dirent *ent;
	// Try opening folder
	if ((dir = opendir(folderPath)) != NULL) {
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
		std::sort(files.begin(), files.end());
	} else {
		throw std::runtime_error(
				"Could not open directory [" + std::string(folderPath) + "]");
	}
}

// --------------------------------------------------------------------------

void FileUtils::saveList(const std::string& list_fpath,
		const std::vector<std::string>& list) {

	// Open file
	std::ofstream outputFileStream(list_fpath.c_str(), std::fstream::out);

	// Check file
	if (outputFileStream.good() == false) {
		throw std::runtime_error(
				"Error while opening file [" + list_fpath + "] for writing\n");
	}

	// Save list to file
	for (const std::string& line : list) {
		outputFileStream << line << std::endl;
	}

	// Close file
	outputFileStream.close();
}

// --------------------------------------------------------------------------

void FileUtils::loadList(const std::string& list_fpath,
		std::vector<std::string>& list) {

	// Initializing variables
	std::ifstream inputFileStream;
	std::string line;
	list.clear();

	// Open file
	inputFileStream.open(list_fpath.c_str(), std::fstream::in);

	// Check file
	if (inputFileStream.good() == false) {
		throw std::runtime_error(
				"Error while opening file [" + list_fpath + "] for reading");
	}

	// Load list from file
	while (getline(inputFileStream, line)) {
		list.push_back(line);
	}

	// Close file
	inputFileStream.close();
}

// --------------------------------------------------------------------------

void FileUtils::saveFeatures(const std::string &filename,
		const std::vector<cv::KeyPoint>& keypoints,
		const cv::Mat& descriptors) {

#if FILEUTILSVERBOSE
	printf(
			"-- Saving feature descriptors to [%s] using OpenCV FileStorage\n",
			filename.c_str());
#endif

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::saveKeypoints] "
				"Unable to open file [" + filename + "] for writing");
	}

	fs << "TotalKeypoints" << descriptors.rows;
	fs << "DescriptorSize" << descriptors.cols; // In Bytes for binary descriptors
	fs << "DescriptorType" << descriptors.type(); // CV_8U = 0 for binary descriptors

	fs << "KeyPoints" << "[";

	for (int i = 0; i < descriptors.rows; i++) {
		cv::KeyPoint k = keypoints[i];
		fs << "{";
		fs << "x" << k.pt.x;
		fs << "y" << k.pt.y;
		fs << "size" << k.size;
		fs << "angle" << k.angle;
		fs << "response" << k.response;
		fs << "octave" << k.octave;

		fs << "descriptor" << descriptors.row(i);

		fs << "}";
	}

	fs << "]";

	fs.release();
}

// --------------------------------------------------------------------------

void FileUtils::loadFeatures(const std::string& filename,
		std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

#if FILEUTILSVERBOSE
	printf(
			"-- Loading feature descriptors from [%s] using OpenCV FileStorage\n",
			filename.c_str());
#endif

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::loadFeatures] "
				"Unable to open file [" + filename + "] for reading");
	}

	int rows, cols, type;

	rows = (int) fs["TotalKeypoints"];
	cols = (int) fs["DescriptorSize"];
	type = (int) fs["DescriptorType"];

	descriptors.create(rows, cols, type);
	keypoints.reserve(rows);

	cv::FileNode keypointsSequence = fs["KeyPoints"];

	if (keypointsSequence.type() != cv::FileNode::SEQ) {
		throw std::runtime_error("[FileUtils::loadFeatures] "
				"Fetched element 'KeyPoints' is not a sequence");
	}

	int idx = 0;

	cv::Mat featureVector;

	for (cv::FileNodeIterator it = keypointsSequence.begin();
			it != keypointsSequence.end(); it++, idx++) {

		keypoints.push_back(
				cv::KeyPoint((float) (*it)["x"], (float) (*it)["y"],
						(float) (*it)["size"], (float) (*it)["angle"],
						(float) (*it)["response"], (int) (*it)["octave"]));

		featureVector = descriptors.row(idx);

		(*it)["descriptor"] >> featureVector;
	}

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::saveDescriptors(const std::string& filename,
		const cv::Mat& descriptors) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::saveDescriptors] "
				"Unable to open file [" + filename + "] for writing");
	}

	fs << "Descriptors" << descriptors;

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::loadDescriptors(const std::string& filename,
		cv::Mat& descriptors) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::loadDescriptors] "
				"Unable to open file [" + filename + "] for reading");
	}

	descriptors.release();
	descriptors = cv::Mat();

	fs["Descriptors"] >> descriptors;

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::saveKeypoints(const std::string& filename,
		const std::vector<cv::KeyPoint>& keypoints) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::saveKeypoints] "
				"Unable to open file [" + filename + "] for writing");
	}

	fs << "KeyPoints" << "[";

	for (size_t i = 0; i < keypoints.size(); i++) {
		cv::KeyPoint k = keypoints[i];
		fs << "{:";
		fs << "x" << k.pt.x;
		fs << "y" << k.pt.y;
		fs << "size" << k.size;
		fs << "angle" << k.angle;
		fs << "response" << k.response;
		fs << "octave" << k.octave;
		fs << "}";
	}

	fs << "]";

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::loadKeypoints(const std::string& filename,
		std::vector<cv::KeyPoint>& keypoints) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::loadKeypoints] "
				"Unable to open file [" + filename + "] for reading");
	}

	cv::FileNode keypointsSequence = fs["KeyPoints"];

	if (keypointsSequence.type() != cv::FileNode::SEQ) {
		throw std::runtime_error("[FileUtils::loadKeypoints] "
				"Fetched element 'KeyPoints' is not a sequence");
	}

	keypoints.clear();

	for (cv::FileNodeIterator it = keypointsSequence.begin();
			it != keypointsSequence.end(); it++) {
		keypoints.push_back(
				cv::KeyPoint((float) (*it)["x"], (float) (*it)["y"],
						(float) (*it)["size"], (float) (*it)["angle"],
						(float) (*it)["response"], (int) (*it)["octave"]));
	}

	fs.release();

}

// --------------------------------------------------------------------------

bool FileUtils::checkFileExist(const std::string& fname) {
	struct stat buffer;
	return stat(fname.c_str(), &buffer) == 0;
}

// --------------------------------------------------------------------------

void FileUtils::loadQueriesList(std::string& filePath,
		std::vector<Query>& list) {

	// Initialize local variables
	list.clear();
	std::ifstream inputFileStream;
	Query query;

	// Open file
	inputFileStream.open(filePath.c_str(), std::fstream::in);

	// Check file
	if (inputFileStream.good() == false) {
		throw std::runtime_error(
				"Error while opening file [" + filePath + "] for reading");
	}

	// Load list from file
	while (inputFileStream >> query.name >> query.x1 >> query.y1 >> query.x2
			>> query.y2) {
		list.push_back(query);
		// Clear variable holding temporary query
		query.clear();
	}

	// Close file
	inputFileStream.close();

}

// --------------------------------------------------------------------------

void FileUtils::saveDescriptorsToBin(const std::string& filename,
		const cv::Mat& descriptors) {

	std::ofstream os;

	// Open file
	os.open(filename.c_str(),
			std::ios::out | std::ios::trunc | std::ios::binary);

	// Check file
	if (os.good() == false) {
		throw std::runtime_error(
				"Unable to open file [" + filename + "] for writing");
	}

	// Write rows byte
	os.write((char*) &descriptors.rows, sizeof(int));

	// Write columns byte
	os.write((char*) &descriptors.cols, sizeof(int));

	// Write type byte
	int type = descriptors.type();
	os.write((char*) &type, sizeof(int));

	// Write data bytes
	os.write((char*) descriptors.data,
			descriptors.elemSize() * descriptors.rows * descriptors.cols);

	// Close file
	os.close();

}

// --------------------------------------------------------------------------

void FileUtils::loadDescriptorsFromBin(const std::string& filename,
		cv::Mat& descriptors) {

	std::ifstream is;

	// Open file
	is.open(filename.c_str(), std::fstream::in | std::fstream::binary);

	// Check file
	if (is.good() == false) {
		throw std::runtime_error(
				"Unable to open file [" + filename + "] for reading");
	}

	// Obtain uncompressed file size
	is.seekg(0, is.end);
	int fileSize = is.tellg();
	is.seekg(0, is.beg);

	// Read rows byte
	int rows = -1;
	is.read((char*) &rows, sizeof(int));

	// Read columns byte
	int cols = -1;
	is.read((char*) &cols, sizeof(int));

	// Read type byte
	int type = -1;
	is.read((char*) &type, sizeof(int));

	// Compute data stream size
	long dataStreamSize = fileSize - 3 * sizeof(int);

	descriptors.release();
	descriptors = cv::Mat();
	if (type != CV_32F && type != CV_8U) {
		throw std::runtime_error("Invalid descriptors type");
	}
	descriptors.create(rows, cols, type);

	// Read data bytes
	is.read((char*) descriptors.data, dataStreamSize);

	// Close file
	is.close();

}

// --------------------------------------------------------------------------

void FileUtils::loadDescriptorsFromZippedBin(const std::string& filename,
		cv::Mat& descriptors) {

	std::ifstream zippedFile;
	boost::iostreams::filtering_istream is;

	// Open file
	zippedFile.open(filename.c_str(), std::fstream::in | std::fstream::binary);

	// Check file
	if (zippedFile.good() == false) {
		throw std::runtime_error(
				"Unable to open file [" + filename + "] for reading");
	}

	// Obtain uncompressed file size
	zippedFile.seekg(-sizeof(int), zippedFile.end);
	int fileSize = -1;
	zippedFile.read((char*) &fileSize, sizeof(int));
	zippedFile.seekg(0, zippedFile.beg);

	try {
		is.push(boost::iostreams::gzip_decompressor());
		is.push(zippedFile);

		// Read rows byte
		int rows = -1;
		is.read((char*) &rows, sizeof(int));

		// Read columns byte
		int cols = -1;
		is.read((char*) &cols, sizeof(int));

		// Read type byte
		int type = -1;
		is.read((char*) &type, sizeof(int));

		// Compute data stream size
		long dataStreamSize = fileSize - 3 * sizeof(int);

		descriptors.release();
		descriptors = cv::Mat();
		if (type != CV_32F && type != CV_8U) {
			throw std::runtime_error("Invalid descriptors type");
		}
		descriptors.create(rows, cols, type);

		// Read data bytes
		is.read((char*) descriptors.data, dataStreamSize);

	} catch (const boost::iostreams::gzip_error& e) {
		throw std::runtime_error(
				"Got error while reading file [" + std::string(e.what()) + "]");
	}

	// Close file
	zippedFile.close();

}

// --------------------------------------------------------------------------

void FileUtils::loadDescriptorsRowFromBin(const std::string& filename,
		cv::Mat& descriptors, int row) {

	FILE* filePtr = fopen(filename.c_str(), "rb");

	if (filePtr == NULL) {
		throw std::runtime_error(
				"Unable to open file [" + filename + "] for reading");
	}

	// Obtain data stream size
	fseek(filePtr, 0, SEEK_END);
	long fileSize = ftell(filePtr);
	rewind(filePtr);

	// Read rows byte
	int rows = -1;
	size_t result = fread(&rows, sizeof(int), 1, filePtr);
	CV_Assert(result == 1);
	CV_Assert(row >= 0 && row < rows);

	// Read columns byte into buffer
	int cols = -1;
	result = fread(&cols, sizeof(int), 1, filePtr);
	CV_Assert(result == 1);

	// Read type byte into buffer
	int type = -1;
	result = fread(&type, sizeof(int), 1, filePtr);
	CV_Assert(result == 1);

	if (type != CV_32F && type != CV_8U) {
		throw std::runtime_error("Invalid descriptors type");
	}

	long posBytes = ftell(filePtr);

	descriptors.release();
	if (type == CV_32F) {

		// Compute data stream size
		long dataStreamSize = cols * sizeof(float);
		CV_Assert(posBytes + rows * dataStreamSize == fileSize);
		fseek(filePtr, row * dataStreamSize, SEEK_CUR);

		// Allocate memory to contain the data bytes
		float* buffer = (float*) malloc(dataStreamSize);

		// Verify allocation was successful
		if (buffer == NULL) {
			throw std::runtime_error(
					"Unable to allocate memory to contain descriptors file");
		}

		// Read data bytes into buffer
		result = fread(buffer, dataStreamSize, 1, filePtr);
		CV_Assert(result == 1);

		descriptors = cv::Mat(1, cols, CV_32F, buffer);

	} else {

		// Compute data stream size
		long dataStreamSize = cols * sizeof(unsigned char);
		CV_Assert(posBytes + rows * dataStreamSize == fileSize);
		fseek(filePtr, row * dataStreamSize, SEEK_CUR);

		// Allocate memory to contain the data bytes
		unsigned char* buffer = (unsigned char*) malloc(dataStreamSize);

		// Verify allocation was successful
		if (buffer == NULL) {
			throw std::runtime_error(
					"Unable to allocate memory to contain descriptors file");
		}

		// Read data bytes into buffer
		result = fread(buffer, dataStreamSize, 1, filePtr);
		CV_Assert(result == 1);

		descriptors = cv::Mat(1, cols, CV_8U, buffer);

	}

	// Clean up
	fclose(filePtr);

}

// --------------------------------------------------------------------------

void FileUtils::loadDescriptorsStats(std::string& filename, MatStats& stats) {

	std::ifstream inputZippedFileStream;
	boost::iostreams::filtering_istream inputFileStream;

	std::string line, field;
	std::stringstream ss;

	enum fields {
		rows, cols, dt, data
	};

	std::string fieldsNames[] = { "rows:", "cols:", "dt:", "data:" };

	// Open file
	inputZippedFileStream.open(filename.c_str(),
			std::fstream::in | std::fstream::binary);

	// Check file
	if (inputZippedFileStream.good() == false) {
		throw std::runtime_error("[VocabTree::load] "
				"Unable to open file [" + filename + "] for reading");
	}

	int _rows = -1;
	int _cols = -1;
	std::string _type;

	try {
		inputFileStream.push(boost::iostreams::gzip_decompressor());
		inputFileStream.push(inputZippedFileStream);

		while (getline(inputFileStream, line)) {
			ss.clear();
			ss.str(line);
			ss >> field;
			if (field.compare(fieldsNames[rows]) == 0) {
				ss >> _rows;
			} else if (field.compare(fieldsNames[cols]) == 0) {
				ss >> _cols;
			} else if (field.compare(fieldsNames[dt]) == 0) {
				ss >> _type;
			} else if (field.compare(fieldsNames[data]) == 0) {
				break;
			}
		}

	} catch (const boost::iostreams::gzip_error& e) {
		throw std::runtime_error("[FileUtils::loadDescriptorsStats] "
				"Got error while parsing file [" + std::string(e.what()) + "]");
	}

	// Close file
	inputZippedFileStream.close();

	stats.rows = _rows;
	stats.cols = _cols;
	stats.descType = _type;

}
