/*
 * DirectIndex.cpp
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#include <DirectIndex.hpp>

#include <sstream>
#include <stdexcept>

namespace bfeat {

DirectIndex::DirectIndex(int level) {
	m_level = level;
}

DirectIndex::~DirectIndex() {
}

size_t DirectIndex::size() const {
	return m_index.size();
}

int DirectIndex::getLevel() const {
	return m_level;
}

void DirectIndex::setLevel(int level) {
	m_level = level;
}

void DirectIndex::addFeature(int imgIdx, int nodeId, int featureId) {

	// Lookup image in the direct index
	// Note: recall that features are added in images order
	if (imgIdx + 1 != int(m_index.size())) {
		// Add new entry to the direct index
		m_index.push_back(TreeNode());
	}
	TreeNode& nodeMap = m_index.back();

	// Lookup node in the image index
	TreeNode::iterator it = nodeMap.find(nodeId);

	if (it == nodeMap.end()) {
		// Insert new node with empty features vector and obtain an iterator to it
		it = nodeMap.insert(
				std::pair<int, FeatureVector>(nodeId, FeatureVector())).first;
	}
	// Obtain a reference to the features vector of the node
	FeatureVector& fv = it->second;

	fv.push_back(featureId);
}

const TreeNode& DirectIndex::lookUpImg(int imgIdx) const {
	if (imgIdx < 0 || imgIdx >= int(m_index.size())) {
		std::stringstream ss;
		ss << "[DirectIndex::lookUpImg] "
				"Image index should be in the range [0, " << m_index.size()
				<< ")";
		throw std::out_of_range(ss.str());
	}
	return m_index[imgIdx];
}

void DirectIndex::save(const std::string& filename) const {

	if (m_index.size() == 0) {
		throw std::runtime_error("[DirectIndex::save] "
				"Index is empty");
	}

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[DirectIndex::save] "
				"Unable to open file [" + filename + "] for writing");
	}

	int imgIdx = 0;

	fs << "Level" << m_level;
	fs << "DirectIndex" << "[";
	for (TreeNode node : m_index) {
		fs << "{";
		fs << "ImgIndex" << imgIdx;
		fs << "Nodes" << "[";
		for (TreeNode::iterator it = node.begin(); it != node.end(); it++) {
			fs << "{";
			fs << "Features" << "[:";
			for (int featIdx : it->second) {
				fs << featIdx;
			}
			fs << "]";
			fs << "}";
		}
		fs << "]";
		fs << "}";
		imgIdx++;
	}
	fs << "]";

	fs.release();

}

void DirectIndex::load(const std::string& filename) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[DirectIndex::load] "
				"Unable to open file [" + filename + "] for reading");
	}

	m_level = int(fs["Level"]);

	cv::FileNode directIndex = fs["DirectIndex"], nodes;

	FeatureVector features;

	// Verify that 'DirectIndex' is a sequence
	if (directIndex.type() != cv::FileNode::SEQ) {
		throw std::runtime_error("[DirectIndex::load] "
				"Fetched element 'DirectIndex' should be a sequence");
	}

	int imgIdx = 0, nodeIdx;

	for (cv::FileNodeIterator img = directIndex.begin();
			img != directIndex.end(); img++) {

		(*img)["ImgIndex"] >> imgIdx;

		nodes = (*img)["Nodes"];

		// Verify that 'Nodes' is a sequence
		if (nodes.type() != cv::FileNode::SEQ) {
			throw std::runtime_error("[DirectIndex::load] "
					"Fetched element 'Nodes' should be a sequence");
		}

		nodeIdx = 0;
		for (cv::FileNodeIterator node = nodes.begin(); node != nodes.end();
				node++, nodeIdx++) {
			(*node)["Features"] >> features;
			for (int featureIdx : features) {
				addFeature(imgIdx, nodeIdx, featureIdx);
			}

		}

	}

	fs.release();

}

void DirectIndex::clear() const {
//	std::vector<TreeNode>().swap(m_index);
}

} /* namespace bfeat */
