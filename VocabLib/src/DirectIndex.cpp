/*
 * DirectIndex.cpp
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#include <DirectIndex.hpp>

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

void DirectIndex::addFeature(uint imgIdx, int nodeId, int featureId) {

	// Lookup image in the direct index
	// Note: recall that features are added in images order
	if (imgIdx + 1 != m_index.size()) {
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
	FeatureVector& fv = (*it).second;

	fv.push_back(featureId);
}

void DirectIndex::save(cv::FileStorage& fs) const {

	int imgIdx = 0;

	fs << "DirectIndex" << "[";
	for (TreeNode node : m_index) {
		fs << "{";
		fs << "NodeIndex" << imgIdx;
		fs << "Nodes" << "[";
		for (TreeNode::iterator it = node.begin(); it != node.end(); it++) {
			fs << "{";
			fs << "Features" << "[";
			for (int featIdx : it->second) {
				fs << "{:" << "FeatureIndex" << featIdx << "}";
			}
			fs << "]";
			fs << "}";
		}
		fs << "]";
		fs << "}";
		imgIdx++;
	}
	fs << "]";

}

void DirectIndex::load(cv::FileStorage& fs) {

	cv::FileNode directIndex = fs["DirectIndex"], nodes, features;

	// Verify that 'DirectIndex' is a sequence
	if (directIndex.type() != cv::FileNode::SEQ) {
		throw std::runtime_error("[DirectIndex::load] "
				"Fetched element 'DirectIndex' should be a sequence");
	}

	uint imgIdx = 0;
	int nodeId, featureId;

	for (cv::FileNodeIterator img = directIndex.begin();
			img != directIndex.end(); img++, imgIdx++) {

		(*img)["NodeIndex"] >> nodeId;
		nodes = (*img)["Nodes"];

		// Verify that 'Nodes' is a sequence
		if (nodes.type() != cv::FileNode::SEQ) {
			throw std::runtime_error("[DirectIndex::load] "
					"Fetched element 'Nodes' should be a sequence");
		}

		for (cv::FileNodeIterator node = nodes.begin(); node != nodes.end();
				node++) {
			features = (*node)["Features"];
			for (cv::FileNodeIterator feature = features.begin();
					feature != features.end(); feature++) {
				{
					(*feature)["FeatureIndex"] >> featureId;
					addFeature(imgIdx, nodeId, featureId);
				}
			}

		}

	}

}

void DirectIndex::clear() const {
//	std::vector<TreeNode>().swap(m_index);
}

} /* namespace bfeat */
