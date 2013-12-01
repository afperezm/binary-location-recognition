/*
 * DirectIndex.cpp
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#include <DirectIndex.hpp>

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

} /* namespace bfeat */
