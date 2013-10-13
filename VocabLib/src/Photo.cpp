/*
 * Photo.cpp
 *
 *  Created on: Aug 14, 2013
 *      Author: andresf
 */

#include <Photo.h>

namespace me {

Photo::Photo() :
		photoID(0), accuracy(0), userID(0), photoLink(""), photoTags(""), dateTaken(
				0), dateUploaded(0), views(0), licenseID(-1), isTest(false) {
}

Photo::~Photo() {
}

} /* namespace me */
