/*
 * Photo.h
 *
 *  Created on: Aug 14, 2013
 *      Author: andresf
 */

#ifndef PHOTO_H_
#define PHOTO_H_

#include <string>

using std::string;

namespace me {

struct coordinates {
	float latitude;
	float longitude;
};

class Photo {
public:
	Photo();
	virtual ~Photo();

	int getAccuracy() const {
		return accuracy;
	}

	void setAccuracy(int accuracy) {
		this->accuracy = accuracy;
	}

	coordinates getCoords() const {
		return coords;
	}

	void setCoords(coordinates coords) {
		this->coords = coords;
	}

	long getDateTaken() const {
		return dateTaken;
	}

	void setDateTaken(long dateTaken) {
		this->dateTaken = dateTaken;
	}

	long getDateUploaded() const {
		return dateUploaded;
	}

	void setDateUploaded(long dateUploaded) {
		this->dateUploaded = dateUploaded;
	}

	bool isIsTest() const {
		return isTest;
	}

	void setIsTest(bool isTest) {
		this->isTest = isTest;
	}

	int getLicenseId() const {
		return licenseID;
	}

	void setLicenseId(int licenseId) {
		licenseID = licenseId;
	}

	long getPhotoId() const {
		return photoID;
	}

	void setPhotoId(long photoId) {
		photoID = photoId;
	}

	string getPhotoLink() const {
		return photoLink;
	}

	void setPhotoLink(string photoLink) {
		this->photoLink = photoLink;
	}

	string getPhotoTags() const {
		return photoTags;
	}

	void setPhotoTags(string photoTags) {
		this->photoTags = photoTags;
	}

	long getUserId() const {
		return userID;
	}

	void setUserId(long userId) {
		userID = userId;
	}

	int getViews() const {
		return views;
	}

	void setViews(int views) {
		this->views = views;
	}

private:
	long photoID;
	int accuracy;
	long userID;
	string photoLink;
	string photoTags;
	long dateTaken;
	long dateUploaded;
	int views;
	int licenseID;
	bool isTest;
	coordinates coords;
};

} /* namespace me */
#endif /* PHOTO_H_ */
