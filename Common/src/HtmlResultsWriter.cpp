/*
 * HtmlResultsWriter.cpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

#include "HtmlResultsWriter.hpp"

void HtmlResultsWriter::basifyFilename(const char *filename, char *base) {
	strcpy(base, filename);
	base[strlen(base) - 8] = 0;
}

// --------------------------------------------------------------------------

void HtmlResultsWriter::writeHeader(FILE *f, int num_nns) {
	fprintf(f, "<html>\n"
			"<header>\n"
			"<title>Vocabulary tree results</title>\n"
			"</header>\n"
			"<body>\n"
			"<h1>Vocabulary tree results</h1>\n"
			"<hr>\n\n");

	fprintf(f, "<table border=2 align=center>\n<tr>\n<th>Query image</th>\n");
	for (int i = 0; i < num_nns; i++) {
		fprintf(f, "<th>Match %d</th>\n", i + 1);
	}
	fprintf(f, "</tr>\n");
}

// --------------------------------------------------------------------------

void HtmlResultsWriter::writeRow(FILE *f, const std::string &query,
		cv::Mat& scores, cv::Mat& perm, int num_nns,
		const std::vector<std::string> &db_images) {
	char q_base[512], q_thumb[512];
	basifyFilename(query.c_str(), q_base);
	sprintf(q_thumb, "%s.thumb.jpg", q_base);

	fprintf(f,
			"<tr align=center>\n<td><img src=\"%s\" style=\"max-height:200px\"><br><p>%s</p></td>\n",
			q_thumb, q_thumb);

	for (int i = 0; i < num_nns; i++) {
		char d_base[512], d_thumb[512];
		basifyFilename(db_images[perm.at<int>(0, i)].c_str(), d_base);
		sprintf(d_thumb, "%s.thumb.jpg", d_base);

		fprintf(f,
				"<td><img src=\"%s\" style=\"max-height:200px\"><br><p>%s</p></td>\n",
				d_thumb, d_thumb);
	}

	fprintf(f, "</tr>\n<tr align=right>\n");

	fprintf(f, "<td></td>\n");
	for (int i = 0; i < num_nns; i++)
		fprintf(f, "<td>%0.5f</td>\n", scores.at<float>(0, perm.at<int>(0, i)));

	fprintf(f, "</tr>\n");
}

// --------------------------------------------------------------------------

void HtmlResultsWriter::writeFooter(FILE *f) {
	fprintf(f, "</tr>\n"
			"</table>\n"
			"<hr>\n"
			"</body>\n"
			"</html>\n");
}

// --------------------------------------------------------------------------

std::string HtmlResultsWriter::getHtml() const {
	return html.str();
}

