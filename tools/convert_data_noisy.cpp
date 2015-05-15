// Copyright 2014 BVLC and contributors.
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "opencvlib.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

void bgr2gray(const cv::Mat& img, cv::Mat& img_out, const float scale = 1.0) {
  const float weight[3] = { 0.114, 0.587, 0.299 };
  const float offset = 0.0;

  const int width = img.cols;
  const int height = img.rows;
  img_out.create(height, width, CV_32FC1);
  for (int i = 0; i < height; ++i) {
	float* img_out_ptr = img_out.ptr<float>(i);
	for (int j = 0; j < width; ++j) {
	  Vec3f img_ptr = img.at<Vec3f>(i, j);
	  img_out_ptr[j] = scale *
		(img_ptr[0] * weight[0] +
		img_ptr[1] * weight[1] +
		img_ptr[2] * weight[2] + offset);
	}
  }
}

void bgr2lum(const cv::Mat& img, cv::Mat& img_out, const float scale = 1.0) {
  const float weight[3] = { 24.966 / 255, 128.533 / 255, 65.481 / 255 };
  const float offset = 16.0 / 255;

  const int width = img.cols;
  const int height = img.rows;
  img_out.create(height, width, CV_32FC1);
  for (int i = 0; i < height; ++i) {
	float* img_out_ptr = img_out.ptr<float>(i);
	for (int j = 0; j < width; ++j) {
	  Vec3f img_ptr = img.at<Vec3f>(i, j);
	  img_out_ptr[j] = scale *
		(img_ptr[0] * weight[0] +
		img_ptr[1] * weight[1] +
		img_ptr[2] * weight[2] + offset);
	}
  }
}

float Wx(float x) {
  const int a = -0.5;
  x = abs(x);
  if (x < 1.0) {
	return ((a + 2.0) * x - (a + 3.0)) * x*x + 1;
  }
  else if (x < 2.0) {
	return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  }
  return 0.0;
}

void bicubicResize(cv::Mat& img, cv::Mat& img_out, int new_width, int new_height) {
  img_out.create(new_height, new_width, CV_32FC1);
  const int width = img.cols;
  const int height = img.rows;
  float tx = (float(width)) / new_width;
  float ty = (float(height)) / new_height;
  //float neighbors[4][4];
  float wfx[4], wfy[4];
  int off_x[4], off_y[4];

  for (int i = 0; i < new_height; ++i) {
	float* img_out_ptr = img_out.ptr<float>(i);
	for (int j = 0; j < new_width; ++j) {
	  int y = int(i * ty);
	  int x = int(j * tx);
	  float fy = i * ty - y;
	  float fx = j * tx - x;
	  float val = 0.0;

	  wfx[0] = Wx(1 + fx);
	  wfx[1] = Wx(fx);
	  wfx[2] = Wx(1 - fx);
	  wfx[3] = Wx(2 - fx);

	  wfy[0] = Wx(1 + fy);
	  wfy[1] = Wx(fy);
	  wfy[2] = Wx(1 - fy);
	  wfy[3] = Wx(2 - fy);

	  if (x == 0) {
		off_x[0] = 0;
	  }
	  else {
		off_x[0] = -1;
	  }
	  off_x[1] = 0;
	  if (x >= width - 1) {
		off_x[2] = 0;
		off_x[3] = 0;
	  }
	  else {
		off_x[2] = 1;
		if (x == width - 2) {
		  off_x[3] = 1;
		}
		else {
		  off_x[3] = 2;
		}
	  }
	  if (y == 0) {
		off_y[0] = 0;
	  }
	  else {
		off_y[0] = -1;
	  }
	  off_y[1] = 0;
	  if (y >= height - 1) {
		off_y[2] = 0;
		off_y[3] = 0;
	  }
	  else {
		off_y[2] = 1;
		if (y == height - 2) {
		  off_y[3] = 1;
		}
		else {
		  off_y[3] = 2;
		}
	  }

	  // wfy * neighbors * wfx
	  for (int ki = 0; ki < 4; ++ki) {
		float* tmp_ptr = img.ptr<float>(y + off_y[ki]);
		for (int kj = 0; kj < 4; ++kj) {
		  //neighbors[ki][kj] = tmp_ptr[x + off_x[kj]];
		  val += tmp_ptr[x + off_x[kj]] * wfx[kj] * wfy[ki];
		}
	  }
	  img_out_ptr[j] = val;
	}
  }
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4 || argc > 8) {
	printf("Convert a set of images to the leveldb format used\n"
	  "as input for Caffe.\n"
	  "Usage:\n"
	  "    convert_data_noisy ROOTFOLDER/ LISTFILE DB_NAME"
	  " SIGMA CROP_SIZE STRIDE MAX_NUM\n");
	return 1;
  }
  std::ifstream infile(argv[2]);
  std::vector<string> lines;
  string filename;
  while (infile >> filename) {
	lines.push_back(filename);
  }
  std::random_shuffle(lines.begin(), lines.end());
  LOG(INFO) << "A total of " << lines.size() << " images.";

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[3];
  leveldb::Status status = leveldb::DB::Open(
	options, argv[3], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

  string root_folder(argv[1]);
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  float sigma = 25.0;								// sigma - noise level
  int crop_size = 32;                                 // to be changeable
  int stride = 14;
  int max_num = 0;
  switch (argc) {
  case 8: max_num = atoi(argv[7]);
  case 7: stride = atoi(argv[6]);
  case 6: crop_size = atoi(argv[5]);
  case 5: sigma = float(atoi(argv[4]));
  }
  LOG(INFO) << "Sigma " << sigma;
  LOG(INFO) << "Crop size " << crop_size;
  LOG(INFO) << "Stride " << stride;
  LOG(INFO) << "Max num of samples " << max_num;
  sigma /= 255;									  // sigma = sigma / 255
  vector<string> data_string;
  if (crop_size == 0) {
	// do not crop
	for (vector<string>::const_iterator line = lines.begin();
	  line != lines.end(); line++) {
	  string& filename = root_folder + (*line);
	  cv::Mat cv_img_origin_color = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	  if (!cv_img_origin_color.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		continue;
	  }
	  LOG(INFO) << "Opened file " << filename;

	  int height = cv_img_origin_color.rows;
	  int width = cv_img_origin_color.cols;
	  // to float data
	  cv::Mat cv_img_origin_color_32f;
	  cv_img_origin_color.convertTo(cv_img_origin_color_32f, CV_32F, 1.0 / 255);

	  // to grayscale
	  cv::Mat cv_img_origin_gray_32f;
	  //bgr2lum(cv_img_origin_color_32f, cv_img_origin_gray_32f);
	  bgr2gray(cv_img_origin_color_32f, cv_img_origin_gray_32f);

	  Datum _datum;
	  _datum.set_channels(1);
	  _datum.set_height(height);
	  _datum.set_width(width);
	  _datum.clear_data();
	  _datum.clear_float_data();
	  // 32f data
	  // clean image
	  for (int h = 0; h < height; h++) {
		float* ptr = cv_img_origin_gray_32f.ptr<float>(h);
		for (int w = 0; w < width; w++) {
		  _datum.add_float_data(ptr[w]);
		}
	  }
	  // noisy image
	  for (int h = 0; h < height; h++) {
		float* ptr = cv_img_origin_gray_32f.ptr<float>(h);
		for (int w = 0; w < width; w++) {
		  _datum.add_float_data(ptr[w] + caffe_rng_gaussian<float>(0.0, sigma));
		}
	  }
	  string value;
	  _datum.SerializeToString(&value);
	  data_string.push_back(value);
	}
  }
  else {
	// do crop
	int max_images = 1000;
	for (vector<string>::const_iterator line = lines.begin();
	  line != lines.end() && count < max_images; line++) {
	  string& filename = root_folder + (*line);
	  cv::Mat cv_img_origin_color = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	  if (!cv_img_origin_color.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		continue;
	  }
	  //LOG(INFO) << "Opened file " << filename;
	  int height = cv_img_origin_color.rows;
	  int width = cv_img_origin_color.cols;
	  if (height < crop_size || width < crop_size) { // skip pictures that are smaller than crop_size
		continue;
	  }
	  if (++count % 100 == 0) {
		  LOG(INFO) << "Processed " << count << " images.";
	  }
	  // to float data
	  cv::Mat cv_img_origin_color_32f;
	  cv_img_origin_color.convertTo(cv_img_origin_color_32f, CV_32F, 1. / 255);
	  // to grayscale
	  cv::Mat cv_img_origin_gray_32f;
	  //bgr2lum(cv_img_origin_color_32f, cv_img_origin_gray_32f);
	  bgr2gray(cv_img_origin_color_32f, cv_img_origin_gray_32f);
	  for (int i = 0; i + crop_size <= height; i += stride) {
		for (int j = 0; j + crop_size <= width; j += stride) {
		  if (stride == 0) {
			i = (height - crop_size) / 2;
			j = (width - crop_size) / 2;
		  }
		  cv::Mat cv_img_roi = cv_img_origin_gray_32f(
			cv::Rect(j, i, crop_size, crop_size));
		  CHECK_EQ(cv_img_roi.rows, crop_size);
		  CHECK_EQ(cv_img_roi.cols, crop_size);
		  Datum _datum;
		  _datum.set_channels(1);
		  _datum.set_height(crop_size);
		  _datum.set_width(crop_size);
		  _datum.clear_data();
		  _datum.clear_float_data();
		  // float data
		  for (int h = 0; h < crop_size; h++) {
			float* ptr = cv_img_roi.ptr<float>(h);
			for (int w = 0; w < crop_size; w++) {
			  _datum.add_float_data(ptr[w]);
			}
		  }
		  for (int h = 0; h < crop_size; h++) {
			float* ptr = cv_img_roi.ptr<float>(h);
			for (int w = 0; w < crop_size; w++) {
			  _datum.add_float_data(ptr[w] + caffe_rng_gaussian<float>(0.0, sigma));
			}
		  }

		  string value;
		  _datum.SerializeToString(&value);
		  data_string.push_back(value);
		  if (stride == 0) {
			i = height + 1;
			j = width + 1;
		  }
		}
	  }
	}
  }

  count = 0;
  // randomly shuffle data
  LOG(INFO) << "Shuffling samples";
  std::random_shuffle(data_string.begin(), data_string.end());
  LOG(INFO) << "A total of " << data_string.size() << " samples.";
  // write to leveldb
  for (vector<string>::const_iterator it = data_string.begin();
	it != data_string.end() && (max_num <= 0 || count < max_num); it++) {
	snprintf(key_cstr, kMaxKeyLength, "%09d", ++count);
	batch->Put(string(key_cstr), (*it));
	if (count % 10000 == 0) {
	  db->Write(leveldb::WriteOptions(), batch);
	  LOG(INFO) << "Processed " << count << " samples.";
	  delete batch;
	  batch = new leveldb::WriteBatch();
	}
  }

  // write the last batch
  if (count % 10000 != 0) {
	db->Write(leveldb::WriteOptions(), batch);
	LOG(ERROR) << "Processed " << count << " samples.";
  }

  infile.close();
  delete batch;
  delete db;
  return 0;
}
