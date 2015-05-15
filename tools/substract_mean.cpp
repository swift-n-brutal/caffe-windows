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

void bgr2lum(const cv::Mat& img, cv::Mat& img_out, const float scale = 1.0) {
  const float weight[3] = { 24.966 / 255, 128.533 / 255, 65.481 / 255};
  const float offset = 16.0 / 255;

  const int width = img.cols;
  const int height = img.rows;
  img_out.create(height, width, CV_32FC1);
  for (int i = 0; i < height; ++i) {
	float* img_out_ptr = img_out.ptr<float>(i);
	for (int j = 0; j < width; ++j) {
	  Vec3f img_point = img.at<Vec3f>(i, j);
	  img_out_ptr[j] = scale *
		(img_point[0] * weight[0] +
		img_point[1] * weight[1] +
		img_point[2] * weight[2] + offset);
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
  if (argc < 2 || argc > 4) {
	printf("Convert a set of images to the leveldb format used\n"
	  "as input for Caffe.\n"
	  "Usage:\n"
	  "    convert_noisydata DB_NAME NEW_DB_NAME [MEAN_DB_NAME]\n");
	return 1;
  }

  leveldb::DB* db_input;
  leveldb::Options options_input;
  options_input.create_if_missing = false;
  options_input.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " << argv[1];
  leveldb::Status status_input = leveldb::DB::Open(
	options_input, argv[1], &db_input);
  CHECK(status_input.ok()) << "Failed to open leveldb "
	<< argv[1] << std::endl
	<< status_input.ToString();
  leveldb::Iterator* it = db_input->NewIterator(leveldb::ReadOptions());

  int count = 0;
  vector<string> data_string;
  Datum datum;
  it->SeekToFirst();
  while (it->Valid()) {
	count++;
	it->Next();
  }
  LOG(INFO) << "A total of " << count << " images.";
  it->SeekToFirst();
  while (it->Valid()) {
	datum.ParseFromString(it->value().ToString());
	int channels = datum.channels();
	int height = datum.height();
	int width = datum.width();
	int size = channels * height * width;

	Datum _datum;
	_datum.set_channels(channels);
	_datum.set_height(height);
	_datum.set_width(width);
	_datum.clear_data();
	_datum.clear_float_data();

	float mean[3];
	if (argc == 4) {
	  for (int c = 0; c < channels; ++c) {
		mean[c] = 0.5;
	  }
	} else {
	  for (int c = 0; c < channels; ++c) {
		mean[c] = 0.0;
		for (int h = 0; h < height; ++h) {
		  int offset = size + (c * height + h) * width;
		  for (int w = 0; w < width; ++w) {
			mean[c] += datum.float_data(offset + w);
		  }
		}
		mean[c] /= size;
	  }
	}
	// 32f data
	// original image & noisy image
	for (int c = 0; c < channels; ++c) {
	  for (int h = 0; h < height; ++h) {
		int offset = (c * height + h) * width;
		for (int w = 0; w < width; ++w) {
		  _datum.add_float_data(datum.float_data(offset + w) - mean[c]);
		}
	  }
	}
	for (int c = 0; c < channels; ++c) {
	  for (int h = 0; h < height; ++h) {
		int offset = size + (c * height + h) * width;
		for (int w = 0; w < width; ++w) {
		  _datum.add_float_data(datum.float_data(offset + w) - mean[c]);
		}
	  }
	}
	string value;
	_datum.SerializeToString(&value);
	data_string.push_back(value);
	it->Next();
  }

  //delete db_input;

  leveldb::DB* db_output;
  leveldb::Options options_output;
  options_output.error_if_exists = true;
  options_output.create_if_missing = true;
  options_output.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[2];
  leveldb::Status status_output = leveldb::DB::Open(
	options_output, argv[2], &db_output);
  CHECK(status_output.ok()) << "Failed to open leveldb " << argv[2];
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  count = 0;
  // randomly shuffle data
  LOG(INFO) << "Shuffling patch samples";
  std::random_shuffle(data_string.begin(), data_string.end());
  LOG(INFO) << "A total of " << data_string.size() << " patches.";
  // write to leveldb
  for (vector<string>::const_iterator it = data_string.begin();
	  it != data_string.end(); it++) {
	snprintf(key_cstr, kMaxKeyLength, "%08d", ++count);
	batch->Put(string(key_cstr), (*it));
	if (count % 1000 == 0) {
	  db_output->Write(leveldb::WriteOptions(), batch);
	  LOG(INFO) << "Processed " << count << " files.";
	  delete batch;
	  batch = new leveldb::WriteBatch();
	}
  }

  // write the last batch
  if (count % 1000 != 0) {
    db_output->Write(leveldb::WriteOptions(), batch);
    LOG(ERROR) << "Processed " << count << " files.";
  }
  LOG(ERROR) << "Clean memory.";
  delete batch;
  delete db_output;
  //delete db_input;
  return 0;
}
