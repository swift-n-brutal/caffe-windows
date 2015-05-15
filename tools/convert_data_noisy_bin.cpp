// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "opencvlib.h"
#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

void bgr2gray(const cv::Mat& img, cv::Mat& img_out, const float scale = 1.0) {
  const float weight[3] = { 0.114f, 0.587f, 0.299f };
  const float offset = 0.0f;

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
  const float weight[3] = { 24.966f/255.0f, 128.553f/255.0f, 65.481f/255.0f };
  const float offset = 16.0f / 255.0f;

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

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetStderrLogging(0);
  ::google::SetLogDestination(0, "./cvt_images_");
  if (argc < 4 || argc > 7) {
    LOG(ERROR) << "convert_data_noisy_bin"
	    << " root_folder/ image_names result_folder/"
        << " [sigma=25] [1-gray/0-lum] [rand_seed=1]";
    return 1;
  }
  string root_folder = argv[1];
  std::ifstream infile(argv[2]);
  string result_folder = argv[3];
  vector<string> lines;
  string filename;
  while (infile >> filename) {
	lines.push_back(filename);
  }
  float sigma = 25.0;
  if (argc >= 5) {
	sigma = (float)atof(argv[4]);
  }

  bool toGray = true;
  if (argc >= 6) {
	toGray = (bool)atoi(argv[5]);
  }

  if (argc == 7) {
	int rand_seed = atoi(argv[6]);
	Caffe::set_random_seed(unsigned int(rand_seed));
  } else {
	Caffe::set_random_seed(1);
  }

  const int kMaxLength = 256;
  char result_path[kMaxLength];

  for (vector<string>::const_iterator line = lines.begin();
	  line != lines.end(); line++) {
	string& image_path = root_folder + (*line);
	cv::Mat cv_img_origin_color = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	if (!cv_img_origin_color.data) {
	  LOG(ERROR) << "Could not open or find file " << filename;
	  continue;
	}
	LOG(INFO) << "Opened file " << *line;
	// to float data
	cv::Mat cv_img_origin_color_32f;
	cv_img_origin_color.convertTo(cv_img_origin_color_32f, CV_32F, 1.0 / 255);
	// to grayscale
	cv::Mat cv_img_origin_gray_32f;
	if (toGray) {
	  bgr2gray(cv_img_origin_color_32f, cv_img_origin_gray_32f);
	} else {
	  bgr2lum(cv_img_origin_color_32f, cv_img_origin_gray_32f);
	}

	const int channels = 1;
	const int height = cv_img_origin_color.rows;
	const int width = cv_img_origin_color.cols;
	const int size = channels * height * width;
	float* image = new float[size * 2];
	caffe_rng_gaussian<float>(size, 0, sigma / 255, image + size);
	int img_pos = 0;
	for (int i = 0; i < height; ++i) {
	  float* img_ptr = cv_img_origin_gray_32f.ptr<float>(i);
	  for (int j = 0; j < width; ++j) {
		image[img_pos] = img_ptr[j];
		image[size + img_pos] += img_ptr[j];
		++img_pos;
	  }
	}

	std::ofstream fout(result_folder + (*line) + ".bin", std::ios::binary);
	fout.write((char *)(&height), sizeof(height));
	fout.write((char *)(&width), sizeof(width));
	fout.write((char*)(image), sizeof(float)*size*2);
	fout.close();
	delete image;
  }
  return 0;
}
