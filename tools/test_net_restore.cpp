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

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetStderrLogging(0);
  ::google::SetLogDestination(0, "tmp/test_");
  if (argc < 4 || argc > 6) {
    LOG(ERROR) << "test_net_restore net_proto pretrained_net_proto iterations"
        << "[CPU/GPU] [Device ID]";
    return 1;
  }

  Caffe::set_phase(Caffe::TEST);

  if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    if (argc == 6) {
      device_id = atoi(argv[5]);
    }
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  int total_iter = atoi(argv[3]);


  const int kMaxLength = 256;
  char result_path[kMaxLength];

  for (int it = 0; it < total_iter; ++it) {
	const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
	const int channels = result[0]->channels();
	const int height = result[0]->height();
	const int width = result[0]->width();
	const int size = result[0]->count();
	const int gap = 3;
	cv::Mat cv_img_comp(height, width * 3 + gap * 2, CV_8U);
	int pos = 0;
	// noisy image
	const int noisy_height = result[1]->height();
	const int noisy_width = result[1]->width();
	const int h_off = (noisy_height - height) / 2;
	const int w_off = (noisy_width - width) / 2;
	const float* noisy_data_ptr = result[1]->cpu_data();
	for (int i = 0; i < height; ++i) {
	  uchar* ptr = cv_img_comp.ptr<uchar>(i);
	  for (int j = 0; j < width; ++j) {
		int val = int(noisy_data_ptr[(i + h_off) * noisy_width + j + w_off] * 255 + 0.5);
		if (val < 0) {
		  val = 0;
		}
		else if (val > 255) {
		  val = 255;
		}
		ptr[j] = static_cast<uchar>(val);
	  }
	}
	// restored image
	pos = 0;
	const float* restored_data_ptr = result[0]->cpu_data();
	for (int i = 0; i < height; ++i) {
	  uchar* ptr = cv_img_comp.ptr<uchar>(i) + width + gap;
	  for (int j = 0; j < width; ++j) {
		int val = int(restored_data_ptr[pos++] * 255 + 0.5);
		if (val < 0) {
		  LOG(ERROR) << val;
		  val = 0;
		}
		else if (val > 255) {
		  LOG(ERROR) << val;
		  val = 255;
		}
		ptr[j] = static_cast<uchar>(val);
	  }
	}
	// original image
	pos = 0;
	float mse_input_raw = 0.0;
	float mse_restored_raw = 0.0;
	float psnr_input_raw = 0.0;
	float psnr_restored_raw = 0.0;
	float mse_input = 0.0;
	float mse_restored = 0.0;
	float psnr_input = 0.0;
	float psnr_restored = 0.0;
	const float* data_ptr = result[2]->cpu_data();
	for (int i = 0; i < height; ++i) {
	  uchar* ptr = cv_img_comp.ptr<uchar>(i) + (width + gap) * 2;
	  for (int j = 0; j < width; ++j) {
		float diff = data_ptr[(i + h_off) * noisy_width + j + w_off] - noisy_data_ptr[(i + h_off) * noisy_width + j + w_off];
		mse_input_raw += diff * diff;
		diff = data_ptr[(i + h_off) * noisy_width + j + w_off] - restored_data_ptr[i * width + j];
		mse_restored_raw += diff * diff;
		int val = int(data_ptr[(i + h_off) * noisy_width + j + w_off] * 255 + 0.5);
		if (val < 0) {
		  val = 0;
		}
		else if (val > 255) {
		  val = 255;
		}
		ptr[j] = static_cast<uchar>(val);
	  }
	}
	// compute loss
	for (int i = 0; i < height; i++) {
	  uchar* ptr = cv_img_comp.ptr<uchar>(i);
	  for (int j = 0; j < width; j++) {
		float diff = ptr[j + (width + gap) * 2] - ptr[j];
		mse_input += diff * diff;
		diff = ptr[j + (width + gap) * 2] - ptr[j + (width + gap)];
		mse_restored += diff * diff;
	  }
	}

	mse_input_raw /= (height * width);
	mse_restored_raw /= (height * width);
	psnr_input_raw = 10 * (-log10(mse_input_raw));
	psnr_restored_raw = 10 * (-log10(mse_restored_raw));

	mse_input /= (height * width);
	mse_restored /= (height * width);
	psnr_input = 10 * (log10(255.0 * 255.0 / mse_input));
	psnr_restored = 10 * (log10(255.0 * 255.0 / mse_restored));
	LOG(ERROR) << "Batch " << it << ", before truncating";
	LOG(ERROR) << "Batch " << it << ", input mse: " << mse_input_raw;
	LOG(ERROR) << "Batch " << it << ", restored mse: " << mse_restored_raw;
	LOG(ERROR) << "Batch " << it << ", input psnr: " << psnr_input_raw;
	LOG(ERROR) << "Batch " << it << ", restored psnr: " << psnr_restored_raw;
	LOG(ERROR) << "Batch " << it << ", after truncating";
	LOG(ERROR) << "Batch " << it << ", input mse: " << mse_input;
	LOG(ERROR) << "Batch " << it << ", restored mse: " << mse_restored;
	LOG(ERROR) << "Batch " << it << ", input psnr: " << psnr_input;
	LOG(ERROR) << "Batch " << it << ", restored psnr: " << psnr_restored;
	cv::imshow("sup res", cv_img_comp);
	cv::waitKey();

	snprintf(result_path, kMaxLength, "tmp/%s_%f(%f)_%f(%f).bmp", 
	    argv[2], psnr_input, psnr_input_raw, psnr_restored, psnr_restored_raw);
	cv::imwrite(result_path, cv_img_comp);
  }
  return 0;
}
