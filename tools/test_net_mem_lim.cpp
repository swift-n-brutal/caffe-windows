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
  ::google::SetLogDestination(0, "./test_");
  if (argc < 5 || argc > 8) {
    LOG(ERROR) << "test_net_mem_lim net_proto pretrained_net_proto folder/ dbnames "
        << "[0/1] [CPU/GPU] [Device ID]";
    return 1;
  }

  Caffe::set_phase(Caffe::TEST);

  if (argc >= 7 && strcmp(argv[6], "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    if (argc == 8) {
      device_id = atoi(argv[7]);
    }
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  string root_folder = argv[3];
  std::ifstream infile(argv[4]);
  vector<string> lines;
  string filename;
  while (infile >> filename) {
	lines.push_back(filename);
  }
  // int patch_size = atoi(argv[5]);
  CHECK(caffe_test_net.num_inputs() == 1) << "Expected 1 input blob.";
  const int patch_size = caffe_test_net.input_blobs()[0]->height();

  bool keep_border = false;
  if (argc >= 6 && strcmp(argv[5], "1") == 0) {
	keep_border = true;
  }

  const int kMaxLength = 256;
  char result_path[kMaxLength];
  const int gap = 3;

  Blob<float> blob(1, 1, patch_size, patch_size);
  vector<Blob<float>*> input_blobs;
  input_blobs.push_back(&blob);
  float *input_ptr = blob.mutable_cpu_data();

  for (vector<string>::const_iterator line = lines.begin();
	  line != lines.end(); line++) {
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = false;
	options.max_open_files = 100;
	LOG(INFO) << "Opening leveldb " << *line;
	leveldb::Status status = leveldb::DB::Open(
	  options, root_folder + *line, &db);
	CHECK(status.ok()) << "Failed to open leveldb "
	  << *line << std::endl
	  << status.ToString();
	leveldb::Iterator *iter = db->NewIterator(leveldb::ReadOptions());
	iter->SeekToFirst();

	Datum image;
	image.ParseFromString(iter->value().ToString());
	const int channels = image.channels();
	const int height = image.height();
	const int width = image.width();
	const int size = channels * height * width;
	const int step = caffe_test_net.output_blobs()[0]->height();
	const int border = (patch_size - step) / 2;

	if (keep_border) {
	  const int output_height = height;
	  const int output_width = width;
	  float *restored_img = new float[channels * output_height * output_width];

	  for (int c = 0; c < channels; ++c) {
		for (int h = 0; h + step <= output_height; h += step) {
		  // LOG(ERROR) << "Row " << h;
		  for (int w = 0; w + step <= output_width; w += step) {
			// copy data to input blob
			for (int i = 0; i < patch_size; ++i) {
			  for (int j = 0; j < patch_size; ++j) {
				int h_org = h + i - border;
				int w_org = w + j - border;
				if (h_org < 0 || h_org >= height ||
				    w_org < 0 || w_org >= width ) {
				  input_ptr[i * patch_size + j] = 0.0;
				} else {
				  input_ptr[i * patch_size + j] =
					  image.float_data(size +(c*height + h_org)*width + w_org);
				}
			  }
			}
			const vector<Blob<float>*>& result =
			    caffe_test_net.Forward(input_blobs, NULL);
			const float *output_patch_ptr = result[0]->cpu_data();
			// copy output to restored image buffer
			for (int i = 0; i < step; ++i) {
			  for (int j = 0; j < step; ++j) {
				restored_img[(c*output_height + h + i)*output_width + w + j] =
				    output_patch_ptr[i * step + j];
			  }
			}
		  }
		  if (output_width % step) {
			int w = output_width - step;
			for (int i = 0; i < patch_size; ++i) {
			  for (int j = 0; j < patch_size; ++j) {
				int h_org = h + i - border;
				int w_org = w + j - border;
				if (h_org < 0 || h_org >= height ||
				    w_org < 0 || w_org >= width) {
				  input_ptr[i * patch_size + j] = 0.0;
				} else {
				  input_ptr[i * patch_size + j] = 
					  image.float_data(size +(c*height + h_org)*width + w_org);
				}
			  }
			}
			const vector<Blob<float>*>& result =
			    caffe_test_net.Forward(input_blobs, NULL);
			const float *output_patch_ptr = result[0]->cpu_data();
			for (int i = 0; i < step; ++i) {
			  for (int j = 0; j < step; ++j) {
				restored_img[(c*output_height + h + i)*output_width + w + j] = 
				    output_patch_ptr[i * step + j];
			  }
			}
		  }
		}
		if (output_height % step) {
		  int h = output_height - step;
		  // LOG(ERROR) << "Row " << h;
		  for (int w = 0; w + step <= output_width; w += step) {
			for (int i = 0; i < patch_size; ++i) {
			  for (int j = 0; j < patch_size; ++j) {
				int h_org = h + i - border;
				int w_org = w + j - border;
				if (h_org < 0 || h_org >= height ||
				    w_org < 0 || w_org >= width) {
				  input_ptr[i * patch_size + j] = 0.0;
				} else {
				  input_ptr[i * patch_size + j] = 
					  image.float_data(size +(c*height + h_org)*width + w_org);
				}
			  }
			}
			const vector<Blob<float>*>& result = 
			    caffe_test_net.Forward(input_blobs, NULL);
			const float *output_patch_ptr = result[0]->cpu_data();
			for (int i = 0; i < step; ++i) {
			  for (int j = 0; j < step; ++j) {
				restored_img[(c*output_height + h + i)*output_width + w + j] = 
				    output_patch_ptr[i * step + j];
			  }
			}
		  }
		  if (output_width % step) {
			int w = output_width - step;
			for (int i = 0; i < patch_size; ++i) {
			  for (int j = 0; j < patch_size; ++j) {
				int h_org = h + i - border;
				int w_org = w + j - border;
				if (h_org < 0 || h_org >= height ||
				    w_org < 0 || w_org >= width) {
				  input_ptr[i * patch_size + j] = 0.0;
				} else {
				  input_ptr[i * patch_size + j] =
					  image.float_data(size +(c*height + h_org)*width + w_org);
				}
			  }
			}
			const vector<Blob<float>*>& result =
			    caffe_test_net.Forward(input_blobs, NULL);
			const float *output_patch_ptr = result[0]->cpu_data();
			for (int i = 0; i < step; ++i) {
			  for (int j = 0; j < step; ++j) {
				restored_img[(c*output_height + h + i)*output_width + w + j] =
				    output_patch_ptr[i * step + j];
			  }
			}
		  }
		}
	  }
	  cv::Mat cv_img_comp(output_height, output_width * 3 + gap * 2, CV_32FC1);
	  // noisy image
	  const int h_off = border;
	  const int w_off = border;
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i);
		for (int j = 0; j < output_width; ++j) {
		  float val = image.float_data(size + i * width + j);
		  val *= 255;
		  if (val < 0) {
			val = 0;
		  }
		  else if (val > 255) {
			val = 255;
		  }
		  ptr[j] = val;
		}
	  }
	  // restored image
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i) + output_width + gap;
		for (int j = 0; j < output_width; ++j) {
		  float val = restored_img[i * output_width + j];
		  val *= 255;
		  if (val < 0) {
			val = 0;
		  }
		  else if (val > 255) {
			val = 255;
		  }
		  ptr[j] = val;
		}
	  }
	  // original image
	  float mse_input_raw = 0.0;
	  float mse_restored_raw = 0.0;
	  float psnr_input_raw = 0.0;
	  float psnr_restored_raw = 0.0;
	  float mse_input = 0.0;
	  float mse_restored = 0.0;
	  float psnr_input = 0.0;
	  float psnr_restored = 0.0;
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i) + (output_width + gap) * 2;
		for (int j = 0; j < output_width; ++j) {
		  float val = image.float_data(i * width + j);
		  float diff = val - image.float_data(size + i * width + j);
		  mse_input_raw += diff * diff;
		  diff = val - restored_img[i * output_width + j];
		  mse_restored_raw += diff * diff;
		  val *= 255;
		  if (val < 0) {
			val = 0;
		  }
		  else if (val > 255) {
			val = 255;
		  }
		  ptr[j] = val;
		}
	  }
	  // compute trancated loss
	  for (int i = 0; i < output_height; i++) {
		float* ptr = cv_img_comp.ptr<float>(i);
		for (int j = 0; j < output_width; j++) {
		  float diff = ptr[j + (output_width + gap) * 2] - ptr[j];
		  mse_input += diff * diff;
		  diff = ptr[j + (output_width + gap)*2] - ptr[j + output_width + gap];
		  mse_restored += diff * diff;
		}
	  }

	  mse_input_raw /= (output_height * output_width);
	  mse_restored_raw /= (output_height * output_width);
	  psnr_input_raw = 10 * (-log10(mse_input_raw));
	  psnr_restored_raw = 10 * (-log10(mse_restored_raw));

	  mse_input /= (output_height * output_width);
	  mse_restored /= (output_height * output_width);
	  psnr_input = 10 * (log10(255.0 * 255.0 / mse_input));
	  psnr_restored = 10 * (log10(255.0 * 255.0 / mse_restored));
	  LOG(ERROR) << *line << ", before truncating";
	  //LOG(ERROR) << "input mse: " << mse_input_raw;
	  //LOG(ERROR) << "restored mse: " << mse_restored_raw;
	  LOG(ERROR) << "input psnr: " << psnr_input_raw;
	  LOG(ERROR) << "restored psnr: " << psnr_restored_raw;
	  LOG(ERROR) << *line << ", after truncating";
	  //LOG(ERROR) << "input mse: " << mse_input;
	  //LOG(ERROR) << "restored mse: " << mse_restored;
	  LOG(ERROR) << "input psnr: " << psnr_input;
	  LOG(ERROR) << "restored psnr: " << psnr_restored;
	  //cv::imshow("denoising", cv_img_comp);
	  //cv::waitKey();

	  snprintf(result_path, kMaxLength, "%s_%f(%f)_%f(%f).png",
		(*line).c_str(), psnr_input, psnr_input_raw, psnr_restored, psnr_restored_raw);
	  cv::imwrite(result_path, cv_img_comp);

	  delete restored_img;
	} else {
	  const int output_height = height - patch_size + step;
	  const int output_width = width - patch_size + step;
	  float *restored_img = new float[channels * output_height * output_width]; //  delete

	  for (int c = 0; c < channels; ++c) {
		for (int h = 0; h + step <= output_height; h += step) {
		  // LOG(ERROR) << "Row " << h;
		  for (int w = 0; w + step <= output_width; w += step) {
			for (int i = 0; i < patch_size; ++i) {
			  for (int j = 0; j < patch_size; ++j) {
				input_ptr[i * patch_size + j] =
				    image.float_data(size +(c*height + h + i)*width + w + j);
			  }
			}
			const vector<Blob<float>*>& result =
			    caffe_test_net.Forward(input_blobs, NULL);
			const float *output_patch_ptr = result[0]->cpu_data();
			for (int i = 0; i < step; ++i) {
			  for (int j = 0; j < step; ++j) {
				restored_img[(c*output_height + h + i)*output_width + w + j] =
				    output_patch_ptr[i * step + j];
			  }
			}
		  }
		  if (output_width % step) {
			int w = output_width - step;
			for (int i = 0; i < patch_size; ++i) {
			  for (int j = 0; j < patch_size; ++j) {
				input_ptr[i * patch_size + j] =
				    image.float_data(size +(c*height + h + i)*width + w + j);
			  }
			}
			const vector<Blob<float>*>& result =
			    caffe_test_net.Forward(input_blobs, NULL);
			const float *output_patch_ptr = result[0]->cpu_data();
			for (int i = 0; i < step; ++i) {
			  for (int j = 0; j < step; ++j) {
				restored_img[(c*output_height + h + i)*output_width + w + j] =
				    output_patch_ptr[i * step + j];
			  }
			}
		  }
		}
		if (output_height % step) {
		  int h = output_height - step;
		  // LOG(ERROR) << "Row " << h;
		  for (int w = 0; w + step <= output_width; w += step) {
			for (int i = 0; i < patch_size; ++i) {
			  for (int j = 0; j < patch_size; ++j) {
				input_ptr[i * patch_size + j] =
				    image.float_data(size +(c*height + h + i)*width + w + j);
			  }
			}
			const vector<Blob<float>*>& result =
			    caffe_test_net.Forward(input_blobs, NULL);
			const float *output_patch_ptr = result[0]->cpu_data();
			for (int i = 0; i < step; ++i) {
			  for (int j = 0; j < step; ++j) {
				restored_img[(c*output_height + h + i)*output_width + w + j] =
				    output_patch_ptr[i * step + j];
			  }
			}
		  }
		  if (output_width % step) {
			int w = output_width - step;
			for (int i = 0; i < patch_size; ++i) {
			  for (int j = 0; j < patch_size; ++j) {
				input_ptr[i * patch_size + j] =
				    image.float_data(size +(c*height + h + i)*width + w + j);
			  }
			}
			const vector<Blob<float>*>& result =
			    caffe_test_net.Forward(input_blobs, NULL);
			const float *output_patch_ptr = result[0]->cpu_data();
			for (int i = 0; i < step; ++i) {
			  for (int j = 0; j < step; ++j) {
				restored_img[(c*output_height + h + i)*output_width + w + j] =
				    output_patch_ptr[i * step + j];
			  }
			}
		  }
		}
	  }
	  cv::Mat cv_img_comp(output_height, output_width * 3 + gap * 2, CV_32FC1);
	  // noisy image
	  const int h_off = border;
	  const int w_off = border;
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i);
		for (int j = 0; j < output_width; ++j) {
		  float val = image.float_data(size + (i + border) * width + j + border);
		  val *= 255;
		  if (val < 0) {
			val = 0;
		  }
		  else if (val > 255) {
			val = 255;
		  }
		  ptr[j] = val;
		}
	  }
	  // restored image
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i) +output_width + gap;
		for (int j = 0; j < output_width; ++j) {
		  float val = restored_img[i * output_width + j];
		  val *= 255;
		  if (val < 0) {
			val = 0;
		  }
		  else if (val > 255) {
			val = 255;
		  }
		  ptr[j] = val;
		}
	  }
	  // original image
	  float mse_input_raw = 0.0;
	  float mse_restored_raw = 0.0;
	  float psnr_input_raw = 0.0;
	  float psnr_restored_raw = 0.0;
	  float mse_input = 0.0;
	  float mse_restored = 0.0;
	  float psnr_input = 0.0;
	  float psnr_restored = 0.0;
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i) +(output_width + gap) * 2;
		for (int j = 0; j < output_width; ++j) {
		  float val = image.float_data((i + border) * width + j + border);
		  float diff = val - image.float_data(size + (i + border) * width + j + border);
		  mse_input_raw += diff * diff;
		  diff = val - restored_img[i * output_width + j];
		  mse_restored_raw += diff * diff;
		  val *= 255;
		  if (val < 0) {
			val = 0;
		  }
		  else if (val > 255) {
			val = 255;
		  }
		  ptr[j] = val;
		}
	  }
	  // compute truncated loss
	  for (int i = 0; i < output_height; i++) {
		float* ptr = cv_img_comp.ptr<float>(i);
		for (int j = 0; j < output_width; j++) {
		  float diff = ptr[j + (output_width + gap) * 2] - ptr[j];
		  mse_input += diff * diff;
		  diff = ptr[j + (output_width + gap) * 2] - ptr[j + (output_width + gap)];
		  mse_restored += diff * diff;
		}
	  }

	  mse_input_raw /= (output_height * output_width);
	  mse_restored_raw /= (output_height * output_width);
	  psnr_input_raw = 10 * (-log10(mse_input_raw));
	  psnr_restored_raw = 10 * (-log10(mse_restored_raw));

	  mse_input /= (output_height * output_width);
	  mse_restored /= (output_height * output_width);
	  psnr_input = 10 * (log10(255.0 * 255.0 / mse_input));
	  psnr_restored = 10 * (log10(255.0 * 255.0 / mse_restored));
	  LOG(ERROR) << *line << ", before truncating";
	  //LOG(ERROR) << "input mse: " << mse_input_raw;
	  //LOG(ERROR) << "restored mse: " << mse_restored_raw;
	  LOG(ERROR) << "input psnr: " << psnr_input_raw;
	  LOG(ERROR) << "restored psnr: " << psnr_restored_raw;
	  LOG(ERROR) << *line << ", after truncating";
	  //LOG(ERROR) << "input mse: " << mse_input;
	  //LOG(ERROR) << "restored mse: " << mse_restored;
	  LOG(ERROR) << "input psnr: " << psnr_input;
	  LOG(ERROR) << "restored psnr: " << psnr_restored;
	  //cv::imshow("denoising", cv_img_comp);
	  //cv::waitKey();

	  snprintf(result_path, kMaxLength, "%s_%f(%f)_%f(%f).png",
		(*line).c_str(), psnr_input, psnr_input_raw, psnr_restored, psnr_restored_raw);
	  cv::imwrite(result_path, cv_img_comp);

	  delete restored_img;
	}
  }
  return 0;
}
