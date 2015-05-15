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

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetStderrLogging(0);
  ::google::SetLogDestination(0, "./images_");
  if (argc < 6 || argc > 10) {
    LOG(ERROR) << "test_net_images net_proto pretrained_net_proto"
	    << " folder/ image_names result_folder/"
        << " [0/1] [rand_seed=1] [CPU/GPU] [Device ID]";
    return 1;
  }

  Caffe::set_phase(Caffe::TEST);

  if (argc >= 9 && strcmp(argv[8], "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    if (argc == 10) {
      device_id = atoi(argv[9]);
    }
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  if (argc >= 8) {
	int rand_seed = atoi(argv[7]);
	Caffe::set_random_seed(unsigned int(rand_seed));
  } else {
	Caffe::set_random_seed(1);
  }

  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  string root_folder = argv[3];
  std::ifstream infile(argv[4]);
  string result_folder = argv[5];
  vector<string> lines;
  string filename;
  while (infile >> filename) {
	lines.push_back(filename);
  }
  CHECK(caffe_test_net.num_inputs() == 1) << "Expected 1 input blob.";
  const int patch_size = caffe_test_net.input_blobs()[0]->height();

  bool keep_border = false;
  if (argc >= 7 && strcmp(argv[6], "1") == 0) {
	keep_border = true;
  }

  const int kMaxLength = 256;
  char result_path[kMaxLength];
  const int gap = 1;
  const float sigma = 25.0;

  Blob<float> blob(1, 1, patch_size, patch_size);
  vector<Blob<float>*> input_blobs;
  input_blobs.push_back(&blob);
  float *input_ptr = blob.mutable_cpu_data();

  for (vector<string>::const_iterator line = lines.begin();
	  line != lines.end(); line++) {
	string& image_path = root_folder + (*line);
	cv::Mat cv_img_origin_color = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	if (!cv_img_origin_color.data) {
	  LOG(ERROR) << "Could not open or find file " << filename;
	  continue;
	}
	LOG(INFO) << "Opened file " << *line;

	const int channels = 1;
	const int height = cv_img_origin_color.rows;
	const int width = cv_img_origin_color.cols;
	const int size = channels * height * width;
	// to float data
	cv::Mat cv_img_origin_color_32f;
	cv_img_origin_color.convertTo(cv_img_origin_color_32f, CV_32F, 1.0 / 255);
	// to grayscale
	cv::Mat cv_img_origin_gray_32f;
	bgr2gray(cv_img_origin_color_32f, cv_img_origin_gray_32f);

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
				if (h_org < 0) {
				  h_org = -h_org;
				} else if (h_org >= height) {
				  h_org = height + height - h_org - 2;
				}
				if (w_org < 0) {
				  w_org = -w_org;
				} else if (w_org >= width) {
				  w_org = width + width - w_org - 2;
				}
				input_ptr[i * patch_size + j] =
				    image[size + (c*height + h_org)*width + w_org];
				/*
				if (h_org < 0 || h_org >= height ||
				    w_org < 0 || w_org >= width ) {
				  input_ptr[i * patch_size + j] = 0.0;
				} else {
				  input_ptr[i * patch_size + j] =
					  image[size + (c*height + h_org)*width + w_org];
				}*/
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
				if (h_org < 0) {
				  h_org = -h_org;
				} else if (h_org >= height) {
				  h_org = height + height - h_org - 2;
				}
				if (w_org < 0) {
				  w_org = -w_org;
				} else if (w_org >= width) {
				  w_org = width + width - w_org - 2;
				}
				input_ptr[i * patch_size + j] =
				  image[size + (c*height + h_org)*width + w_org];
				/*
				if (h_org < 0 || h_org >= height ||
				    w_org < 0 || w_org >= width) {
				  input_ptr[i * patch_size + j] = 0.0;
				} else {
				  input_ptr[i * patch_size + j] = 
					  image[size + (c*height + h_org)*width + w_org];
				}
				*/
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
				if (h_org < 0) {
				  h_org = -h_org;
				} else if (h_org >= height) {
				  h_org = height + height - h_org - 2;
				}
				if (w_org < 0) {
				  w_org = -w_org;
				} else if (w_org >= width) {
				  w_org = width + width - w_org - 2;
				}
				input_ptr[i * patch_size + j] =
				  image[size + (c*height + h_org)*width + w_org];
				/*
				if (h_org < 0 || h_org >= height ||
				    w_org < 0 || w_org >= width) {
				  input_ptr[i * patch_size + j] = 0.0;
				} else {
				  input_ptr[i * patch_size + j] = 
					  image[size +(c*height + h_org)*width + w_org];
				}
				*/
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
				if (h_org < 0) {
				  h_org = -h_org;
				} else if (h_org >= height) {
				  h_org = height + height - h_org - 2;
				}
				if (w_org < 0) {
				  w_org = -w_org;
				} else if (w_org >= width) {
				  w_org = width + width - w_org - 2;
				}
				input_ptr[i * patch_size + j] =
				  image[size + (c*height + h_org)*width + w_org];
				/*
				if (h_org < 0 || h_org >= height ||
				    w_org < 0 || w_org >= width) {
				  input_ptr[i * patch_size + j] = 0.0;
				} else {
				  input_ptr[i * patch_size + j] =
					  image[size +(c*height + h_org)*width + w_org];
				}
				*/
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
	  cv::Mat cv_img_comp(output_height, output_width, CV_32FC1);
	  // restored image
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i);
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
	  // compute loss
	  float mse_restored_raw = 0.0;
	  float psnr_restored_raw = 0.0;
	  float mse_restored = 0.0;
	  float psnr_restored = 0.0;
	  // compute trancated loss
	  img_pos = 0;
	  for (int i = 0; i < output_height; i++) {
		float* ptr = cv_img_comp.ptr<float>(i);
		for (int j = 0; j < output_width; j++) {
		  float val = image[img_pos];
		  float diff = val - restored_img[img_pos];
		  ++img_pos;
		  mse_restored_raw += diff * diff;
		  val *= 255;
		  if (val < 0) {
			val = 0;
		  } else if (val > 255) {
			val = 255;
		  }
		  diff = val - ptr[j];
		  mse_restored += diff * diff;
		}
	  }

	  mse_restored_raw /= (output_height * output_width);
	  psnr_restored_raw = 10 * (-log10(mse_restored_raw));

	  mse_restored /= (output_height * output_width);
	  psnr_restored = 10 * (log10(255.0 * 255.0 / mse_restored));
	  LOG(ERROR) << *line << ", before truncating";
	  //LOG(ERROR) << "restored mse: " << mse_restored_raw;
	  LOG(ERROR) << "restored psnr: " << psnr_restored_raw;
	  LOG(ERROR) << *line << ", after truncating";
	  //LOG(ERROR) << "restored mse: " << mse_restored;
	  LOG(ERROR) << "restored psnr: " << psnr_restored;
	  //cv::imshow("denoising", cv_img_comp);
	  //cv::waitKey();

	  snprintf(result_path, kMaxLength, "%s_%f(%f).png",
		(*line).c_str(), psnr_restored, psnr_restored_raw);
	  cv::imwrite(result_folder + result_path, cv_img_comp);

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
				    image[size + (c*height + h + i)*width + w + j];
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
				    image[size + (c*height + h + i)*width + w + j];
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
				    image[size + (c*height + h + i)*width + w + j];
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
				    image[size +(c*height + h + i)*width + w + j];
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
	  cv::Mat cv_img_comp(output_height, output_width, CV_32FC1);
	  // restored image
	  img_pos = 0;
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i);
		for (int j = 0; j < output_width; ++j) {
		  float val = restored_img[img_pos];
		  ++img_pos;
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
	  // compute loss
	  float mse_restored_raw = 0.0;
	  float psnr_restored_raw = 0.0;
	  float mse_restored = 0.0;
	  float psnr_restored = 0.0;
	  for (int i = 0; i < output_height; ++i) {
		float* ptr = cv_img_comp.ptr<float>(i);
		for (int j = 0; j < output_width; ++j) {
		  float val = image[(i + border) * width + j + border];
		  float diff = val - restored_img[i * output_width + j];
		  mse_restored_raw += diff * diff;
		  val *= 255;
		  if (val < 0) {
			val = 0;
		  }
		  else if (val > 255) {
			val = 255;
		  }
		  diff = val - ptr[j];
		  mse_restored += diff * diff;
		}
	  }

	  mse_restored_raw /= (output_height * output_width);
	  psnr_restored_raw = 10 * (-log10(mse_restored_raw));

	  mse_restored /= (output_height * output_width);
	  psnr_restored = 10 * (log10(255.0 * 255.0 / mse_restored));
	  LOG(ERROR) << *line << ", before truncating";
	  //LOG(ERROR) << "restored mse: " << mse_restored_raw;
	  LOG(ERROR) << "restored psnr: " << psnr_restored_raw;
	  LOG(ERROR) << *line << ", after truncating";
	  //LOG(ERROR) << "restored mse: " << mse_restored;
	  LOG(ERROR) << "restored psnr: " << psnr_restored;
	  //cv::imshow("denoising", cv_img_comp);
	  //cv::waitKey();

	  snprintf(result_path, kMaxLength, "%s_%f(%f).png",
		(*line).c_str(), psnr_restored, psnr_restored_raw);
	  cv::imwrite(result_folder + result_path, cv_img_comp);

	  delete restored_img;
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
