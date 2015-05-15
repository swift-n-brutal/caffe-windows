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

#include "opencvlib.h"
#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

void show_filter(const string& name, shared_ptr<Blob<float>> blob,
    const int pad = 1, const int scale = 4) {
  int num = blob->num();
  int channels = blob->channels();
  int height = blob->height();
  int width = blob->width();
  int count = blob->count();
  int num_filters = num * channels;

  //const int pad = 1;
  //const int scale = 4;
  uchar pad_b = 0, pad_g = 0, pad_r = 0;
  int img_height = (height + pad) * num - pad;
  int img_width = (width + pad) * channels - pad;
  cv::Mat img;
  img.create(img_height, img_width, CV_32FC1);

  const float* data = blob->cpu_data();
  float min_val = FLT_MAX;
  float max_val = -FLT_MAX;

  for (int i = 0; i < img_height; ++i) {
	float* ptr = img.ptr<float>(i);
	for (int j = 0; j < img_width; ++j) {
	  ptr[j] = 0.0;
	}
  }

  for (int n = 0; n < num; ++n) {
	for (int c = 0; c < channels; ++c) {
	  int h_off = (height + pad) * n;
	  int w_off = (width + pad) * c;
	  float sum_sqr = 0.;
	  float sum = 0.;
	  max_val = min_val = blob->data_at(n, c, 0, 0);
	  for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
		  float val = blob->data_at(n, c, h, w);
		  sum_sqr += val*val;
		  sum += val;
		  if (val > max_val) {
			max_val = val;
		  } else if (val < min_val) {
			min_val = val;
		  }
		}
	  }
	  sum_sqr /= (height * width);
	  sum /= (height * width);
	  float denom = max_val - min_val + FLT_EPSILON;
	  LOG(INFO) << name << "_" << n << "_" << c << " : " <<
		denom << " " << sum_sqr - sum*sum << " " <<
		sum << " " << min_val << " " << max_val;
	  for (int h = 0; h < height; ++h) {
		float* ptr = img.ptr<float>(h_off + h);
		for (int w = 0; w < width; ++w) {
			float val = blob->data_at(n, c, h, w);
			ptr[w_off + w] = 255.0 * (val - min_val) / denom;
		}
	  }
	}
  }

  cv::Mat upscale_img;
  cv::resize(img, upscale_img, Size(img_width * scale, img_height * scale), 0.0, 0.0, CV_INTER_NN);
  //cv::imshow(name, upscale_img);
  cv::imwrite(name + ".png", upscale_img);
  //cv::waitKey();
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  //::google::SetStderrLogging(0);
  ::google::SetLogDestination(0, "./vis_");
  if (argc != 3) {
    LOG(ERROR) << "show_filters net_proto pretrained_net_proto";
    return 1;
  }

  Caffe::set_phase(Caffe::TEST);

  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);

  // show filters layer by layer
  const vector<string>& layer_names = caffe_test_net.layer_names();
  const vector<shared_ptr<Layer<float>>>& layers = caffe_test_net.layers();
  int num_layers = layer_names.size();
  for (int i = 0; i < num_layers; ++i) {
	const vector<shared_ptr<Blob<float>>>& blobs = layers[i]->blobs();
	int num_blobs = blobs.size();
	LOG(ERROR) << layer_names[i] << " has " << num_blobs << " blobs";
	if (num_blobs > 0) {
	  for (int j = 0; j < num_blobs; ++j) {
		LOG(ERROR) << "[" << blobs[j]->num()
		  << " " << blobs[j]->channels()
		  << " " << blobs[j]->height()
		  << " " << blobs[j]->width() << "]";
		show_filter(layer_names[i] + "_" + std::to_string(j), blobs[j]);
	  }
	}
  }

  return 0;
}
