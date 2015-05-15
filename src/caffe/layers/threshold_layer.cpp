// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void ThresholdLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  hw_ = height_*width_;
  chw_ = channels_*hw_;
  const int count = bottom[0]->count();

  if (this->blobs_.size() > 0) {
	LOG(INFO) << "Skipping parameter initialization";
  } else {
	this->blobs_.resize(1);
	switch (this->layer_param_.threshold_param().type()) {
	case ThresholdParameter_ThresholdType_HARD:
	  // Hard threshold
	  // Initialize the threshold
	  this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
	  break;
	case ThresholdParameter_ThresholdType_SOFT_LAYER:
	  // Soft threshold with all channels sharing the same threshold
	  // Initialize the threshold
	  this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
	  this->diff_buffer_.Reshape(1, channels_, height_, width_);
	  break;
	case ThresholdParameter_ThresholdType_SOFT_CHANNEL:
	  // Soft threshold with each channel sharing the same threshold
	  // Initialize the thresholds
	  this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1));
	  this->diff_buffer_.Reshape(1, channels_, height_, width_);
	  break;
	case ThresholdParameter_ThresholdType_SOFT_POINT:
	  // Soft threshold with each point using a diifferent threshold
	  // Initialize the thresholds
	  this->blobs_[0].reset(new Blob<Dtype>(1, channels_,
		  height_, width_));
	  //this->diff_buffer_.Reshape(1, channels_, height_, width_);
	  break;
	default:
	  LOG(FATAL) << "Unknown threshold type.";
	  break;
	}
	// Fill the thresholds
	shared_ptr<Filler<Dtype> > threshold_filler(GetFiller<Dtype>(
	  this->layer_param_.threshold_param().threshold_filler()));
	threshold_filler->Fill(this->blobs_[0].get());
  }
  // Set up diff multiplier?
  if (this->layer_param_.threshold_param().type() ==
	  ThresholdParameter_ThresholdType_SOFT_CHANNEL) {
	this->diff_multiplier_.reset(new SyncedMemory(hw_ * sizeof(Dtype)));
	Dtype* diff_multiplier_data =
	  reinterpret_cast<Dtype*>(diff_multiplier_->mutable_cpu_data());
	for (int i = 0; i < hw_; ++i) {
	  diff_multiplier_data[i] = 1.;
	}
  } else if (this->layer_param_.threshold_param().type() ==
	  ThresholdParameter_ThresholdType_SOFT_LAYER) {
	this->diff_multiplier_.reset(new SyncedMemory(chw_ * sizeof(Dtype)));
	Dtype* diff_multiplier_data =
	  reinterpret_cast<Dtype*>(diff_multiplier_->mutable_cpu_data());
	for (int i = 0; i < chw_; ++i) {
	  diff_multiplier_data[i] = 1.;
	}
  }
}

template <typename Dtype>
Dtype ThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* thresholds = this->blobs_[0]->cpu_data();
  const int count = bottom[0]->count();

  switch (this->layer_param_.threshold_param().type()) {
  case ThresholdParameter_ThresholdType_HARD:
	for (int i = 0; i < count; ++i) {
	  top_data[i] = (bottom_data[i] < thresholds[0] &&
		bottom_data[i] > -thresholds[0]) ? 0 : bottom_data[i];
	}
	break;
  case ThresholdParameter_ThresholdType_SOFT_LAYER:
	for (int i = 0; i < count; ++i) {
	  if (bottom_data[i] > 0) {
		top_data[i] = max(bottom_data[i] - thresholds[0], Dtype(0.));
	  } else {
		top_data[i] = min(bottom_data[i] + thresholds[0], Dtype(0.));
	  }
	}
	break;
  case ThresholdParameter_ThresholdType_SOFT_CHANNEL:
	for (int n = 0; n < num_; ++n) {
	  for (int c = 0; c < channels_; ++c) {
		const Dtype* bottom_ch_data = &bottom_data[chw_*n + hw_*c];
		Dtype* top_ch_data = &top_data[chw_*n + hw_*c];
		for (int i = 0; i < hw_; ++i) {
		  if (bottom_ch_data[i] > 0) {
			top_ch_data[i] = max(bottom_ch_data[i] - thresholds[c], Dtype(0.));
		  } else {
			top_ch_data[i] = min(bottom_ch_data[i] + thresholds[c], Dtype(0.));
		  }
		}
	  }
	}
	break;
  case ThresholdParameter_ThresholdType_SOFT_POINT:
	for (int n = 0; n < num_; ++n) {
	  const Dtype* bottom_data_n = &bottom_data[chw_ * n];
	  Dtype* top_data_n = &top_data[chw_ * n];
	  for (int i = 0; i < chw_; ++i) {
		if (bottom_data_n[i] > 0) {
		  top_data_n[i] = max(bottom_data_n[i] - thresholds[i], Dtype(0.));
		} else {
		  top_data_n[i] = min(bottom_data_n[i] + thresholds[i], Dtype(0.));
		}
	  }
	}
	break;
  default:
	LOG(FATAL) << "Unknown threshold type.";
	break;
  }
  return Dtype(0.);
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* thresholds = this->blobs_[0]->cpu_data();
  Dtype* thresholds_diff = this->blobs_[0]->mutable_cpu_data();
  const int count = top[0]->count();

  switch (this->layer_param_.threshold_param().type()) {
  case ThresholdParameter_ThresholdType_HARD:
	thresholds_diff[0] = Dtype(0.);
	break;
  case ThresholdParameter_ThresholdType_SOFT_LAYER:
	thresholds_diff[0] = Dtype(0.);
	for (int i = 0; i < count; ++i) {
	  if (top_data[i] > FLT_EPSILON) {
		thresholds_diff[0] -= top_diff[i];
	  } else if (top_data[i] < -FLT_EPSILON) {
		thresholds_diff[0] += top_diff[i];
	  }
	}
	break;
  case ThresholdParameter_ThresholdType_SOFT_CHANNEL:
	memset(thresholds_diff, 0, sizeof(Dtype)* this->blobs_[0]->count());
	for (int n = 0; n < num_; ++n) {
	  for (int c = 0; c < channels_; ++c) {
		const Dtype* top_data_ch = &top_data[chw_*n + hw_*c];
		const Dtype* top_diff_ch = &top_diff[chw_*n + hw_*c];
		for (int i = 0; i < hw_; ++i) {
		  if (top_data_ch[i] > FLT_EPSILON) {
			thresholds_diff[c] -= top_diff_ch[i];
		  } else if (top_data_ch[i] < -FLT_EPSILON) {
			thresholds_diff[c] += top_diff_ch[i];
		  }
		}
	  }
	}
	break;
  case ThresholdParameter_ThresholdType_SOFT_POINT:
	memset(thresholds_diff, 0, sizeof(Dtype)* this->blobs_[0]->count());
	for (int n = 0; n < num_; ++n) {
	  const Dtype* top_data_n = &top_data[chw_ * n];
	  const Dtype* top_diff_n = &top_diff[chw_ * n];
	  for (int i = 0; i < chw_; ++i) {
		if (top_data_n[i] > FLT_EPSILON) {
		  thresholds_diff[i] -= top_diff_n[i];
		} else if (top_data_n[i] < -FLT_EPSILON) {
		  thresholds_diff[i] += top_diff_n[i];
		}
	  }
	}
	break;
  default:
	LOG(FATAL) << "Unknown threshold type.";
	break;
  }
  if (propagate_down) {
	for (int i = 0; i < count; ++i) {
	  bottom_diff[i] = (top_data[i] == 0) ? 0 : top_diff[i];
	}
  }
}

INSTANTIATE_CLASS(ThresholdLayer);


}  // namespace caffe
