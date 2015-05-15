// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void CheckThres(const int n, Dtype* thresholds) {
  CUDA_KERNEL_LOOP(index, n) {
	if (thresholds[index] < FLT_EPSILON) {
	  thresholds[index] = FLT_EPSILON;
	}
  }
}

template <typename Dtype>
__global__ void HardThresForward(const int n, const Dtype* thresholds,
    const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
	top_data[index] = (bottom_data[index] < thresholds[0] &&
	    bottom_data[index] > -thresholds[0]) ? 0 : bottom_data[index];
  }
}

template <typename Dtype>
__global__ void SoftLayerThresForward(const int n, const Dtype* thresholds,
    const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
	if (bottom_data[index] > thresholds[0]) {
	  top_data[index] = bottom_data[index] - thresholds[0];
	} else if (bottom_data[index] < -thresholds[0]) {
	  top_data[index] = bottom_data[index] + thresholds[0];
	} else {
	  top_data[index] = 0.;
	}
  }
}

template <typename Dtype>
__global__ void SoftChannelThresForward(const int n, const Dtype* thresholds,
    const int channels, const int patch_size,
    const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
	const Dtype th = thresholds[(index / patch_size) % channels];
	if (bottom_data[index] > th) {
	  top_data[index] = bottom_data[index] - th;
	} else if (bottom_data[index] < -th) {
	  top_data[index] = bottom_data[index] + th;
	} else {
	  top_data[index] = 0.;
	}
  }
}

template <typename Dtype>
__global__ void SoftPointThresForward(const int n, const Dtype* thresholds,
    const int channels_patch_size,
    const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
	const Dtype th = thresholds[index % channels_patch_size];
	if (bottom_data[index] > th) {
	  top_data[index] = bottom_data[index] - th;
	} else if (bottom_data[index] < -th) {
	  top_data[index] = bottom_data[index] + th;
	} else {
	  top_data[index] = 0.;
	}
  }
}

template <typename Dtype>
Dtype ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  CheckThres<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	  count, this->blobs_[0]->mutable_gpu_data());
  const Dtype* thresholds = this->blobs_[0]->gpu_data();

  switch (this->layer_param_.threshold_param().type()) {
  case ThresholdParameter_ThresholdType_HARD:
    // NOLINT_NEXT_LINE(whitespace/operators)
    HardThresForward<Dtype><<<CAFFE_GET_BLOCKS(count),
	    CAFFE_CUDA_NUM_THREADS>>>(count, thresholds,
		bottom_data, top_data);
    break;
  case ThresholdParameter_ThresholdType_SOFT_LAYER:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftLayerThresForward<Dtype><<<CAFFE_GET_BLOCKS(count),
	    CAFFE_CUDA_NUM_THREADS>>>(count, thresholds,
		bottom_data, top_data);
    break;
  case ThresholdParameter_ThresholdType_SOFT_CHANNEL:
	// NOLINT_NEXT_LINE(whitespace/operators)
	SoftChannelThresForward<Dtype><<<CAFFE_GET_BLOCKS(count),
	    CAFFE_CUDA_NUM_THREADS>>>(count, thresholds, 
		channels_, hw_,
		bottom_data, top_data);
    break;
  case ThresholdParameter_ThresholdType_SOFT_POINT:
	// NOLINT_NEXT_LINE(whitespace/operators)
	SoftPointThresForward<Dtype><<<CAFFE_GET_BLOCKS(count),
	    CAFFE_CUDA_NUM_THREADS>>>(count, thresholds,
		chw_,
		bottom_data, top_data);
	break;
  default:
    LOG(FATAL) << "Unknown threshold type.";
  }
  CUDA_POST_KERNEL_CHECK;
  return Dtype(0.);
}

template <typename Dtype>
__global__ void HardThresBackward(const int n,
    const Dtype* top_data, const Dtype* top_diff,
	Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
	bottom_diff[index] = (top_data[index] == 0) ? 0 : top_diff[index];
  }
}

template <typename Dtype>
__global__ void SoftLayerThresBackward(const int n,
    const Dtype* top_data, const Dtype* top_diff,
    Dtype* thresholds_diff, Dtype* bottom_diff,
	const int count, const bool propagate_down) {
  CUDA_KERNEL_LOOP(index, n) {
	thresholds_diff[index] = Dtype(0.);
	for (int i = index; i < count; i += n) {
	  if (top_data[i] > 0) {
		thresholds_diff[index] -= top_diff[i];
	  } else if (top_data[i] < 0) {
		thresholds_diff[index] += top_diff[i];
	  }
	}
  }
  if (!propagate_down) {
	return;
  }
  CUDA_KERNEL_LOOP(index, n) {
	for (int i = index; i < count; i += n) {
	  bottom_diff[i] = (top_data[i] == 0) ? 0 : top_diff[i];
	}
  }
}

template <typename Dtype>
__global__ void SoftChannelThresBackward(const int n,
    const Dtype* top_data, const Dtype* top_diff,
    Dtype* thresholds_diff, Dtype* bottom_diff,
    const int count, const bool propagate_down) {
  CUDA_KERNEL_LOOP(index, n) {
	thresholds_diff[index] = Dtype(0.);
	for (int i = index; i < count; i += n) {
	  if (top_data[i] > 0) {
		thresholds_diff[index] -= top_diff[i];
	  } else if (top_data[i] < 0) {
		thresholds_diff[index] += top_diff[i];
	  }
	}
  }
  if (!propagate_down) {
	return;
  }
  CUDA_KERNEL_LOOP(index, n) {
	for (int i = index; i < count; i += n) {
	  bottom_diff[i] = (top_data[i] == 0) ? 0 : top_diff[i];
	}
  }
}

template <typename Dtype>
__global__ void SoftPointThresBackward(const int n,
    const Dtype* top_data, const Dtype* top_diff,
    Dtype* thresholds_diff, Dtype* bottom_diff,
	const int count, const bool propagate_down) {
  CUDA_KERNEL_LOOP(index, n) {
	thresholds_diff[index] = Dtype(0.);
	for (int i = index; i < count; i += n) {
	  if (top_data[i] > 0) {
		thresholds_diff[index] -= top_diff[i];
	  } else if (top_data[i] < 0) {
		thresholds_diff[index] += top_diff[i];
	  }
	}
  }
  if (!propagate_down) {
	return;
  }
  CUDA_KERNEL_LOOP(index, n) {
	for (int i = index; i < count; i += n) {
	  bottom_diff[i] = (top_data[i] == 0) ? 0 : top_diff[i];
	}
  }
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  // const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  // const Dtype* thresholds = this->blobs_[0]->gpu_data();
  Dtype* thresholds_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* diff_buffer = NULL; // this->diff_buffer_.mutable_gpu_data();
  int count = (*bottom)[0]->count();

  CUDA_CHECK(cudaMemset(thresholds_diff, 0,
	  sizeof(Dtype)* this->blobs_[0]->count()));
  switch (this->layer_param_.threshold_param().type()) {
  case ThresholdParameter_ThresholdType_HARD:
    // NOLINT_NEXT_LINE(whitespace/operators)
	if (propagate_down) {
	  HardThresBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
		  CAFFE_CUDA_NUM_THREADS>>>(count, 
		  top_data, top_diff,
		  bottom_diff);
	}
    break;
  case ThresholdParameter_ThresholdType_SOFT_LAYER:
	diff_buffer = this->diff_buffer_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
	SoftLayerThresBackward<Dtype><<<CAFFE_GET_BLOCKS(chw_),
	    CAFFE_CUDA_NUM_THREADS>>>(chw_,
		top_data, top_diff,
	    diff_buffer, bottom_diff,
	    count, propagate_down);
	// cuda dot returns the result to cpu, so we temporarily change the pointer
	// mode
	CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
	    CUBLAS_POINTER_MODE_DEVICE));
	caffe_gpu_dot<Dtype>(chw_, diff_buffer,
	    reinterpret_cast<const Dtype*>(diff_multiplier_->gpu_data()),
	    thresholds_diff);
	CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
	    CUBLAS_POINTER_MODE_HOST));
    break;
  case ThresholdParameter_ThresholdType_SOFT_CHANNEL:
	diff_buffer = this->diff_buffer_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
	SoftChannelThresBackward<Dtype><<<CAFFE_GET_BLOCKS(chw_),
	    CAFFE_CUDA_NUM_THREADS>>>(chw_,
		top_data, top_diff,
		diff_buffer, bottom_diff,
		count, propagate_down);
	caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_, hw_,
		Dtype(1.), diff_buffer,
		reinterpret_cast<const Dtype*>(diff_multiplier_->gpu_data()),
		Dtype(0.), thresholds_diff);
    break;
  case ThresholdParameter_ThresholdType_SOFT_POINT:
	// NOLINT_NEXT_LINE(whitespace/operators)
	SoftPointThresBackward<Dtype><<<CAFFE_GET_BLOCKS(chw_),
	    CAFFE_CUDA_NUM_THREADS>>>(chw_,
		top_data, top_diff,
		thresholds_diff, bottom_diff,
		count, propagate_down);
	break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_CLASS(ThresholdLayer);


}  // namespace caffe
