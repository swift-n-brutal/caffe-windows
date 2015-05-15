// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
//#include <pthread.h>

#include <string>
#include <vector>

#include "opencv/highgui.h"
#include "opencv/cv.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void* NoisyDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  // NoisyDataLayer<Dtype>* layer = static_cast<NoisyDataLayer<Dtype>*>(layer_pointer);
  NoisyDataLayer<Dtype>* layer = reinterpret_cast<NoisyDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer) << "Layer type conversion error.";
  Datum datum;
  CHECK(layer->prefetch_data_) << "Prefetch memory error.";
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.data_param().scale();
  const int batch_size = layer->layer_param_.data_param().batch_size();
  const int crop_size = layer->layer_param_.data_param().crop_size();
  const bool mirror = layer->layer_param_.data_param().mirror();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;
  const Dtype* mean_ptr = layer->data_mean_.cpu_data();
  const bool scalar_mean = layer->layer_param_.data_param().has_mean_scalar();
  Dtype mean = 0;
  // const double sigma = layer->layer_param_.data_param().sigma();
  if (scalar_mean) {
	mean = layer->layer_param_.data_param().mean_scalar();
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(layer->iter_);
    CHECK(layer->iter_->Valid());
    datum.ParseFromString(layer->iter_->value().ToString());
	const int d_channels = datum.channels();
	const int d_height = datum.height();
	const int d_width = datum.width();
	const int d_size = d_channels * d_height * d_width;

	const string& data = datum.data();
	if (data.size()) {
	  //LOG(INFO) << "Loaded uint data";
	  if (crop_size) {
		if (layer->phase_ == Caffe::TRAIN
		  && mirror && caffe_rng_uniform_int(2) == 1) {
		  // uchar, crop, mirror
		  for (int j = 0; j < size; ++j) {
			Dtype noisy_datum_element =
			  static_cast<Dtype>(static_cast<uint8_t>(data[d_size + j]));
			if (!scalar_mean) {
			  mean = mean_ptr[j];
			}
			top_data[item_id * size + j] = scale *
			  (noisy_datum_element - mean);
		  }
		  int h_off, w_off;
		  h_off = (d_height - crop_size) / 2;
		  w_off = (d_width - crop_size) / 2;
		  for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < crop_size; ++h) {
			  for (int w = 0; w < crop_size; ++w) {
				int top_index = ((item_id * channels + c) * crop_size + h)
				  * crop_size + w;
				int data_index = (c * d_height + h + h_off) * d_width
				  + w + w_off;
				Dtype clean_datum_element =
				  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
				if (!scalar_mean) {
				  mean = mean_ptr[data_index];
				}
				top_label[top_index] = scale *
				  (clean_datum_element - mean);
			  }
			}
		  }
		} else {
		  // uchar, crop, no mirror
		  for (int j = 0; j < size; ++j) {
			Dtype noisy_datum_element =
			  static_cast<Dtype>(static_cast<uint8_t>(data[d_size + j]));
			if (!scalar_mean) {
			  mean = mean_ptr[j];
			}
			top_data[item_id * size + j] = scale *
			  (noisy_datum_element - mean);
		  }
		  int h_off, w_off;
		  h_off = (d_height - crop_size) / 2;
		  w_off = (d_width - crop_size) / 2;
		  for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < crop_size; ++h) {
			  for (int w = 0; w < crop_size; ++w) {
				int top_index = ((item_id * channels + c) * crop_size + h)
				  * crop_size + w;
				int data_index = (c * d_height + h + h_off) * d_width
				  + w + w_off;
				Dtype clean_datum_element =
				  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
				if (!scalar_mean) {
				  mean = mean_ptr[data_index];
				}
				top_label[top_index] = scale *
				  (clean_datum_element - mean);
			  }
			}
		  }
		}
	  } else {
		if (layer->phase_ == Caffe::TRAIN
		    && mirror && caffe_rng_uniform_int(2) == 1) {
		  // uchar, no crop, mirror
		  for (int j = 0; j < size; ++j) {
			Dtype clean_datum_element = 
			    static_cast<Dtype>(static_cast<uint8_t>(data[j]));
			Dtype noisy_datum_element =
			    static_cast<Dtype>(static_cast<uint8_t>(data[d_size + j]));
			if (!scalar_mean) {
			  mean = mean_ptr[j];
			}
			top_data[item_id * size + j] = scale *
			    (noisy_datum_element - mean);
			top_label[item_id * size + j] = scale *
			    (clean_datum_element - mean);
		  }
		} else {
		  // uchar, no crop, no mirror
		  for (int j = 0; j < size; ++j) {
			Dtype clean_datum_element =
			  static_cast<Dtype>(static_cast<uint8_t>(data[j]));
			Dtype noisy_datum_element =
			  static_cast<Dtype>(static_cast<uint8_t>(data[d_size + j]));
			if (!scalar_mean) {
			  mean = mean_ptr[j];
			}
			top_data[item_id * size + j] = scale *
			    (noisy_datum_element - mean);
			top_label[item_id * size + j] = scale *
			    (clean_datum_element - mean);
		  }
		}
	  }
	} else if (datum.float_data().size()) {
	  if (crop_size) {
		if (layer->phase_ == Caffe::TRAIN &&
		    mirror && caffe_rng_uniform_int(2) == 1) {
		  // float, crop, mirror
		  for (int j = 0; j < size; ++j) {
			if (!scalar_mean) {
			  mean = mean_ptr[j];
			}
			top_data[item_id * size + j] = scale *
			  (static_cast<Dtype>(datum.float_data(d_size + j)) - mean);
		  }
		  int h_off, w_off;
		  h_off = (d_height - crop_size) / 2;
		  w_off = (d_width - crop_size) / 2;
		  for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < crop_size; ++h) {
			  for (int w = 0; w < crop_size; ++w) {
				int top_index = ((item_id * channels + c) * crop_size + h)
				  * crop_size + w;
				int data_index = (c * d_height + h + h_off) * d_width
				  + w + w_off;
				if (!scalar_mean) {
				  mean = mean_ptr[data_index];
				}
				top_label[top_index] = scale *
				  (static_cast<Dtype>(datum.float_data(data_index)) - mean);
			  }
			}
		  }
		}
		else {
		  // float, crop, no mirror
		  for (int j = 0; j < size; ++j) {
			if (!scalar_mean) {
			  mean = mean_ptr[j];
			}
			top_data[item_id * size + j] = scale *
			  (static_cast<Dtype>(datum.float_data(d_size + j)) - mean);
		  }
		  int h_off, w_off;
		  h_off = (d_height - crop_size) / 2;
		  w_off = (d_width - crop_size) / 2;
		  for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < crop_size; ++h) {
			  for (int w = 0; w < crop_size; ++w) {
				int top_index = ((item_id * channels + c) * crop_size + h)
				  * crop_size + w;
				int data_index = (c * d_height + h + h_off) * d_width
				  + w + w_off;
				if (!scalar_mean) {
				  mean = mean_ptr[data_index];
				}
				top_label[top_index] = scale *
				  (static_cast<Dtype>(datum.float_data(data_index)) - mean);
			  }
			}
		  }
		}
	  } else {
		if (layer->phase_ == Caffe::TRAIN
		    && mirror && caffe_rng_uniform_int(2) == 1) {
		  // float, no crop, mirror
		  for (int j = 0; j < size; ++j) {
			if (!scalar_mean) {
			  mean = mean_ptr[j];
			}
			top_data[item_id * size + j] = scale *
			  (static_cast<Dtype>(datum.float_data(d_size + j)) - mean);
			top_label[item_id * size + j] = scale *
			  (static_cast<Dtype>(datum.float_data(j)) - mean);
		  }
		} else {
		  // float, no crop, no mirror
		  for (int j = 0; j < size; ++j) {
			if (!scalar_mean) {
			  mean = mean_ptr[j];
			}
			top_data[item_id * size + j] = scale *
			  (static_cast<Dtype>(datum.float_data(d_size + j)) - mean);
			top_label[item_id * size + j] = scale *
			  (static_cast<Dtype>(datum.float_data(j)) - mean);
		  }
		}
	  }
	}

    // go to the next iter
    layer->iter_->Next();
    if (!layer->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->iter_->SeekToFirst();
    }
  }

  return static_cast<void*>(NULL);
}

template <typename Dtype>
NoisyDataLayer<Dtype>::~NoisyDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void NoisyDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Data Layer takes exactly two blobs as output.";

  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
  leveldb::Status status = leveldb::DB::Open(
	  options, this->layer_param_.data_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.data_param().source() << std::endl
      << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_uniform_int(this->layer_param_.data_param().rand_skip());
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
      if (!iter_->Valid()) {
        iter_->SeekToFirst();
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());
  // image
  int crop_size = this->layer_param_.data_param().crop_size();
  if (crop_size > 0) {
	(*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
	  datum.channels(), datum.height(), datum.width());
	prefetch_data_.reset(new Blob<Dtype>(
	  this->layer_param_.data_param().batch_size(), datum.channels(),
	  datum.height(), datum.width()));
	(*top)[1]->Reshape(
		this->layer_param_.data_param().batch_size(), datum.channels(),
		crop_size, crop_size);
	prefetch_label_.reset(new Blob<Dtype>(
		this->layer_param_.data_param().batch_size(), datum.channels(),
		crop_size, crop_size));
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), datum.channels(),
		datum.height(), datum.width()));
	(*top)[1]->Reshape(
		this->layer_param_.data_param().batch_size(), datum.channels(),
		datum.height(), datum.width());
	prefetch_label_.reset(new Blob<Dtype>(
		this->layer_param_.data_param().batch_size(), datum.channels(),
		datum.height(), datum.width()));
  }
  LOG(INFO) << "output data size: " << "[" << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
	  << (*top)[0]->width() << "] [" << (*top)[1]->num() << ","
	  << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
	  << (*top)[1]->width() << "]";

  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(datum_height_, crop_size);
  CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void NoisyDataLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
      (this->layer_param_.data_param().mirror() ||
       this->layer_param_.data_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  //CHECK(!pthread_create(&thread_, NULL, NoisyDataLayerPrefetch<Dtype>,
  //      static_cast<void*>(this))) << "Pthread execution failed.";
  thread_ = thread(NoisyDataLayerPrefetch<Dtype>,reinterpret_cast<void*>(this));
}

template <typename Dtype>
void NoisyDataLayer<Dtype>::JoinPrefetchThread() {
  //CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  thread_.join();
}

template <typename Dtype>
unsigned int NoisyDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype NoisyDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
             (*top)[1]->mutable_cpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

template <typename Dtype>
void NoisyDataLayer<Dtype>::ShuffleImages() {
  return;
}

INSTANTIATE_CLASS(NoisyDataLayer);

}  // namespace caffe
