// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_ALIGN_H_
#define CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_ALIGN_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"

namespace caffe2 {

template <class Context>
class Int8AABBRoIAlignOp final : public Operator<Context> {
 public:
  Int8AABBRoIAlignOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        roi_stride_(
            this->template GetSingleArgument<float>("roi_stride", 1.0f)),
        output_height_(
            this->template GetSingleArgument<int>("output_height", 1)),
        output_width_(this->template GetSingleArgument<int>("output_width", 1)),
        sampling_height_(
            this->template GetSingleArgument<int>("sampling_height", 0)),
        sampling_width_(
            this->template GetSingleArgument<int>("sampling_width", 0)) {
    DCHECK_GT(roi_stride_, 0.0f);
    DCHECK_GT(output_height_, 0);
    DCHECK_GT(output_width_, 0);
    DCHECK_GE(sampling_height_, 0);
    DCHECK_GE(sampling_width_, 0);
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NHWC, "Int8 only supports NHWC order.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  StorageOrder order_;
  float roi_stride_;
  int output_height_;
  int output_width_;
  int sampling_height_;
  int sampling_width_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_ALIGN_H_
