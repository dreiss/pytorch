// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_AABB_ROI_NMS_H_
#define CAFFE2_OPERATORS_AABB_ROI_NMS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class AABBRoINMSOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AABBRoINMSOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        min_score_(this->template GetSingleArgument<float>("min_score", 0.05)),
        max_iou_(this->template GetSingleArgument<float>("max_iou", 0.3)),
        max_objects_(
            this->template GetSingleArgument<int>("max_objects", 100)),
        soft_nms_method_(GetSoftNmsMethod(this->template GetSingleArgument<std::string>("soft_nms_method", "none"))),
        soft_nms_sigma_(this->template GetSingleArgument<float>("soft_nms_sigma", 0.5)),
        soft_nms_min_score_(this->template GetSingleArgument<float>("soft_nms_min_score", 0))
  {}

  ~AABBRoINMSOp() {}

  bool RunOnDevice() override;

 protected:
  static const int SOFT_NMS_NONE = 0;
  static const int SOFT_NMS_LINEAR = 1;
  static const int SOFT_NMS_GAUSSIAN = 2;
  static int GetSoftNmsMethod(const std::string& name) {
    if (name == "none") {
      return SOFT_NMS_NONE;
    }
    if (name == "linear") {
      return SOFT_NMS_LINEAR;
    }
    if (name == "gaussian") {
      return SOFT_NMS_GAUSSIAN;
    }
    CAFFE_THROW("Unexpected soft_nms_method ", name);
    return -1;
  }

  /* Min score for output bounding boxes */
  float min_score_ = 0.05;
  /* Max allowed IoU between bounding boxes of the same class */
  float max_iou_ = 0.3;
  /* Max number of detected objects per image */
  int max_objects_ = 100;

  int soft_nms_method_;
  float soft_nms_sigma_;
  float soft_nms_min_score_;
};

} // namespace caffe2
#endif // CAFFE2_OPERATORS_AABB_ROI_NMS_H_
