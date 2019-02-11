#ifndef CAFFE2_OPERATORS_INT8_TRANSPOSE_OP_H_
#define CAFFE2_OPERATORS_INT8_TRANSPOSE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"
#include "caffe2/operators/transpose_op.h"

namespace caffe2 {

namespace int8 {

class Int8TransposeOp final : public TransposeOp<CPUContext> {
 public:
  Int8TransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : TransposeOp(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& X = Inputs()[0]->Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;
    TransposeImpl<uint8_t>(X.t, &Y->t);
    return true;
  }
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_TRANSPOSE_OP_H_
