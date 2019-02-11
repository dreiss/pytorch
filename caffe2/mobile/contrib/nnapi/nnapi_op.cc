/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ANDROID__
#warning "Compiling AndroidNnapiOp on non-Android platform"
#endif

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/utils/proto_utils.h"

#include "dlnnapi.h"
#include "nnapi_model_loader.h"

C10_DEFINE_string(
    caffe2_nnapi_compilation_preference,
    "sustained_speed",
    "Set to \"low_power\", \"fast_single_answer\", or \"sustained_speed\" to "
    "override Android NNAPI compilation preference");

namespace caffe2 {

class AndroidNnapiOp final : public Operator<CPUContext> {
 public:
  AndroidNnapiOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws)
      , model_inputs_(GetRepeatedArgument<int>("model_inputs"))
      , num_model_outputs_(GetSingleArgument<int>("num_model_outputs", -1))
  {
    PreferenceCode compilation_preference;
    if (FLAGS_caffe2_nnapi_compilation_preference == "low_power") {
      compilation_preference = ANEURALNETWORKS_PREFER_LOW_POWER;
    } else if (FLAGS_caffe2_nnapi_compilation_preference == "fast_single_answer") {
      compilation_preference = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
    } else if (FLAGS_caffe2_nnapi_compilation_preference == "sustained_speed") {
      compilation_preference = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED;
    } else {
      CAFFE_THROW("Invalid compilation preference: ", FLAGS_caffe2_nnapi_compilation_preference);
    }

    LoadPlatformLibrary();

    std::vector<const void*> buffers;
    std::vector<int32_t> buffer_sizes;

    for (int idx : GetRepeatedArgument<int>("weight_inputs")) {
      const Blob& blob = InputBlob(idx);  // XXX
      const void* raw_data;
      size_t nbytes;
      GetTensorContents(&blob, &raw_data, &nbytes, NULL, NULL);
      buffers.push_back(raw_data);
      buffer_sizes.push_back(nbytes);
    }

    std::string ser_model = GetSingleArgument("model", std::string());
    CAFFE_ENFORCE(!ser_model.empty());

    check_nnapi_->Model_create(&model_);
    CAFFE_ENFORCE(model_);

    int32_t num_inputs;
    int32_t num_outputs;
    int load_result = ::caffe2::nnapi::load_nnapi_model(
        nnapi_,
        model_,
        ser_model.data(),
        ser_model.size(),
        buffers.size(),
        buffers.data(),
        buffer_sizes.data(),
        0,
        nullptr,
        nullptr,
        &num_inputs,
        &num_outputs,
        nullptr);
    CAFFE_ENFORCE(load_result == 0);

    CAFFE_ENFORCE_EQ(num_inputs, model_inputs_.size());
    CAFFE_ENFORCE_EQ(num_outputs, num_model_outputs_);

    check_nnapi_->Model_finish(model_);

    check_nnapi_->Compilation_create(model_, &compilation_);
    check_nnapi_->Compilation_setPreference(compilation_, compilation_preference);
    check_nnapi_->Compilation_finish(compilation_);
  }

  ~AndroidNnapiOp() {
    // Note: These free functions accept null and cannot fail.
    nnapi_->Compilation_free(compilation_);
    nnapi_->Model_free(model_);
  }

  struct ExecutionFreer {
    void operator()(ANeuralNetworksExecution* execution) {
      nnapi_->Execution_free(execution);
    }
  };

  bool RunOnDevice() override {
    ANeuralNetworksExecution* execution;
    check_nnapi_->Execution_create(compilation_, &execution);
    std::unique_ptr<ANeuralNetworksExecution, ExecutionFreer> execution_unique_ptr(execution);

    for (int i = 0; i < model_inputs_.size(); i++) {
      const Blob* blob = Inputs()[model_inputs_[i]];
      const void* raw_data;
      size_t nbytes;
      ANeuralNetworksOperandType op_type;
      std::vector<uint32_t> dims;
      GetTensorContents(blob, &raw_data, &nbytes, &op_type, &dims);
      check_nnapi_->Execution_setInput(
          execution,
          i,
          &op_type,
          raw_data,
          nbytes);
    }

    for (int i = 0; i < num_model_outputs_; i++) {
      Blob* blob = Outputs()[i];
      void* raw_data;
      size_t nbytes;
      GetTensorContents(blob, &raw_data, &nbytes, NULL, NULL);
      check_nnapi_->Execution_setOutput(
          execution,
          i,
          NULL,
          raw_data,
          nbytes);
    }

    check_nnapi_->Execution_compute(execution);
    
    // TODO: Maybe skip this for fixed-size outputs?
    for (int i = 0; i < num_model_outputs_; i++) {
      Blob* blob = Outputs()[i];
      Tensor* tensor = nullptr;
      if (blob->IsType<Tensor>()) {
        tensor = blob->GetMutable<Tensor>();
      } else if (blob->IsType<int8::Int8TensorCPU>()) {
        auto* int8_tensor = blob->GetMutable<int8::Int8TensorCPU>();
        tensor = &int8_tensor->t;
      } else {
        CAFFE_THROW("Unknown blob type", blob->meta());
      }

      uint32_t rank;
      check_nnapi_->Execution_getOutputOperandRank(execution, i, &rank);
      std::vector<int> dims(rank);
      check_nnapi_->Execution_getOutputOperandDimensions(execution, i, (uint32_t*)dims.data());
      // TODO: Maybe check that only the batch dimension is changed?
      tensor->Resize(dims);
    }

    return true;
  }

 private:
  template <typename BLOB, typename VOID>
  static void GetTensorContents(BLOB* blob, VOID** raw_data, size_t* nbytes, ANeuralNetworksOperandType* operand, std::vector<uint32_t>* dims) {
    ANeuralNetworksOperandType op_type;
    if (operand == NULL) {
      operand = &op_type;
    }
    const Tensor* tensor = nullptr;
    if (blob->template IsType<Tensor>()) {
      GetTensorAndData(blob, &tensor, raw_data);
      operand->type = ANEURALNETWORKS_TENSOR_FLOAT32;
      operand->scale = 0;
      operand->zeroPoint = 0;
    } else if (blob->template IsType<int8::Int8TensorCPU>()) {
      const int8::Int8TensorCPU* int8_tensor;
      GetInt8TensorAndData(blob, &int8_tensor, raw_data);
      tensor = &int8_tensor->t;
      operand->type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
      operand->scale = int8_tensor->scale;
      operand->zeroPoint = int8_tensor->zero_point;
    } else {
      CAFFE_THROW("Unknown blob type", blob->meta());
    }
    *nbytes = tensor->nbytes();
    operand->dimensionCount = tensor->dim();
    if (dims != NULL) {
      dims->resize(tensor->dim());
      operand->dimensions = dims->data();
      for (int i = 0; i < dims->size(); i++) {
        (*dims)[i] = tensor->dim32(i);
      }
    }
  }

  static void GetTensorAndData(const Blob* blob, const Tensor** out_tensor, const void** raw_data) {
    const Tensor* tensor = &blob->Get<Tensor>();
    *out_tensor = tensor;
    *raw_data = tensor->raw_data();
  }

  static void GetTensorAndData(Blob* blob, const Tensor** out_tensor, void** raw_data) {
    Tensor* tensor = blob->GetMutable<Tensor>();
    *out_tensor = tensor;
    *raw_data = tensor->raw_mutable_data();
  }

  static void GetInt8TensorAndData(const Blob* blob, const int8::Int8TensorCPU** out_tensor, const void** raw_data) {
    const auto* int8_tensor = &blob->Get<int8::Int8TensorCPU>();
    *out_tensor = int8_tensor;
    *raw_data = int8_tensor->t.raw_data();
  }

  static void GetInt8TensorAndData(Blob* blob, const int8::Int8TensorCPU** out_tensor, void** raw_data) {
    auto* int8_tensor = blob->GetMutable<int8::Int8TensorCPU>();
    *out_tensor = int8_tensor;
    *raw_data = int8_tensor->t.raw_data();
  }

  static dlnnapi* nnapi_;
  static dlnnapi* check_nnapi_;
  static void LoadPlatformLibrary() {
    dlnnapi_load(&nnapi_, &check_nnapi_);
    CAFFE_ENFORCE(nnapi_->Model_free);
    CAFFE_ENFORCE(nnapi_->Compilation_free);
    CAFFE_ENFORCE(nnapi_->Execution_free);
  }

  std::vector<int> model_inputs_;
  int num_model_outputs_;
  ANeuralNetworksModel* model_{nullptr};
  ANeuralNetworksCompilation* compilation_{nullptr};
};

dlnnapi* AndroidNnapiOp::nnapi_;
dlnnapi* AndroidNnapiOp::check_nnapi_;

REGISTER_CPU_OPERATOR(AndroidNNAPI, AndroidNnapiOp);

OPERATOR_SCHEMA(AndroidNNAPI)
    .SetDoc(R"DOC(
Wrapper operator Android Neural Networks API.
  )DOC")
    .Arg("model", "(string) Serialized NNAPI model")
    .Arg("weight_inputs", "(ints) Indicies of inputs that should be passed to the NNAPI model as weights.")
    .Arg("model_inputs", "(ints) Indicies of inputs that should be passed to the NNAPI execution as inputs.")
    .Arg("num_model_outputs", "(int) Number of outputs to expect from the NNAPI model.  (Note that they must already be created and sized.)")
    ;

} // namespace caffe2
