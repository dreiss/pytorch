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

#include <cstdio>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

using namespace caffe2;


C10_DEFINE_string(init_net, "", "Init net path.");
C10_DEFINE_string(predict_net, "", "Predict net path.");
C10_DEFINE_string(inout_net, "", "Path to net file containing inputs and output shapes.");
C10_DEFINE_string(out_path, "", "Path to write output.");
C10_DEFINE_int(warmup_iters, 0, "Number of iterations to run for warmup.");
C10_DEFINE_int(benchmark_iters, 0, "Number of iterations to run for benchmark.");
C10_DEFINE_bool(use_caffe2_reference, false, "Use Caffe2 runtime instead of NNAPI.");

int main(int argc, char* argv[]) {
  caffe2::GlobalInit(&argc, &argv);

  CAFFE_ENFORCE(!FLAGS_init_net.empty());
  CAFFE_ENFORCE(!FLAGS_predict_net.empty());
  CAFFE_ENFORCE(!FLAGS_inout_net.empty());
  CAFFE_ENFORCE(!FLAGS_out_path.empty());

  std::unique_ptr<caffe2::Workspace> ws(new caffe2::Workspace());
  ws->GetThreadPool()->setMinWorkSize(0);

  NetDef init_net;
  NetDef predict_net;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));

  ArgumentHelper predict_helper(predict_net);
  NetDef prep_net = predict_helper.GetSingleArgument("nnapi_prep", NetDef());
  NetDef run_net = predict_helper.GetSingleArgument("nnapi_run", NetDef());

  CAFFE_ENFORCE(ws->RunNetOnce(init_net));

  NetDef input_output;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_inout_net, &input_output));
  ArgumentHelper inout_args(input_output);

  std::vector<std::string> blob_names = inout_args.GetRepeatedArgument<std::string>("blob_names");
  std::vector<std::string> ser_blobs = inout_args.GetRepeatedArgument<std::string>("ser_blobs");
  CAFFE_ENFORCE(blob_names.size() == ser_blobs.size());
  for (int i = 0; i < blob_names.size(); i++) {
    DeserializeBlob(ser_blobs[i], ws->CreateBlob(blob_names[i]));
  }

  NetDef* net_to_run;
  if (FLAGS_use_caffe2_reference) {
    std::fprintf(stderr, "Using Caffe2 reference implementation.\n");
    net_to_run = &predict_net;
  } else {
    std::fprintf(stderr, "Using NNAPI implementation.\n");
    CAFFE_ENFORCE(ws->RunNetOnce(prep_net));
    net_to_run = &run_net;
  }
  CAFFE_ENFORCE(ws->CreateNet(*net_to_run));
  std::fprintf(stderr, "Running net for output recording.\n");
  CAFFE_ENFORCE(ws->RunNet(net_to_run->name()));

  std::vector<std::string> ser_outputs;
  for (std::string name : predict_net.external_output()) {
    if (ws->HasBlob(name)) {
      ser_outputs.push_back(SerializeBlob(*ws->GetBlob(name), name));
    }
  }
  NetDef output_def;
  output_def.add_arg()->CopyFrom(MakeArgument("outputs", ser_outputs));
  CAFFE_ENFORCE(caffe2::WriteStringToFile(output_def.SerializeAsString(), FLAGS_out_path.c_str()));

  if (FLAGS_benchmark_iters == 0) {
    CAFFE_ENFORCE(FLAGS_warmup_iters == 0,
        "warmup_iters must be zero if benchmark_iters is zero.");
    return 0;
  }

  for (int i = 0; i < FLAGS_warmup_iters; i++) {
    std::fprintf(stderr, "Running warmup iteration %d\n", i);
    CAFFE_ENFORCE(ws->RunNet(net_to_run->name()));
  }
  Timer timer;
  timer.Start();
  for (int i = 0; i < FLAGS_benchmark_iters; i++) {
    std::fprintf(stderr, "Running benchmark iteration %d\n", i);
    CAFFE_ENFORCE(ws->RunNet(net_to_run->name()));
  }

  std::fprintf(stderr, "ms/run: %f\n", (double)timer.MilliSeconds() / FLAGS_benchmark_iters);
  return 0;
}
