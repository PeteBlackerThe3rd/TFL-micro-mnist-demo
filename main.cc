/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define DEBUG_PTR_TYPE long int

#include <stdio.h>
#include "tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_dense_model.h"
#include "tensorflow/lite/experimental/micro/examples/mnist_demo/example_mnist_input.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

const char *typeString(TfLiteType type)
{
    static const char noTypeStr[] = "NoType";
    static const char float32Str[] = "Float32";
    static const char int32Str[] = "Int32";
    static const char uInt8Str[] = "UInt8";
    static const char int64Str[] = "Int64";
    static const char stringStr[] = "String";
    static const char boolStr[] = "Bool";
    static const char int16Str[] = "Int16";
    static const char complex64Str[] = "Complex64";
    static const char int8Str[] = "Int8";

    switch(type)
    {
        case kTfLiteFloat32 :   return float32Str;
        case kTfLiteInt32 :     return int32Str;
        case kTfLiteUInt8 :     return uInt8Str;
        case kTfLiteInt64 :     return int64Str;
        case kTfLiteString :    return stringStr;
        case kTfLiteBool :      return boolStr;
        case kTfLiteInt16 :     return int16Str;
        case kTfLiteComplex64 : return complex64Str;
        case kTfLiteInt8 :      return int8Str;
        default :               return noTypeStr;
    }
}

void printTensorDetails(TfLiteTensor* tensor)
{
  printf("Rank %d, type [%s], shape [", tensor->dims->size, typeString(tensor->type));
  for (int d=0; d<tensor->dims->size; ++d)
  {
    if (d != 0)
      printf(", ");
    printf("%d", tensor->dims->data[d]);
  }
  printf("]\n");
}

// Black Magic!
#define IS_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)

int main(int argc, char* argv[]) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  if (IS_BIG_ENDIAN)
    printf("System is Big Endian.\n");
  else
    printf("System is Little Endian.\n");

  if (FLATBUFFERS_LITTLEENDIAN)
    printf("FlatBuffers thinks the system is Little Endian.\n");
  else
    printf("FlatBuffers thinks the system is Big Endian.\n");

  printf("Parsing model FlatBuffer.\n");

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(mnist_dense_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  printf("Model parsed.\n");

  // This pulls in all the operation implementations we need.
  printf("Pull in operation implementations.");
  tflite::ops::micro::AllOpsResolver resolver;
  printf("Done.\n");

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  printf("Allocate memory buffer.\n");
  const int tensor_arena_size = 200 * 1024;
  uint8_t tensor_arena[tensor_arena_size];
  tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                                 tensor_arena_size);
  printf("Done. Allocatated 200k at [0x%016X] %u\n", (DEBUG_PTR_TYPE)(&tensor_arena), (DEBUG_PTR_TYPE)(&tensor_arena));

  // Build an interpreter to run the model with.
  printf("Build interpreter.\n");
  tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                       error_reporter);
  printf("Done.\n");

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* model_input = interpreter.input(0);
  printf("Input tensor address is [0x%016X]\n", (DEBUG_PTR_TYPE)model_input);
  printf("Details of input tensors 0 :\n");
  printTensorDetails(model_input);

  printf("Setting input data.\n");
  for (int d=0; d<784; ++d)
    model_input->data.f[d] = ((float*)exampleInputDataHex)[d];
  printf("Done.\n");

  printf("Output tensor address is [0x%016X]\n", (DEBUG_PTR_TYPE)interpreter.output(0));

  // perform inference
  printf("Perform inference.\n");
  TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      printf("Invoke failed.\n");
      return 1;
    }
  printf("Done.\n");

  TfLiteTensor* model_output = interpreter.output(0);
  printf("Details of output tensors 0 :\n");
  printTensorDetails(model_output);

  printf("Output tensor values:\n");
  for (int d=0; d<10; ++d)
    printf("[%d] %f\n", d, model_output->data.f[d]);

  return 0;
}
