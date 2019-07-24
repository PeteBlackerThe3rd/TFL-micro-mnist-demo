



MNIST_DEMO_SRCS := \
tensorflow/lite/experimental/micro/examples/mnist_demo/main.cc \
tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_dense_model.cc

MNIST_DEMO_HDRS := \
tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_dense_model.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/mnist_demo/*/Makefile.inc)

# Builds a standalone speech command recognizer binary.
$(eval $(call microlite_test,mnist_demo,\
$(MNIST_DEMO_SRCS),$(MNIST_DEMO_HDRS)))

