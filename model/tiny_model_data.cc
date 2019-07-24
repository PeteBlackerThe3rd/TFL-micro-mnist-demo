
#include "tensorflow/lite/experimental/micro/examples/mnist_dense/model/tiny_model_data.h"

const unsigned char tiny_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
  0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x5c, 0x05, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xb0, 0x03, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x54, 0x4f, 0x43, 0x4f, 0x20, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74,
  0x65, 0x64, 0x2e, 0x00, 0x05, 0x00, 0x00, 0x00, 0x84, 0x03, 0x00, 0x00,
  0x54, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xf0, 0xfa, 0xff, 0xff, 0xc6, 0xff, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xcc, 0xfb, 0xff, 0xff, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x20, 0x03, 0x00, 0x00,
  0xb7, 0xe0, 0xe6, 0xbd, 0xd5, 0x90, 0x7b, 0xbd, 0xe6, 0x85, 0xb0, 0xbc,
  0x9d, 0x41, 0x03, 0xbe, 0xf4, 0xcb, 0x0b, 0xbd, 0xfb, 0x87, 0xc5, 0x3d,
  0xf8, 0x81, 0x3d, 0x3e, 0xfe, 0x55, 0x03, 0xbe, 0x22, 0x36, 0xec, 0xbd,
  0xa0, 0x90, 0x40, 0xbd, 0xda, 0x73, 0xba, 0xbc, 0x3b, 0x67, 0x69, 0xbd,
  0x4f, 0x15, 0xb3, 0xbd, 0xd7, 0x1e, 0x78, 0x3b, 0x1f, 0x28, 0xbe, 0xbd,
  0x7a, 0xdf, 0x78, 0x3c, 0xe3, 0x5b, 0x5a, 0xbd, 0xbd, 0x0e, 0xe6, 0x3d,
  0x1b, 0x21, 0xb1, 0xbd, 0x01, 0x7c, 0x21, 0x3e, 0x48, 0x6a, 0xea, 0xbc,
  0x23, 0xad, 0x50, 0xbd, 0xaa, 0xe8, 0xd2, 0xbd, 0x63, 0x97, 0xb4, 0xbd,
  0x22, 0x53, 0xe2, 0x3d, 0x52, 0x7f, 0x39, 0x3e, 0xe5, 0xa1, 0x4a, 0x3e,
  0x57, 0xbf, 0xb1, 0xbc, 0x7b, 0x59, 0x7e, 0xbd, 0xd5, 0xdc, 0x27, 0x3d,
  0x63, 0xee, 0x1b, 0xbd, 0xa3, 0xdf, 0xf6, 0x3d, 0x5b, 0xa1, 0x45, 0xbd,
  0x57, 0x0e, 0x11, 0xbe, 0x43, 0xa4, 0x08, 0xbe, 0x3c, 0x66, 0x9f, 0xbd,
  0x24, 0x27, 0xcc, 0x3c, 0x58, 0x36, 0xcf, 0xbc, 0x12, 0x50, 0x1d, 0xbd,
  0xaf, 0xe8, 0xde, 0x3d, 0x4b, 0x4a, 0xe9, 0x3d, 0x77, 0x02, 0xb2, 0x3c,
  0xe9, 0x2b, 0x90, 0xbd, 0x8c, 0xd9, 0x80, 0x3d, 0x88, 0x72, 0x94, 0xbb,
  0x0f, 0x6b, 0xa8, 0xbd, 0x77, 0xd2, 0x5c, 0x3d, 0x9b, 0xc1, 0xad, 0xbd,
  0x20, 0x52, 0xb8, 0x3d, 0xd5, 0x17, 0x00, 0x3e, 0x89, 0x84, 0x39, 0xbd,
  0xdf, 0x44, 0x13, 0x3c, 0xb9, 0x5a, 0x45, 0x3e, 0xcd, 0xe5, 0xe9, 0xbd,
  0x9f, 0xd4, 0xec, 0xbc, 0xcd, 0x6f, 0x21, 0xbd, 0xf7, 0x26, 0xdf, 0xbd,
  0x40, 0xde, 0xef, 0x3d, 0x67, 0x74, 0x3a, 0x3e, 0xe1, 0xfb, 0x81, 0xbc,
  0xf7, 0xdc, 0x6f, 0x3d, 0xd4, 0x9c, 0x05, 0x3e, 0xe8, 0xc4, 0xbd, 0xbb,
  0xba, 0x43, 0x22, 0x3e, 0x8b, 0xb9, 0x73, 0x3c, 0x28, 0x0e, 0xf5, 0x3c,
  0xe2, 0xcd, 0xf7, 0x3d, 0x4b, 0xee, 0xd5, 0x3b, 0x28, 0xb4, 0x08, 0xbe,
  0x9a, 0xb3, 0x1c, 0x3c, 0xf8, 0x43, 0x24, 0xbd, 0xf7, 0xa4, 0xfc, 0x3d,
  0x2b, 0xb9, 0x8e, 0x3c, 0x5a, 0x4a, 0xdb, 0xbd, 0xef, 0x06, 0x19, 0xbe,
  0x2c, 0x00, 0x3b, 0x3e, 0x3c, 0x56, 0x40, 0xbd, 0xa7, 0x0d, 0x2e, 0xbd,
  0x3b, 0xd7, 0x55, 0xbd, 0x0e, 0xdb, 0x0a, 0xbe, 0xd0, 0xb7, 0xf9, 0x3d,
  0xed, 0xa3, 0xa2, 0xbd, 0x22, 0xd4, 0xe7, 0xbc, 0xa5, 0x34, 0x55, 0x3d,
  0x7f, 0xe0, 0x06, 0xbe, 0x7c, 0x7a, 0x0c, 0x3d, 0xe7, 0x5b, 0xe3, 0xbd,
  0xaf, 0xc0, 0xbb, 0xbd, 0x66, 0x16, 0x4b, 0xbd, 0x8e, 0x8d, 0x95, 0xbd,
  0xe7, 0x78, 0x9e, 0x3c, 0xd9, 0x9c, 0xa2, 0xba, 0xb9, 0x7f, 0x81, 0x3d,
  0x97, 0x7f, 0x55, 0xbd, 0x0e, 0xd8, 0x1b, 0xbe, 0x61, 0xb9, 0x9b, 0xbd,
  0x78, 0x5f, 0x24, 0xbe, 0x59, 0x9d, 0x25, 0x3d, 0x8a, 0xa2, 0xef, 0x3c,
  0xf2, 0x1d, 0xd2, 0x3d, 0xac, 0x56, 0x30, 0x3d, 0xa7, 0xe5, 0x28, 0xbe,
  0x33, 0x9f, 0x9f, 0xbd, 0xc9, 0x13, 0x43, 0xbd, 0xef, 0x4d, 0xf1, 0x3d,
  0x6b, 0xc3, 0x1e, 0xbe, 0x8b, 0x50, 0xb1, 0xbd, 0xf9, 0x5c, 0x01, 0x3e,
  0x41, 0x2c, 0x47, 0x3e, 0x71, 0xf0, 0x9d, 0xbd, 0x14, 0x8d, 0x88, 0xbd,
  0x70, 0x6f, 0xeb, 0x3d, 0x51, 0x28, 0x42, 0xbe, 0xad, 0xa7, 0x43, 0xbe,
  0xc6, 0x18, 0x1c, 0xbe, 0x40, 0xd9, 0x5c, 0x3c, 0x7f, 0xfb, 0x7c, 0xbc,
  0x3a, 0xe2, 0x95, 0x3d, 0x50, 0xc6, 0x42, 0xbc, 0xc8, 0x20, 0xe9, 0xbd,
  0x83, 0x33, 0x27, 0xbc, 0x97, 0x12, 0x13, 0x3e, 0x03, 0xa4, 0xba, 0x3d,
  0xa2, 0x33, 0x80, 0xbc, 0xeb, 0x63, 0xcc, 0x3c, 0x0f, 0xe2, 0x84, 0xbd,
  0x08, 0xf4, 0x00, 0x3d, 0x21, 0xd0, 0x2b, 0x3e, 0xb6, 0x1a, 0x3a, 0xbd,
  0x6e, 0xb5, 0x09, 0xbd, 0x12, 0xcb, 0x4d, 0xbd, 0x29, 0x76, 0xb7, 0x3d,
  0xd4, 0xa5, 0x45, 0x3c, 0x18, 0x4d, 0xa7, 0xbd, 0x5b, 0xce, 0xcb, 0xbd,
  0x87, 0x92, 0x38, 0x3d, 0x4a, 0x2b, 0x5d, 0x3d, 0x5f, 0x96, 0x26, 0x3e,
  0x41, 0x90, 0xb5, 0xbd, 0xea, 0xc4, 0xf4, 0xbc, 0x92, 0xe2, 0xd9, 0xbd,
  0x1c, 0x37, 0x22, 0x3d, 0xee, 0x91, 0xa3, 0x3d, 0xa7, 0x9a, 0x84, 0x3c,
  0x3c, 0xb4, 0xb2, 0xbd, 0x56, 0x84, 0x28, 0xbd, 0x58, 0x61, 0x3f, 0x3d,
  0x3b, 0x19, 0xf0, 0xbd, 0xae, 0xe8, 0x06, 0xbd, 0x27, 0x65, 0x17, 0xbe,
  0xcf, 0x80, 0x23, 0xbd, 0x26, 0x25, 0x9d, 0x3c, 0x47, 0x29, 0x15, 0x3d,
  0x70, 0xa6, 0x90, 0xbd, 0xc8, 0xd1, 0x16, 0xbd, 0x5e, 0x53, 0x2a, 0x3e,
  0x93, 0x43, 0xc3, 0xbd, 0x76, 0xbc, 0xc6, 0x3d, 0xef, 0xcd, 0x5f, 0x3d,
  0x7c, 0x30, 0x2e, 0xbc, 0x0b, 0x0c, 0x50, 0x3d, 0x4d, 0xa6, 0x15, 0x3d,
  0xc6, 0x54, 0x9d, 0xbc, 0x11, 0xef, 0x2b, 0xbd, 0x67, 0x22, 0xd3, 0x3d,
  0x42, 0xf4, 0x65, 0xbc, 0x75, 0x57, 0x93, 0xbb, 0x1f, 0x0d, 0x3d, 0xbe,
  0x5a, 0xc0, 0x46, 0xbd, 0xb3, 0x3e, 0xf0, 0xbc, 0x99, 0x43, 0x1e, 0xbe,
  0xf1, 0x94, 0x0f, 0x3c, 0x6d, 0x31, 0x63, 0xbd, 0xa8, 0xd1, 0xd3, 0xbd,
  0xe7, 0xc7, 0x17, 0x3d, 0xb1, 0x99, 0x0a, 0xbe, 0x5f, 0x80, 0x0f, 0x3d,
  0x94, 0x67, 0xab, 0x3d, 0x02, 0x58, 0x73, 0x3d, 0x0d, 0x00, 0xa5, 0x3d,
  0x88, 0x36, 0xd1, 0xbd, 0x08, 0x81, 0xb4, 0xbc, 0x18, 0x18, 0x11, 0x3d,
  0x62, 0x4f, 0x6f, 0x3d, 0xd4, 0x07, 0xc9, 0xba, 0xe7, 0x27, 0xda, 0x3d,
  0xba, 0x2d, 0x97, 0xbd, 0x40, 0x2c, 0x7b, 0x3d, 0xa9, 0x4e, 0x0d, 0x3e,
  0x5d, 0xd5, 0x09, 0xbd, 0x08, 0x9d, 0x6d, 0xbb, 0x5f, 0x8a, 0x3c, 0x3d,
  0x38, 0x1a, 0x8c, 0x3d, 0xbf, 0x31, 0x00, 0xbe, 0xb5, 0x38, 0x6f, 0xbd,
  0x57, 0xf0, 0xb4, 0xbb, 0x97, 0xaa, 0x24, 0x3d, 0x67, 0x27, 0xf4, 0x3c,
  0x13, 0x6a, 0x0d, 0xbd, 0x27, 0xb9, 0x8b, 0xbb, 0x60, 0xfe, 0xff, 0xff,
  0x0c, 0x00, 0x14, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x30, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x98, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
  0xd4, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x42, 0xff, 0xff, 0xff,
  0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x69, 0x6e, 0x70, 0x75,
  0x74, 0x2f, 0x78, 0x2d, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x0c, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x7f, 0x43, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x9a, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x4d, 0x61, 0x74, 0x4d,
  0x75, 0x6c, 0x5f, 0x62, 0x69, 0x61, 0x73, 0x00, 0x2c, 0xff, 0xff, 0xff,
  0xca, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x00, 0x04, 0x00, 0x06, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x14, 0x00, 0x04, 0x00,
  0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x28, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x56, 0x61, 0x72, 0x69,
  0x61, 0x62, 0x6c, 0x65, 0x2f, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x6f,
  0x73, 0x65, 0x00, 0x00, 0xac, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x14, 0x00, 0x18, 0x00, 0x00, 0x00, 0x08, 0x00,
  0x0c, 0x00, 0x07, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x10, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
  0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09
};
unsigned int tiny_tflite_len = 1428;
