//
//  ConvolutionExample.m
//  OperationCluster
//
//  Created by Lun on 2017/6/15.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "ConvolutionExample.h"
#import <Accelerate/Accelerate.h>
#import "nnpack.h"

@implementation ConvolutionExample

// BNNS
- (void)BNNSConvWithInputSize:(const uint)inputSize
                 inputChannel:(const uint)inputChannel
                   outputSize:(const uint)outputSize
                outputChannel:(const uint)outputChannel
                   kernelSize:(const uint)kernelSize
                       stride:(const uint)stride
                      padding:(const uint)padding
                       weight:(const float *)weight
                         bias:(const float *)bias
                        input:(const float *)input
                       output:(float *)output {
    
    BNNSImageStackDescriptor input_desc;
    bzero(&input_desc,sizeof(input_desc));
    input_desc.width = inputSize;
    input_desc.height = inputSize;
    input_desc.channels = inputChannel;
    input_desc.row_stride = inputSize;
    input_desc.image_stride = inputSize * inputSize;
    input_desc.data_type = BNNSDataTypeFloat32;
    
    BNNSImageStackDescriptor output_desc;
    bzero(&output_desc,sizeof(output_desc));
    output_desc.width = outputSize;
    output_desc.height = outputSize;
    output_desc.channels = outputChannel;
    output_desc.row_stride = outputSize;
    output_desc.image_stride = outputSize * outputSize;
    output_desc.data_type = BNNSDataTypeFloat32;
    
    BNNSActivation relu;
    relu.function = BNNSActivationFunctionRectifiedLinear;
    relu.alpha = 0;
    relu.beta = 0;
    
    BNNSFilterParameters filter_params;
    bzero(&filter_params, sizeof(filter_params));
    
    BNNSConvolutionLayerParameters conv_params;
    bzero(&conv_params, sizeof(conv_params));
    conv_params.k_width = kernelSize;
    conv_params.k_height = kernelSize;
    conv_params.in_channels = inputChannel;
    conv_params.out_channels = outputChannel;
    conv_params.x_stride = stride;
    conv_params.y_stride = stride;
    conv_params.x_padding = padding;
    conv_params.y_padding = padding;
    conv_params.weights.data = weight;
    conv_params.weights.data_type = BNNSDataTypeFloat32;
    conv_params.bias.data = bias;
    conv_params.bias.data_type = BNNSDataTypeFloat32;
    conv_params.activation = relu;
    
    BNNSFilter conv = BNNSFilterCreateConvolutionLayer(&input_desc, &output_desc, &conv_params, &filter_params);
    
    BNNSFilterApply(conv, input, output);
}

// NNPACK
- (void)NNPACKConvWithInputSize:(const uint)inputSize
                   inputChannel:(const uint)inputChannel
                     outputSize:(const uint)outputSize
                  outputChannel:(const uint)outputChannel
                     kernelSize:(const uint)kernelSize
                         stride:(const uint)stride
                        padding:(const uint)padding
                         weight:(const float *)weight
                           bias:(const float *)bias
                          input:(const float *)input
                         output:(float *)output {
    
    enum nnp_status init_status = nnp_initialize();
    if (init_status != nnp_status_success) {
        fprintf(stderr, "NNPACK initialization failed: error code %d\n", init_status);
        exit(EXIT_FAILURE);
    }
    pthreadpool_t threadpool = pthreadpool_create(0);
    
    struct nnp_profile computation_profile;
    
    nnp_convolution_inference(nnp_convolution_algorithm_implicit_gemm,
                              nnp_convolution_transform_strategy_compute,
                              inputChannel,
                              outputChannel,
                              (struct nnp_size) {inputSize, inputSize},
                              (struct nnp_padding) {padding, padding, padding, padding},
                              (struct nnp_size) {kernelSize, kernelSize},
                              (struct nnp_size) {stride, stride},
                              input,
                              weight,
                              bias,
                              output,
                              NULL,
                              NULL,
                              nnp_activation_relu,
                              NULL,
                              threadpool,
                              &computation_profile);
}

@end
