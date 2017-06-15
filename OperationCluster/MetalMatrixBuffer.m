//
//  MetalMatrixBuffer.m
//  GPUMatrix
//
//  Created by Lun on 2017/6/14.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "MetalMatrixBuffer.h"

// A utility class to encapsulate instantiation of Metal matrix buffer.
// By the virtue of this encapsulation, and after the buffers are added
// to a mutable array of buffers, all buffers are kept alive in the
// matrix multiplication object's life-cycle.
@implementation MetalMatrixBuffer {
@private
    size_t         _size;        // Buffer size in bytes
    void*          _baseAddr;    // Base address for buffers
    id<MTLBuffer>  _buffer;      // Buffer for matrices
    id<MTLDevice>  _device;      // Default Metal system device
}

- (MetalMatrixBuffer *) initWithDevice:(id<MTLDevice>)device
                                matrix:(float *)matrix
                                  size:(size_t)size {
    if (self = [super init]) {
        if (device) {
            _device = device;
            _size = [self _sizeAlign:size];
            
            if (!matrix) {
                _buffer = [_device newBufferWithLength:_size
                                               options:MTLResourceStorageModeShared];
            } else {
                _buffer = [_device newBufferWithBytesNoCopy:matrix
                                                     length:_size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
            }
            
            if (_buffer) {
                _baseAddr = [_buffer contents];
            } else {
                NSLog(@">> ERROR: Failed creating a Metal buffer!");
                
                if (matrix) {
                    _buffer = [_device newBufferWithBytes:matrix
                                                   length:_size
                                                  options:MTLResourceStorageModeShared];
                    
                    NSAssert(_buffer, @">> ERROR: Failed again!");
                    _baseAddr = [_buffer contents];
                }
            }
        } else {
            NSLog(@">> ERROR: Invalid default Metal system device!");
        }
    }
    
    return self;
}

- (size_t) _sizeAlign:(size_t)size {
    return size % getpagesize() ? (size / getpagesize() + 1) * getpagesize(): size;
}

@end
