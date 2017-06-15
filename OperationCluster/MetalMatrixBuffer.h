//
//  MetalMatrixBuffer.h
//  GPUMatrix
//
//  Created by Lun on 2017/6/14.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// A utility class to encapsulate instantiation of Metal matrix buffer.
// By the virtue of this encapsulation, and after the buffers are added
// to a mutable array of buffers, all buffers are kept alive in the
// matrix multiplication object's life-cycle.
@interface MetalMatrixBuffer : NSObject

typedef NS_ENUM (NSInteger,MetalMatrixBufferTypes) {
    eMTLMatBufferInA = 0,
    eMTLMatBufferInB,
    eMTLMatBufferOutA,
    eMTLMatBufferOutB,
    eMTLMatBufferMax
};

@property (nonatomic)           size_t         size;
@property (nonatomic)           void*          baseAddr;
@property (nonatomic)           id<MTLBuffer>  buffer;
@property (nonatomic, readonly) id<MTLDevice>  device;

- (MetalMatrixBuffer *) initWithDevice:(id<MTLDevice>)device
                                matrix:(float *)matrix
                                  size:(size_t)size;

@end
