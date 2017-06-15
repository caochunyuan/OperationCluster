//
//  MPSMatrixMult.m
//  GPUMatrix
//
//  Created by Lun on 2017/3/20.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "MPSMatrixMult.h"

static const uint32_t kSzMPSFloat  = sizeof(float);

typedef NS_ENUM (NSInteger,MPSMatrixBufferTypes) {
    eMPSMatBufferInA = 0,
    eMPSMatBufferInB,
    eMPSMatBufferOut,
    eMPSMatBufferMax
};

@implementation MPSMatrixMult {
    // Metal assests for the compute kernel
    id<MTLDevice>         m_Device;
    id<MTLCommandQueue>   m_Queue;
    id <MTLCommandBuffer> m_CmdBuffer;
    
    // Dispatch Quueue
    dispatch_group_t  m_DGroup;
    dispatch_queue_t  m_DQueue;
    
    // Matrices in the MPS format
    NSMutableArray *m_Matrices;
}

static MPSMatrixMult *multiplicationHandler = nil;

+ (MPSMatrixMult *) callSingleton {
    @synchronized(self) {
        if (multiplicationHandler == nil) {
            multiplicationHandler = [[self alloc] init];
        }
    }
    return multiplicationHandler;
}

+ (id) allocWithZone:(NSZone *)zone{
    @synchronized(self) {
        if (multiplicationHandler == nil) {
            multiplicationHandler = [super allocWithZone:zone];
            return multiplicationHandler;
        }
    }
    return nil;
}

// Parameters initialized here should not be recreated during the lifecycle
- (id) init {
    if (self = [super init]) {
        // Dispatch group for  Metal buffer initializations
        m_DGroup = dispatch_group_create();
        
        // Dispatch queue for Metal buffers
        m_DQueue = dispatch_queue_create("com.dressplus.matrixmult.mps.main", DISPATCH_QUEUE_SERIAL);
        
        // Metal default system device
        m_Device = MTLCreateSystemDefaultDevice();
        NSAssert(m_Device, @">> ERROR: Failed creating a system default device!");
        
        // Command queue
        m_Queue = [m_Device newCommandQueue];
        NSAssert(m_Queue, @">> ERROR: Failed creating a command queue!");
        
        // Create a mutable array for matrices
        m_Matrices = [[NSMutableArray alloc] initWithCapacity:eMPSMatBufferMax];
        NSAssert(m_Matrices, @">> ERROR: Failed creating a mutable array for MPS matrices!");
        
        for (int i = 0; i < eMPSMatBufferMax; i++) {
            m_Matrices[i] = [NSNull null];
        }
    }
    
    return self;
}

- (void) _multiplyWithTransA:(const BOOL)transA
                      TransB:(const BOOL)transB
                           M:(const int)m
                           N:(const int)n
                           K:(const int)k
                       alpha:(const float)alpha
                           A:(const float *)A
                           B:(const float *)B
                        beta:(const float)beta
                           C:(float *)C
                  completion:(void (^)())completion {
    // The singleton would not handle more than one multiplication operation
    @synchronized(self) {
        // Set buffers, descriptors and matrices
        [self _setMatrix:(float *)A row:m column:k atIndex:eMPSMatBufferInA];
        [self _setMatrix:(float *)B row:k column:n atIndex:eMPSMatBufferInB];
        [self _setMatrix:C row:m column:n atIndex:eMPSMatBufferOut];
        
        dispatch_group_wait(m_DGroup, DISPATCH_TIME_FOREVER);
        
        if ([self _encodeWithKernel:[[MPSMatrixMultiplication alloc] initWithDevice:m_Device
                                                                      transposeLeft:transA
                                                                     transposeRight:transB
                                                                         resultRows:m
                                                                      resultColumns:n
                                                                    interiorColumns:k
                                                                              alpha:alpha
                                                                               beta:beta]]) {
            [m_CmdBuffer commit];
            [m_CmdBuffer waitUntilCompleted];
            
            completion();
        } else {
            NSLog(@">> ERROR: Failed in encoding!");
        }
    }
}

- (void) _setMatrix:(float *)mat
                row:(size_t)row
             column:(size_t)col
            atIndex:(const uint32_t)idx {
    // Enter the group for creating a matrix buffer
    dispatch_group_enter(m_DGroup);
    
    // Create an input matrix buffer
    dispatch_group_async(m_DGroup, m_DQueue, ^{
        
        size_t size = [self _sizeAlign:row * col * kSzMPSFloat];
        
        id<MTLBuffer> buffer = [m_Device newBufferWithBytesNoCopy:mat
                                                           length:size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        
        if (!buffer) {
            NSLog(@">> ERROR: Failed creating a Metal buffer. Now try to allocate a new one.");
            
            buffer = [m_Device newBufferWithBytes:mat
                                           length:size
                                          options:MTLResourceStorageModeShared];
            
            NSAssert(buffer, @">> ERROR: Failed again!");
        }
        
        MPSMatrixDescriptor * matrixDescriptor = [MPSMatrixDescriptor matrixDescriptorWithDimensions:row columns:col rowBytes:col * kSzMPSFloat dataType:MPSDataTypeFloat32];
        m_Matrices[idx] = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:matrixDescriptor];
        
        // Leave the group for creating a new input matrix buffer
        dispatch_group_leave(m_DGroup);
    });
}

- (BOOL) _encodeWithKernel:(MPSMatrixMultiplication *)kernel {
    // Acquire a command buffer for compute
    m_CmdBuffer = [m_Queue commandBuffer];
    
    if (!m_CmdBuffer) {
        // Wait until matrix initializations are complete
        dispatch_group_wait(m_DGroup, DISPATCH_TIME_FOREVER);
        
        NSLog(@">> ERROR: Failed acquiring a command buffer!");
        
        return NO;
    }
    
    [kernel encodeToCommandBuffer:m_CmdBuffer
                       leftMatrix:(MPSMatrix *)m_Matrices[eMPSMatBufferInA]
                      rightMatrix:(MPSMatrix *)m_Matrices[eMPSMatBufferInB]
                     resultMatrix:(MPSMatrix *)m_Matrices[eMPSMatBufferOut]];
    
    return YES;
}

- (size_t) _sizeAlign:(size_t)size {
    return size % getpagesize() ? (size / getpagesize() + 1) * getpagesize(): size;
}

void mps_gemm(const bool transA,
              const bool transB,
              const int M,
              const int N,
              const int K,
              const float alpha,
              const float* A,
              const float* B,
              const float beta,
              float* C) {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [[MPSMatrixMult callSingleton] _multiplyWithTransA:transA
                                                    TransB:transB
                                                         M:M
                                                         N:N
                                                         K:K
                                                     alpha:alpha
                                                         A:A
                                                         B:B
                                                      beta:beta
                                                         C:C
                                                completion:^() {
                                                    dispatch_semaphore_signal(semaphore);
                                                }];
    });
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
}

@end
