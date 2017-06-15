/*
 Copyright (C) 2016 Apple Inc. All Rights Reserved.
 See LICENSE.txt for this sample’s licensing information
 
 Abstract:
 Utility class for performing matrix multiplication using a Metal compute kernel
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalMatrixMult.h"
#import "MetalMatrixBuffer.h"

struct MetalMatrixDim {
    bool trans_a, trans_b;
    uint16_t m, n, k, rmd_a, rmd_b;
    float alpha, beta;
};

typedef struct MetalMatrixDim   MetalMatrixDim;
typedef        MetalMatrixDim*  MetalMatrixDimRef;

static const uint32_t kSzMTLFloat  = sizeof(float);
static const uint32_t kSzMTLMatDim = sizeof(MetalMatrixDim);

// Utility class encapsulating Metal matrix multiplication compute
@implementation MetalMatrixMult {
@private
    // Metal assests for the compute kernel
    id<MTLDevice>                m_Device;
    id<MTLCommandQueue>          m_Queue;
    id<MTLComputePipelineState>  m_Kernel8x8;
    id<MTLComputePipelineState>  m_Kernel4x4;
    id<MTLCommandBuffer>         m_CmdBuffer;
    id<MTLComputeCommandEncoder> m_Encoder;
    
    // Buffers for matrices
    NSMutableArray *m_Buffers;
    
    // Number of rows in matrices A and C.
    uint16_t _m;
    
    // Number of columns in matrices B and C.
    uint16_t _n;
    
    // Number of columns in matrix A; number of rows in matrix B.
    uint16_t _k;
    
    // C = alpha * op(A) * op(B) + beta * op(C)
    float _alpha;
    float _beta;
    
    // Compute kernel threads
    MTLSize m_ThreadGroupSize;
    MTLSize m_ThreadGroups;
    
    // Dispatch Quueue
    dispatch_group_t  m_DGroup;
    dispatch_queue_t  m_DQueue;
}

static MetalMatrixMult *multiplicationHandler = nil;

+ (MetalMatrixMult *) callSingleton {
    @synchronized(self) {
        if (multiplicationHandler == nil) {
            multiplicationHandler = [[self alloc] init];
        }
    }
    return multiplicationHandler;
}

+ (id) allocWithZone:(NSZone *)zone {
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
        m_DQueue = dispatch_queue_create("com.dressplus.matrixmult.metal.main", DISPATCH_QUEUE_SERIAL);
        
        // Metal default system device
        m_Device = MTLCreateSystemDefaultDevice();
        NSAssert(m_Device, @">> ERROR: Failed creating a system default device!");
        
        // Command queue
        m_Queue = [m_Device newCommandQueue];
        NSAssert(m_Queue, @">> ERROR: Failed creating a command queue!");
        
        // Default library
        id <MTLLibrary> library = [m_Device newDefaultLibrary];
        NSAssert(library, @">> ERROR: Failed creating a library!");
        
//        NSError *libraryError = NULL;
//        NSString *libraryFile = [[NSBundle bundleForClass:[self class]] pathForResource:@"MetalMatrixMult" ofType:@"metallib"];
//        id <MTLLibrary> library = [m_Device newLibraryWithFile:libraryFile error:&libraryError];
//        NSAssert(library, @">> ERROR: Failed creating a library: %@", libraryError);
        
        // New compute kernel function
        id<MTLFunction> func8x8 = [library newFunctionWithName:@"MetalGemm8x8"];
        id<MTLFunction> func4x4 = [library newFunctionWithName:@"MetalGemm4x4"];
        NSAssert(func8x8 && func4x4, @">> ERROR: Failed creating a named function!");
        
        // Pipeline state, or the compute kernel
        m_Kernel8x8 = [m_Device newComputePipelineStateWithFunction:func8x8 error:nil];
        m_Kernel4x4 = [m_Device newComputePipelineStateWithFunction:func4x4 error:nil];
        NSAssert(m_Kernel8x8 && m_Kernel4x4, @">> ERROR: Failed creating a compute pipeline state!");
        
        // Create a mutable array for buffers
        m_Buffers = [[NSMutableArray alloc] initWithCapacity:eMTLMatBufferMax];
        NSAssert(m_Buffers, @">> ERROR: Failed creating a mutable array for Metal buffers!");
        
        for (int i = 0; i < eMTLMatBufferMax; i++) {
            m_Buffers[i] = [NSNull null];
        }
    }
    return self;
}

// Multiplication handler
- (void) _multiplyWithTransA:(const METAL_TRANSPOSE)transA
                      TransB:(const METAL_TRANSPOSE)transB
                           M:(const int)M
                           N:(const int)N
                           K:(const int)K
                       alpha:(const float)alpha
                           A:(const float *)A
                           B:(const float *)B
                        beta:(const float)beta
                           C:(float *)C
                   algorithm:(const METAL_ALGORITHM)algorithm
                  completion:(void (^)())completion {
    // The singleton would not handle more than one multiplication operation
    @synchronized(self) {
        // Reset parameters of matrices
        _m = M;
        _n = N;
        _k = K;
        _alpha = alpha;
        _beta = beta;
        
        // Set up buffers with the group and the queue
        [self _newBuffersWithA:(float *)A B:(float *)B C:C];
        [self _initDimsWithTransA:transA transB:transB algorithm:algorithm];
        
        // Set thread-group size
        m_ThreadGroupSize = MTLSizeMake(8, 8, 1);
        
        // Set thread-group parameters
        m_ThreadGroups.depth  = 1;
        
        // Thread group size based on row count of matrix A
        NSUInteger width;
        switch (algorithm) {
            case MetalGemm8x8:
                width = _m % 8 ? (_m + 8) / 8 : _m / 8;
                break;
            case MetalGemm4x4:
                width = _m % 4 ? (_m + 4) / 4 : _m / 4;
                break;
            default:
                break;
        }
        
        m_ThreadGroups.width = (width % m_ThreadGroupSize.width) ?
        (width + m_ThreadGroupSize.width) / m_ThreadGroupSize.width :
        width / m_ThreadGroupSize.width;
        
        // Thread group size based on column count of matrix B
        NSUInteger height;
        switch (algorithm) {
            case MetalGemm8x8:
                height = _n % 8 ? (_n + 8) / 8 : _n / 8;
                break;
            case MetalGemm4x4:
                height = _n % 4 ? (_n + 4) / 4 : _n / 4;
                break;
            default:
                break;
        }
        
        m_ThreadGroups.height = (height % m_ThreadGroupSize.height) ?
        (height + m_ThreadGroupSize.height) / m_ThreadGroupSize.height :
        height / m_ThreadGroupSize.height;
        
        dispatch_group_wait(m_DGroup, DISPATCH_TIME_FOREVER);
        
        // Do multiplication after initializations
        if ([self _encodeWithAlgorithm:algorithm]) {
            [self _dispatch];
            [self _finish];
            
            completion();
        } else {
            NSLog(@">> ERROR: Failed in encoding!");
        }
    }
}

- (void) _newBuffersWithA:(float *)A
                        B:(float *)B
                        C:(float *)C {
    [self _fillBuffer:eMTLMatBufferInA withMatrix:A length:_m * _k];
    [self _fillBuffer:eMTLMatBufferInB withMatrix:B length:_k * _n];
    [self _fillBuffer:eMTLMatBufferOutA withMatrix:C length:_m * _n];
}
    
- (void) _fillBuffer:(const uint32_t)index
          withMatrix:(float *)matrix
              length:(size_t)length {
    // Enter the group for creating a matrix buffer
    dispatch_group_enter(m_DGroup);
    
    // Create an input matrix buffer
    dispatch_group_async(m_DGroup, m_DQueue, ^{
        m_Buffers[index] = [[MetalMatrixBuffer alloc] initWithDevice:m_Device
                                                              matrix:matrix
                                                                size:length * kSzMTLFloat];
        
        // Leave the group for creating a new input matrix buffer
        dispatch_group_leave(m_DGroup);
    });
}

- (void) _initDimsWithTransA:(const METAL_TRANSPOSE)transA
                      transB:(const METAL_TRANSPOSE)transB
                   algorithm:(const METAL_ALGORITHM)algorithm {
    // Set the buffer parameters for matrix dimensions
    MetalMatrixBuffer* pOutBufferB = [[MetalMatrixBuffer alloc] initWithDevice:m_Device
                                                                        matrix:nil
                                                                          size:kSzMTLMatDim];
    
    if (pOutBufferB) {
        MetalMatrixDimRef pOutMatrixDims = (MetalMatrixDimRef)pOutBufferB.baseAddr;
        if (pOutMatrixDims) {
            pOutMatrixDims->m = _m;
            pOutMatrixDims->n = _n;
            pOutMatrixDims->k = _k;
            
            switch (algorithm) {
                case MetalGemm8x8:
                    pOutMatrixDims->rmd_a = _m % 8;
                    pOutMatrixDims->rmd_b = _n % 8;
                    break;
                case MetalGemm4x4:
                    pOutMatrixDims->rmd_a = _m % 4;
                    pOutMatrixDims->rmd_b = _n % 4;
                    break;
                default:
                    break;
            }
            
            pOutMatrixDims->trans_a = transA == MetalTrans;
            pOutMatrixDims->trans_b = transB == MetalTrans;
            
            pOutMatrixDims->alpha = _alpha;
            pOutMatrixDims->beta = _beta;
        }
        
        m_Buffers[eMTLMatBufferOutB] = pOutBufferB;
    }
}

// Create a command buffer, encode and set buffers
- (BOOL) _encodeWithAlgorithm:(METAL_ALGORITHM)algorithm {
    // Acquire a command buffer for compute
    m_CmdBuffer = [m_Queue commandBuffer];
    
    if (!m_CmdBuffer) {
        // Wait until matrix initializations are complete
        dispatch_group_wait(m_DGroup, DISPATCH_TIME_FOREVER);
        
        NSLog(@">> ERROR: Failed acquiring a command buffer!");
        
        return NO;
    }
    
    // Acquire a compute command encoder
    m_Encoder = [m_CmdBuffer computeCommandEncoder];
    
    if(!m_Encoder) {
        // Wait until matrix initializations are complete
        dispatch_group_wait(m_DGroup, DISPATCH_TIME_FOREVER);
        
        NSLog(@">> ERROR: Failed acquiring a compute command encoder!");
        
        return NO;
    }
    
    // Set the encoder with the buffers
    switch (algorithm) {
        case MetalGemm8x8:
            [m_Encoder setComputePipelineState:m_Kernel8x8];
            break;
        case MetalGemm4x4:
            [m_Encoder setComputePipelineState:m_Kernel4x4];
            break;
        default:
            break;
    }
    
    // Wait until matrix initializations are complete
    dispatch_group_wait(m_DGroup, DISPATCH_TIME_FOREVER);
    
    // Submit input matrix buffer A to the encoder
    [self _encodeBufferAtIndex:eMTLMatBufferInA];
    
    // Submit input matrix buffer B to the encoder
    [self _encodeBufferAtIndex:eMTLMatBufferInB];
    
    // Submit output matrix buffer C to the encoder
    [self _encodeBufferAtIndex:eMTLMatBufferOutA];
    
    // Submit buffer for output matrix dimensions to the encoder
    [self _encodeBufferAtIndex:eMTLMatBufferOutB];
    
    // Wait until the encoding is complete
    dispatch_group_wait(m_DGroup, DISPATCH_TIME_FOREVER);
    
    return YES;
}

// Submit a specific buffer to our encoder
- (void) _encodeBufferAtIndex:(const uint32_t)idx {
    MetalMatrixBuffer* pBuffer = m_Buffers[idx];
    
    if (pBuffer) {
        // Dispatch group for input matrix A
        dispatch_group_enter(m_DGroup);
        
        // Submit buffer for input matrix A to the encoder
        dispatch_group_async(m_DGroup, m_DQueue, ^{
            [m_Encoder setBuffer:pBuffer.buffer
                          offset:0
                         atIndex:idx];
            
            dispatch_group_leave(m_DGroup);
        });
    }
}

// Dispatch the compute kernel for matrix multiplication
- (void) _dispatch {
    [m_Encoder dispatchThreadgroups:m_ThreadGroups
              threadsPerThreadgroup:m_ThreadGroupSize];
}

// Wait until the matrix computation is complete
- (void) _finish {
    [m_Encoder endEncoding];
    [m_CmdBuffer commit];
    [m_CmdBuffer waitUntilCompleted];
}

// Interface for C = alpha * op(A) * op(B) + beta * op(C)
void metal_gemm(const enum METAL_TRANSPOSE transA,
                const enum METAL_TRANSPOSE transB,
                const int M,
                const int N,
                const int K,
                const float alpha,
                const float* A,
                const float* B,
                const float beta,
                float* C,
                const enum METAL_ALGORITHM algorithm) {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [[MetalMatrixMult callSingleton] _multiplyWithTransA:transA
                                                      TransB:transB
                                                           M:M
                                                           N:N
                                                           K:K
                                                       alpha:alpha
                                                           A:A
                                                           B:B
                                                        beta:beta
                                                           C:C
                                                   algorithm:algorithm
                                                  completion:^() {
                                                      dispatch_semaphore_signal(semaphore);
                                                  }];
    });
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
}

@end
