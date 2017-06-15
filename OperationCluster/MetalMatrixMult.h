/*
 Copyright (C) 2016 Apple Inc. All Rights Reserved.
 See LICENSE.txt for this sample’s licensing information
 
 Abstract:
 Utility class for performing matrix multiplication using a Metal compute kernel
 */

@interface MetalMatrixMult: NSObject
    
typedef NS_ENUM (NSInteger, METAL_TRANSPOSE) {
    MetalTrans = 101,
    MetalNoTrans
};

typedef NS_ENUM (NSInteger, METAL_ALGORITHM) {
    MetalGemm8x8 = 151,
    MetalGemm4x4
};

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
                const enum METAL_ALGORITHM algorithm);

@end
