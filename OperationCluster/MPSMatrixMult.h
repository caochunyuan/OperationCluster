//
//  MPSMatrixMult.h
//  GPUMatrix
//
//  Created by Lun on 2017/3/20.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSMatrixMult : NSObject

void mps_gemm(const bool transA,
              const bool transB,
              const int M,
              const int N,
              const int K,
              const float alpha,
              const float* A,
              const float* B,
              const float beta,
              float* C);

@end
