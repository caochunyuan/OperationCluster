//
//  ViewController.m
//  OperationCluster
//
//  Created by Lun on 2017/6/13.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "ViewController.h"
#import <Accelerate/Accelerate.h>
#import "nnpackGemm.h"
#import "eigenGemmWrapper.h"

@interface ViewController () {
    NSMutableArray *time1Arr;
    NSMutableArray *time2Arr;
}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    time1Arr = [NSMutableArray new];
    time2Arr = [NSMutableArray new];
    
    for (int iter = 1; iter < 501; iter++) {
        
        int m = iter;   //(int)((float)arc4random() / UINT32_MAX * 500 + 1);
        int n = m;      //(int)((float)arc4random() / UINT32_MAX * 500 + 1);
        int k = m;      //(int)((float)arc4random() / UINT32_MAX * 500 + 1);
        float alpha = 1;//(float)arc4random() / UINT32_MAX;
        float beta = 1; //(float)arc4random() / UINT32_MAX;
        BOOL transA = NO;
        BOOL transB = NO;
        BOOL verify = NO;
        
        printf("m, n, k, alpha, beta are %d %d %d %f %f\n",m,n,k,alpha,beta);
        
        float *A = malloc(m * k * sizeof(float));
        float *B = malloc(k * n * sizeof(float));
        float *C = malloc(m * n * sizeof(float));
        float *D = malloc(m * n * sizeof(float));
        
        for (int i = 0; i < m * k; i++) {
            A[i] = (float)arc4random() / UINT32_MAX - 0.5;
        }
        
        for (int i = 0; i < k * n; i++) {
            B[i] = (float)arc4random() / UINT32_MAX - 0.5;
        }
        
        for (int i = 0; i < m * n; i++) {
            D[i] = (float)arc4random() / UINT32_MAX - 0.5;
        }
        
        NSDate *start = [NSDate date];
        
        memcpy(C, D, m * n * sizeof(float));
        start = [NSDate date];
        
        nnpack_gemm(nnpackGemmAuto, transA? nnpackTrans : nnpackNoTrans, transB? nnpackTrans : nnpackNoTrans, m, n, k, alpha, A, B, beta, C);
        
        float time1 = -[start timeIntervalSinceNow]*1000;
        [time1Arr addObject:@(time1)];
        
        float sum1 = 0.0;
        if (verify) {
            vDSP_sve(C, 1, &sum1, m*n);
        }
        
        memcpy(C, D, m * n * sizeof(float));
        start = [NSDate date];
        
        [eigenGemmWrapper gemmWithTransA:transA
                                  transB:transB
                                       M:m
                                       N:n
                                       K:k
                                   alpha:alpha
                                       A:A
                                       B:B
                                    beta:beta
                                       C:C];
        
        float time2 = -[start timeIntervalSinceNow]*1000;
        [time2Arr addObject:@(time2)];

        float sum2 = 0.0;
        if (verify) {
            vDSP_sve(C, 1, &sum2, m*n);
        }
        
        if (verify) {
            if (fabsf(sum1 - sum2) > 0.1) {
                printf("diff: %f", sum1 - sum2);
                assert(0);
            }
        }
        
//        printf("nnpack: %f   eigen: %f\n", time1, time2);
        
        free(A);
        free(B);
        free(C);
        free(D);
    }
    
    for (int i = 0; i < time1Arr.count; i++) {
        printf("%.2f ", ((NSNumber *)time1Arr[i]).floatValue);
    }
    
    printf("\n\n");
    
    for (int i = 0; i < time2Arr.count; i++) {
        printf("%.2f ", ((NSNumber *)time2Arr[i]).floatValue);
    }
    
    printf("\n\n");
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
