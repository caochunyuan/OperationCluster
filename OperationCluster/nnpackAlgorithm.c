//
//  nnpackAlgorithm.c
//  GeneralNet
//
//  Created by Lun on 2017/6/26.
//  Copyright © 2017年 Lun. All rights reserved.
//

#include <arm_neon.h>
#include "nnpackAlgorithm.h"

static inline float32x4_t vmuladdq_f32(float32x4_t c, float32x4_t a, float32x4_t b)
{
#if defined(__aarch64__)
    return vfmaq_f32(c, a, b);
#else
    return vmlaq_f32(c, a, b);
#endif
}

static inline float32x4_t vld1q_f32_aligned(const float* address) {
    return vld1q_f32((const float*) __builtin_assume_aligned(address, sizeof(float32x4_t)));
}

// modified from https://github.com/Maratyszcza/NNPACK/blob/e42421c248d746c92e655ec47e2c0fa4f9fc8e8c/src/neon/blas/sgemm.c/#L8
void nnp_sgemm_only_4x12(size_t k,
                         size_t update,
                         size_t output_row,
                         size_t output_col,
                         size_t reduction_size,
                         const bool trans_a,
                         const bool trans_b,
                         const float alpha,
                         const float beta,
                         const float *a,
                         const float *b,
                         float *c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f), vc02 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f), vc12 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f), vc22 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f), vc32 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t va;
        if (trans_a) {
            va = vld1q_f32_aligned(a);
            a += output_row;
        } else {
            va = (float32x4_t) {
                *(a + reduction_size * 0),
                *(a + reduction_size * 1),
                *(a + reduction_size * 2),
                *(a + reduction_size * 3),
            };
            a += 1;
        }
        
        float32x4_t vb0, vb1, vb2;
        if (trans_b) {
            vb0 = (float32x4_t) {
                *(b + reduction_size * 0),
                *(b + reduction_size * 1),
                *(b + reduction_size * 2),
                *(b + reduction_size * 3),
            };
            vb1 = (float32x4_t) {
                *(b + reduction_size * 4),
                *(b + reduction_size * 5),
                *(b + reduction_size * 6),
                *(b + reduction_size * 7),
            };
            vb2 = (float32x4_t) {
                *(b + reduction_size * 8),
                *(b + reduction_size * 9),
                *(b + reduction_size * 10),
                *(b + reduction_size * 11),
            };
            b += 1;
        } else {
            vb0 = vld1q_f32_aligned(b + 0);
            vb1 = vld1q_f32_aligned(b + 4);
            vb2 = vld1q_f32_aligned(b + 8);
            b += output_col;
        }
        
#if defined(__aarch64__)
        vc00 = vfmaq_lane_f32(vc00, vb0, vget_low_f32(va),  0);
        vc10 = vfmaq_lane_f32(vc10, vb0, vget_low_f32(va),  1);
        vc20 = vfmaq_lane_f32(vc20, vb0, vget_high_f32(va), 0);
        vc30 = vfmaq_lane_f32(vc30, vb0, vget_high_f32(va), 1);
        vc01 = vfmaq_lane_f32(vc01, vb1, vget_low_f32(va),  0);
        vc11 = vfmaq_lane_f32(vc11, vb1, vget_low_f32(va),  1);
        vc21 = vfmaq_lane_f32(vc21, vb1, vget_high_f32(va), 0);
        vc31 = vfmaq_lane_f32(vc31, vb1, vget_high_f32(va), 1);
        vc02 = vfmaq_lane_f32(vc02, vb2, vget_low_f32(va),  0);
        vc12 = vfmaq_lane_f32(vc12, vb2, vget_low_f32(va),  1);
        vc22 = vfmaq_lane_f32(vc22, vb2, vget_high_f32(va), 0);
        vc32 = vfmaq_lane_f32(vc32, vb2, vget_high_f32(va), 1);
#else
        vc00 = vmlaq_lane_f32(vc00, vb0, vget_low_f32(va),  0);
        vc10 = vmlaq_lane_f32(vc10, vb0, vget_low_f32(va),  1);
        vc20 = vmlaq_lane_f32(vc20, vb0, vget_high_f32(va), 0);
        vc30 = vmlaq_lane_f32(vc30, vb0, vget_high_f32(va), 1);
        vc01 = vmlaq_lane_f32(vc01, vb1, vget_low_f32(va),  0);
        vc11 = vmlaq_lane_f32(vc11, vb1, vget_low_f32(va),  1);
        vc21 = vmlaq_lane_f32(vc21, vb1, vget_high_f32(va), 0);
        vc31 = vmlaq_lane_f32(vc31, vb1, vget_high_f32(va), 1);
        vc02 = vmlaq_lane_f32(vc02, vb2, vget_low_f32(va),  0);
        vc12 = vmlaq_lane_f32(vc12, vb2, vget_low_f32(va),  1);
        vc22 = vmlaq_lane_f32(vc22, vb2, vget_high_f32(va), 0);
        vc32 = vmlaq_lane_f32(vc32, vb2, vget_high_f32(va), 1);
#endif
    } while (--k);
    
    // c = alpha * a * b + beta * c
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    
    if (update) {
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc00, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc01, alpha_v)));
        vst1q_f32(c + 8, vaddq_f32(vld1q_f32_aligned(c + 8), vmulq_f32(vc02, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc10, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc11, alpha_v)));
        vst1q_f32(c + 8, vaddq_f32(vld1q_f32_aligned(c + 8), vmulq_f32(vc12, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc20, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc21, alpha_v)));
        vst1q_f32(c + 8, vaddq_f32(vld1q_f32_aligned(c + 8), vmulq_f32(vc22, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc30, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc31, alpha_v)));
        vst1q_f32(c + 8, vaddq_f32(vld1q_f32_aligned(c + 8), vmulq_f32(vc32, alpha_v)));
    } else {
        float32x4_t beta_v = vdupq_n_f32(beta);
        
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc00, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc01, alpha_v)));
        vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 8), beta_v), vmulq_f32(vc02, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc10, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc11, alpha_v)));
        vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 8), beta_v), vmulq_f32(vc12, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc20, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc21, alpha_v)));
        vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 8), beta_v), vmulq_f32(vc22, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc30, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc31, alpha_v)));
        vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 8), beta_v), vmulq_f32(vc32, alpha_v)));
    }
}

// modified from https://github.com/Maratyszcza/NNPACK/blob/e42421c248d746c92e655ec47e2c0fa4f9fc8e8c/src/neon/blas/sgemm.c/#L86
void nnp_sgemm_upto_4x12(size_t mr,
                         size_t nr,
                         size_t k,
                         size_t update,
                         size_t output_row,
                         size_t output_col,
                         size_t reduction_size,
                         const bool trans_a,
                         const bool trans_b,
                         const float alpha,
                         const float beta,
                         const float *a,
                         const float *b,
                         float *c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f), vc02 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f), vc12 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f), vc22 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f), vc32 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t vb0 = vdupq_n_f32(0.0f), vb1 = vdupq_n_f32(0.0f), vb2 = vdupq_n_f32(0.0f);
        
        if (trans_b) {
            if (nr >= 4) {
                vb0 = (float32x4_t) {
                    *(b + reduction_size * 0),
                    *(b + reduction_size * 1),
                    *(b + reduction_size * 2),
                    *(b + reduction_size * 3)
                };
                if (nr != 4) {
                    if (nr >= 8) {
                        vb1 = (float32x4_t) {
                            *(b + reduction_size * 4),
                            *(b + reduction_size * 5),
                            *(b + reduction_size * 6),
                            *(b + reduction_size * 7)
                        };
                        if (nr != 8) {
                            if (nr == 12) {
                                vb2 = (float32x4_t) {
                                    *(b + reduction_size * 8),
                                    *(b + reduction_size * 9),
                                    *(b + reduction_size * 10),
                                    *(b + reduction_size * 11)
                                };
                            } else {
                                switch (nr) {
                                    case 9:
                                        vb2 = (float32x4_t) {
                                            *(b + reduction_size * 8),
                                            0.0f, 0.0f, 0.0f
                                        };
                                        break;
                                    case 10:
                                        vb2 = (float32x4_t) {
                                            *(b + reduction_size * 8),
                                            *(b + reduction_size * 9),
                                            0.0f, 0.0f
                                        };
                                        break;
                                    case 11:
                                        vb2 = (float32x4_t) {
                                            *(b + reduction_size * 8),
                                            *(b + reduction_size * 9),
                                            *(b + reduction_size * 10),
                                            0.0f
                                        };
                                        break;
                                }
                            }
                        }
                    } else {
                        switch (nr) {
                            case 5:
                                vb1 = (float32x4_t) {
                                    *(b + reduction_size * 4),
                                    0.0f, 0.0f, 0.0f
                                };
                                break;
                            case 6:
                                vb1 = (float32x4_t) {
                                    *(b + reduction_size * 4),
                                    *(b + reduction_size * 5),
                                    0.0f, 0.0f
                                };
                                break;
                            case 7:
                                vb1 = (float32x4_t) {
                                    *(b + reduction_size * 4),
                                    *(b + reduction_size * 5),
                                    *(b + reduction_size * 6),
                                    0.0f
                                };
                                break;
                        }
                    }
                }
            } else {
                switch (nr) {
                    case 1:
                        vb0 = (float32x4_t) {
                            *(b + reduction_size * 0),
                            0.0f, 0.0f, 0.0f
                        };
                        break;
                    case 2:
                        vb0 = (float32x4_t) {
                            *(b + reduction_size * 0),
                            *(b + reduction_size * 1),
                            0.0f, 0.0f
                        };
                        break;
                    case 3:
                        vb0 = (float32x4_t) {
                            *(b + reduction_size * 0),
                            *(b + reduction_size * 1),
                            *(b + reduction_size * 2),
                            0.0f
                        };
                        break;
                }
            }
            b += 1;
        } else {
            if (nr >= 4) {
                vb0 = vld1q_f32_aligned(b + 0);
                
                if (nr != 4) {
                    if (nr >= 8) {
                        vb1 = vld1q_f32_aligned(b + 4);
                        
                        if (nr != 8) {
                            if (nr == 12) {
                                vb2 = vld1q_f32_aligned(b + 8);
                                
                            } else {
                                switch (nr) {
                                    case 9:
                                        vb2 = (float32x4_t) {
                                            *(b + 8),
                                            0.0f, 0.0f, 0.0f
                                        };
                                        break;
                                    case 10:
                                        vb2 = (float32x4_t) {
                                            *(b + 8),
                                            *(b + 9),
                                            0.0f, 0.0f
                                        };
                                        break;
                                    case 11:
                                        vb2 = (float32x4_t) {
                                            *(b + 8),
                                            *(b + 9),
                                            *(b + 10),
                                            0.0f
                                        };
                                        break;
                                }
                            }
                        }
                    } else {
                        switch (nr) {
                            case 5:
                                vb1 = (float32x4_t) {
                                    *(b + 4),
                                    0.0f, 0.0f, 0.0f
                                };
                                break;
                            case 6:
                                vb1 = (float32x4_t) {
                                    *(b + 4),
                                    *(b + 5),
                                    0.0f, 0.0f
                                };
                                break;
                            case 7:
                                vb1 = (float32x4_t) {
                                    *(b + 4),
                                    *(b + 5),
                                    *(b + 6),
                                    0.0f
                                };
                                break;
                        }
                    }
                }
            } else {
                switch (nr) {
                    case 1:
                        vb0 = (float32x4_t) {
                            *(b + 0),
                            0.0f, 0.0f, 0.0f
                        };
                        break;
                    case 2:
                        vb0 = (float32x4_t) {
                            *(b + 0),
                            *(b + 1),
                            0.0f, 0.0f
                        };
                        break;
                    case 3:
                        vb0 = (float32x4_t) {
                            *(b + 0),
                            *(b + 1),
                            *(b + 2),
                            0.0f
                        };
                        break;
                }
            }
            b += output_col;
        }
        
        const float32x4_t va0 = trans_a ? vld1q_dup_f32(a + 0) : vld1q_dup_f32(a + reduction_size * 0);
        vc00 = vmuladdq_f32(vc00, va0, vb0);
        vc01 = vmuladdq_f32(vc01, va0, vb1);
        vc02 = vmuladdq_f32(vc02, va0, vb2);
        
        if (mr > 1) {
            const float32x4_t va1 = trans_a ? vld1q_dup_f32(a + 1) : vld1q_dup_f32(a + reduction_size * 1);
            vc10 = vmuladdq_f32(vc10, va1, vb0);
            vc11 = vmuladdq_f32(vc11, va1, vb1);
            vc12 = vmuladdq_f32(vc12, va1, vb2);
            
            if (mr > 2) {
                const float32x4_t va2 = trans_a ? vld1q_dup_f32(a + 2) : vld1q_dup_f32(a + reduction_size * 2);
                vc20 = vmuladdq_f32(vc20, va2, vb0);
                vc21 = vmuladdq_f32(vc21, va2, vb1);
                vc22 = vmuladdq_f32(vc22, va2, vb2);
                
                if (mr > 3) {
                    const float32x4_t va3 = trans_a ? vld1q_dup_f32(a + 3) : vld1q_dup_f32(a + reduction_size * 3);
                    vc30 = vmuladdq_f32(vc30, va3, vb0);
                    vc31 = vmuladdq_f32(vc31, va3, vb1);
                    vc32 = vmuladdq_f32(vc32, va3, vb2);
                }
            }
        }
        a += trans_a ? output_row : 1;
        
    } while (--k);
    
    // c = alpha * a * b + beta * c
    float32x4_t alpha_v4 = vdupq_n_f32(alpha);
    float32x2_t alpha_v2 = vdup_n_f32(alpha);
    float32x4_t beta_v4 = vdupq_n_f32(beta);
    float32x2_t beta_v2 = vdup_n_f32(beta);
    
    if (update) {
        float32x4_t vc0n = vc00;
        size_t nr0 = nr;
        float* c0n = c;
        if (nr0 > 4) {
            vst1q_f32(c0n, vaddq_f32(vld1q_f32_aligned(c0n), vmulq_f32(vc0n, alpha_v4)));
            c0n += 4;
            nr0 -= 4;
            vc0n = vc01;
            if (nr0 > 4) {
                vst1q_f32(c0n, vaddq_f32(vld1q_f32_aligned(c0n), vmulq_f32(vc0n, alpha_v4)));
                c0n += 4;
                nr0 -= 4;
                vc0n = vc02;
            }
        }
        switch (nr0) {
            case 4:
                vst1q_f32(c0n, vaddq_f32(vld1q_f32_aligned(c0n), vmulq_f32(vc0n, alpha_v4)));
                break;
            case 3:
                vst1_lane_f32(c0n + 2, vadd_f32(vld1_dup_f32(c0n + 2), vmul_f32(vget_high_f32(vc0n), alpha_v2)), 0);
            case 2:
                vst1_f32(c0n, vadd_f32(vld1_f32(c0n), vmul_f32(vget_low_f32(vc0n), alpha_v2)));
                break;
            case 1:
                vst1_lane_f32(c0n, vadd_f32(vld1_dup_f32(c0n), vmul_f32(vget_low_f32(vc0n), alpha_v2)), 0);
                break;
            default:
                break;
        }
        if (mr > 1) {
            c += output_col;
            float32x4_t vc1n = vc10;
            size_t nr1 = nr;
            float* c1n = c;
            if (nr1 > 4) {
                vst1q_f32(c1n, vaddq_f32(vld1q_f32_aligned(c1n), vmulq_f32(vc1n, alpha_v4)));
                c1n += 4;
                nr1 -= 4;
                vc1n = vc11;
                if (nr1 > 4) {
                    vst1q_f32(c1n, vaddq_f32(vld1q_f32_aligned(c1n), vmulq_f32(vc1n, alpha_v4)));
                    c1n += 4;
                    nr1 -= 4;
                    vc1n = vc12;
                }
            }
            switch (nr1) {
                case 4:
                    vst1q_f32(c1n, vaddq_f32(vld1q_f32_aligned(c1n), vmulq_f32(vc1n, alpha_v4)));
                    break;
                case 3:
                    vst1_lane_f32(c1n + 2, vadd_f32(vld1_dup_f32(c1n + 2), vmul_f32(vget_high_f32(vc1n), alpha_v2)), 0);
                case 2:
                    vst1_f32(c1n, vadd_f32(vld1_f32(c1n), vmul_f32(vget_low_f32(vc1n), alpha_v2)));
                    break;
                case 1:
                    vst1_lane_f32(c1n, vadd_f32(vld1_dup_f32(c1n), vmul_f32(vget_low_f32(vc1n), alpha_v2)), 0);
                    break;
                default:
                    break;
            }
            if (mr > 2) {
                c += output_col;
                float32x4_t vc2n = vc20;
                size_t nr2 = nr;
                float* c2n = c;
                if (nr2 > 4) {
                    vst1q_f32(c2n, vaddq_f32(vld1q_f32_aligned(c2n), vmulq_f32(vc2n, alpha_v4)));
                    c2n += 4;
                    nr2 -= 4;
                    vc2n = vc21;
                    if (nr2 > 4) {
                        vst1q_f32(c2n, vaddq_f32(vld1q_f32_aligned(c2n), vmulq_f32(vc2n, alpha_v4)));
                        c2n += 4;
                        nr2 -= 4;
                        vc2n = vc22;
                    }
                }
                switch (nr2) {
                    case 4:
                        vst1q_f32(c2n, vaddq_f32(vld1q_f32_aligned(c2n), vmulq_f32(vc2n, alpha_v4)));
                        break;
                    case 3:
                        vst1_lane_f32(c2n + 2, vadd_f32(vld1_dup_f32(c2n + 2), vmul_f32(vget_high_f32(vc2n), alpha_v2)), 0);
                    case 2:
                        vst1_f32(c2n, vadd_f32(vld1_f32(c2n), vmul_f32(vget_low_f32(vc2n), alpha_v2)));
                        break;
                    case 1:
                        vst1_lane_f32(c2n, vadd_f32(vld1_dup_f32(c2n), vmul_f32(vget_low_f32(vc2n), alpha_v2)), 0);
                        break;
                    default:
                        break;
                }
                if (mr > 3) {
                    c += output_col;
                    float32x4_t vc3n = vc30;
                    size_t nr3 = nr;
                    float* c3n = c;
                    if (nr3 > 4) {
                        vst1q_f32(c3n, vaddq_f32(vld1q_f32_aligned(c3n), vmulq_f32(vc3n, alpha_v4)));
                        c3n += 4;
                        nr3 -= 4;
                        vc3n = vc31;
                        if (nr3 > 4) {
                            vst1q_f32(c3n, vaddq_f32(vld1q_f32_aligned(c3n), vmulq_f32(vc3n, alpha_v4)));
                            c3n += 4;
                            nr3 -= 4;
                            vc3n = vc32;
                        }
                    }
                    switch (nr3) {
                        case 4:
                            vst1q_f32(c3n, vaddq_f32(vld1q_f32_aligned(c3n), vmulq_f32(vc3n, alpha_v4)));
                            break;
                        case 3:
                            vst1_lane_f32(c3n + 2, vadd_f32(vld1_dup_f32(c3n + 2), vmul_f32(vget_high_f32(vc3n), alpha_v2)), 0);
                        case 2:
                            vst1_f32(c3n, vadd_f32(vld1_f32(c3n), vmul_f32(vget_low_f32(vc3n), alpha_v2)));
                            break;
                        case 1:
                            vst1_lane_f32(c3n, vadd_f32(vld1_dup_f32(c3n), vmul_f32(vget_low_f32(vc3n), alpha_v2)), 0);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    } else {
        float32x4_t vc0n = vc00;
        size_t nr0 = nr;
        float* c0n = c;
        if (nr0 > 4) {
            vst1q_f32(c0n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c0n), beta_v4), vmulq_f32(vc0n, alpha_v4)));
            c0n += 4;
            nr0 -= 4;
            vc0n = vc01;
            if (nr0 > 4) {
                vst1q_f32(c0n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c0n), beta_v4), vmulq_f32(vc0n, alpha_v4)));
                c0n += 4;
                nr0 -= 4;
                vc0n = vc02;
            }
        }
        switch (nr0) {
            case 4:
                vst1q_f32(c0n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c0n), beta_v4), vmulq_f32(vc0n, alpha_v4)));
                break;
            case 3:
                vst1_lane_f32(c0n + 2, vadd_f32(vmul_f32(vld1_f32(c0n + 2), beta_v2), vmul_f32(vget_high_f32(vc0n), alpha_v2)), 0);
            case 2:
                vst1_f32(c0n, vadd_f32(vmul_f32(vld1_f32(c0n), beta_v2), vmul_f32(vget_low_f32(vc0n), alpha_v2)));
                break;
            case 1:
                vst1_lane_f32(c0n, vadd_f32(vmul_f32(vld1_f32(c0n), beta_v2), vmul_f32(vget_low_f32(vc0n), alpha_v2)), 0);
                break;
            default:
                break;
        }
        if (mr > 1) {
            c += output_col;
            float32x4_t vc1n = vc10;
            size_t nr1 = nr;
            float* c1n = c;
            if (nr1 > 4) {
                vst1q_f32(c1n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c1n), beta_v4), vmulq_f32(vc1n, alpha_v4)));
                c1n += 4;
                nr1 -= 4;
                vc1n = vc11;
                if (nr1 > 4) {
                    vst1q_f32(c1n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c1n), beta_v4), vmulq_f32(vc1n, alpha_v4)));
                    c1n += 4;
                    nr1 -= 4;
                    vc1n = vc12;
                }
            }
            switch (nr1) {
                case 4:
                    vst1q_f32(c1n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c1n), beta_v4), vmulq_f32(vc1n, alpha_v4)));
                    break;
                case 3:
                    vst1_lane_f32(c1n + 2, vadd_f32(vmul_f32(vld1_f32(c1n + 2), beta_v2), vmul_f32(vget_high_f32(vc1n), alpha_v2)), 0);
                case 2:
                    vst1_f32(c1n, vadd_f32(vmul_f32(vld1_f32(c1n), beta_v2), vmul_f32(vget_low_f32(vc1n), alpha_v2)));
                    break;
                case 1:
                    vst1_lane_f32(c1n, vadd_f32(vmul_f32(vld1_f32(c1n), beta_v2), vmul_f32(vget_low_f32(vc1n), alpha_v2)), 0);
                    break;
                default:
                    break;
            }
            if (mr > 2) {
                c += output_col;
                float32x4_t vc2n = vc20;
                size_t nr2 = nr;
                float* c2n = c;
                if (nr2 > 4) {
                    vst1q_f32(c2n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c2n), beta_v4), vmulq_f32(vc2n, alpha_v4)));
                    c2n += 4;
                    nr2 -= 4;
                    vc2n = vc21;
                    if (nr2 > 4) {
                        vst1q_f32(c2n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c2n), beta_v4), vmulq_f32(vc2n, alpha_v4)));
                        c2n += 4;
                        nr2 -= 4;
                        vc2n = vc22;
                    }
                }
                switch (nr2) {
                    case 4:
                        vst1q_f32(c2n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c2n), beta_v4), vmulq_f32(vc2n, alpha_v4)));
                        break;
                    case 3:
                        vst1_lane_f32(c2n + 2, vadd_f32(vmul_f32(vld1_f32(c2n + 2), beta_v2), vmul_f32(vget_high_f32(vc2n), alpha_v2)), 0);
                    case 2:
                        vst1_f32(c2n, vadd_f32(vmul_f32(vld1_f32(c2n), beta_v2), vmul_f32(vget_low_f32(vc2n), alpha_v2)));
                        break;
                    case 1:
                        vst1_lane_f32(c2n, vadd_f32(vmul_f32(vld1_f32(c2n), beta_v2), vmul_f32(vget_low_f32(vc2n), alpha_v2)), 0);
                        break;
                    default:
                        break;
                }
                if (mr > 3) {
                    c += output_col;
                    float32x4_t vc3n = vc30;
                    size_t nr3 = nr;
                    float* c3n = c;
                    if (nr3 > 4) {
                        vst1q_f32(c3n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c3n), beta_v4), vmulq_f32(vc3n, alpha_v4)));
                        c3n += 4;
                        nr3 -= 4;
                        vc3n = vc31;
                        if (nr3 > 4) {
                            vst1q_f32(c3n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c3n), beta_v4), vmulq_f32(vc3n, alpha_v4)));
                            c3n += 4;
                            nr3 -= 4;
                            vc3n = vc32;
                        }
                    }
                    switch (nr3) {
                        case 4:
                            vst1q_f32(c3n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c3n), beta_v4), vmulq_f32(vc3n, alpha_v4)));
                            break;
                        case 3:
                            vst1_lane_f32(c3n + 2, vadd_f32(vmul_f32(vld1_f32(c3n + 2), beta_v2), vmul_f32(vget_high_f32(vc3n), alpha_v2)), 0);
                        case 2:
                            vst1_f32(c3n, vadd_f32(vmul_f32(vld1_f32(c3n), beta_v2), vmul_f32(vget_low_f32(vc3n), alpha_v2)));
                            break;
                        case 1:
                            vst1_lane_f32(c3n, vadd_f32(vmul_f32(vld1_f32(c3n), beta_v2), vmul_f32(vget_low_f32(vc3n), alpha_v2)), 0);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
}

void nnp_sgemm_only_8x8(size_t k,
                        size_t update,
                        size_t output_row,
                        size_t output_col,
                        size_t reduction_size,
                        const bool trans_a,
                        const bool trans_b,
                        const float alpha,
                        const float beta,
                        const float *a,
                        const float *b,
                        float *c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f);
    float32x4_t vc40 = vdupq_n_f32(0.0f), vc41 = vdupq_n_f32(0.0f);
    float32x4_t vc50 = vdupq_n_f32(0.0f), vc51 = vdupq_n_f32(0.0f);
    float32x4_t vc60 = vdupq_n_f32(0.0f), vc61 = vdupq_n_f32(0.0f);
    float32x4_t vc70 = vdupq_n_f32(0.0f), vc71 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t va0, va1;
        if (trans_a) {
            va0 = vld1q_f32_aligned(a + 0);
            va1 = vld1q_f32_aligned(a + 4);
            a += output_row;
        } else {
            va0 = (float32x4_t) {
                *(a + reduction_size * 0),
                *(a + reduction_size * 1),
                *(a + reduction_size * 2),
                *(a + reduction_size * 3),
            };
            va1 = (float32x4_t) {
                *(a + reduction_size * 4),
                *(a + reduction_size * 5),
                *(a + reduction_size * 6),
                *(a + reduction_size * 7),
            };
            a += 1;
        }
        
        float32x4_t vb0, vb1;
        if (trans_b) {
            vb0 = (float32x4_t) {
                *(b + reduction_size * 0),
                *(b + reduction_size * 1),
                *(b + reduction_size * 2),
                *(b + reduction_size * 3),
            };
            vb1 = (float32x4_t) {
                *(b + reduction_size * 4),
                *(b + reduction_size * 5),
                *(b + reduction_size * 6),
                *(b + reduction_size * 7),
            };
            b += 1;
        } else {
            vb0 = vld1q_f32_aligned(b + 0);
            vb1 = vld1q_f32_aligned(b + 4);
            b += output_col;
        }
        
#if defined(__aarch64__)
        vc00 = vfmaq_lane_f32(vc00, vb0, vget_low_f32(va0),  0);
        vc10 = vfmaq_lane_f32(vc10, vb0, vget_low_f32(va0),  1);
        vc20 = vfmaq_lane_f32(vc20, vb0, vget_high_f32(va0), 0);
        vc30 = vfmaq_lane_f32(vc30, vb0, vget_high_f32(va0), 1);
        vc01 = vfmaq_lane_f32(vc01, vb1, vget_low_f32(va0),  0);
        vc11 = vfmaq_lane_f32(vc11, vb1, vget_low_f32(va0),  1);
        vc21 = vfmaq_lane_f32(vc21, vb1, vget_high_f32(va0), 0);
        vc31 = vfmaq_lane_f32(vc31, vb1, vget_high_f32(va0), 1);
        vc40 = vfmaq_lane_f32(vc40, vb0, vget_low_f32(va1),  0);
        vc50 = vfmaq_lane_f32(vc50, vb0, vget_low_f32(va1),  1);
        vc60 = vfmaq_lane_f32(vc60, vb0, vget_high_f32(va1), 0);
        vc70 = vfmaq_lane_f32(vc70, vb0, vget_high_f32(va1), 1);
        vc41 = vfmaq_lane_f32(vc41, vb1, vget_low_f32(va1),  0);
        vc51 = vfmaq_lane_f32(vc51, vb1, vget_low_f32(va1),  1);
        vc61 = vfmaq_lane_f32(vc61, vb1, vget_high_f32(va1), 0);
        vc71 = vfmaq_lane_f32(vc71, vb1, vget_high_f32(va1), 1);
#else
        vc00 = vmlaq_lane_f32(vc00, vb0, vget_low_f32(va0),  0);
        vc10 = vmlaq_lane_f32(vc10, vb0, vget_low_f32(va0),  1);
        vc20 = vmlaq_lane_f32(vc20, vb0, vget_high_f32(va0), 0);
        vc30 = vmlaq_lane_f32(vc30, vb0, vget_high_f32(va0), 1);
        vc01 = vmlaq_lane_f32(vc01, vb1, vget_low_f32(va0),  0);
        vc11 = vmlaq_lane_f32(vc11, vb1, vget_low_f32(va0),  1);
        vc21 = vmlaq_lane_f32(vc21, vb1, vget_high_f32(va0), 0);
        vc31 = vmlaq_lane_f32(vc31, vb1, vget_high_f32(va0), 1);
        vc40 = vmlaq_lane_f32(vc40, vb0, vget_low_f32(va1),  0);
        vc50 = vmlaq_lane_f32(vc50, vb0, vget_low_f32(va1),  1);
        vc60 = vmlaq_lane_f32(vc60, vb0, vget_high_f32(va1), 0);
        vc70 = vmlaq_lane_f32(vc70, vb0, vget_high_f32(va1), 1);
        vc41 = vmlaq_lane_f32(vc41, vb1, vget_low_f32(va1),  0);
        vc51 = vmlaq_lane_f32(vc51, vb1, vget_low_f32(va1),  1);
        vc61 = vmlaq_lane_f32(vc61, vb1, vget_high_f32(va1), 0);
        vc71 = vmlaq_lane_f32(vc71, vb1, vget_high_f32(va1), 1);
#endif
    } while (--k);
    
    // c = alpha * a * b + beta * c
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    
    if (update) {
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc00, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc01, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc10, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc11, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc20, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc21, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc30, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc31, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc40, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc41, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc50, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc51, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc60, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc61, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), vmulq_f32(vc70, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), vmulq_f32(vc71, alpha_v)));
    } else {
        float32x4_t beta_v = vdupq_n_f32(beta);
        
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc00, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc01, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc10, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc11, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc20, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc21, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc30, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc31, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc40, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc41, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc50, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc51, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc60, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc61, alpha_v)));
        c += output_col;
        vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 0), beta_v), vmulq_f32(vc70, alpha_v)));
        vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c + 4), beta_v), vmulq_f32(vc71, alpha_v)));
    }
}

void nnp_sgemm_upto_8x8(size_t mr,
                        size_t nr,
                        size_t k,
                        size_t update,
                        size_t output_row,
                        size_t output_col,
                        size_t reduction_size,
                        const bool trans_a,
                        const bool trans_b,
                        const float alpha,
                        const float beta,
                        const float *a,
                        const float *b,
                        float *c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f);
    float32x4_t vc40 = vdupq_n_f32(0.0f), vc41 = vdupq_n_f32(0.0f);
    float32x4_t vc50 = vdupq_n_f32(0.0f), vc51 = vdupq_n_f32(0.0f);
    float32x4_t vc60 = vdupq_n_f32(0.0f), vc61 = vdupq_n_f32(0.0f);
    float32x4_t vc70 = vdupq_n_f32(0.0f), vc71 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t vb0 = vdupq_n_f32(0.0f), vb1 = vdupq_n_f32(0.0f);
        
        if (trans_b) {
            if (nr >= 4) {
                vb0 = (float32x4_t) {
                    *(b + reduction_size * 0),
                    *(b + reduction_size * 1),
                    *(b + reduction_size * 2),
                    *(b + reduction_size * 3)
                };
                if (nr != 4) {
                    if (nr == 8) {
                        vb1 = (float32x4_t) {
                            *(b + reduction_size * 4),
                            *(b + reduction_size * 5),
                            *(b + reduction_size * 6),
                            *(b + reduction_size * 7)
                        };
                    } else {
                        switch (nr) {
                            case 5:
                                vb1 = (float32x4_t) {
                                    *(b + reduction_size * 4),
                                    0.0f, 0.0f, 0.0f
                                };
                                break;
                            case 6:
                                vb1 = (float32x4_t) {
                                    *(b + reduction_size * 4),
                                    *(b + reduction_size * 5),
                                    0.0f, 0.0f
                                };
                                break;
                            case 7:
                                vb1 = (float32x4_t) {
                                    *(b + reduction_size * 4),
                                    *(b + reduction_size * 5),
                                    *(b + reduction_size * 6),
                                    0.0f
                                };
                                break;
                        }
                    }
                }
            } else {
                switch (nr) {
                    case 1:
                        vb0 = (float32x4_t) {
                            *(b + reduction_size * 0),
                            0.0f, 0.0f, 0.0f
                        };
                        break;
                    case 2:
                        vb0 = (float32x4_t) {
                            *(b + reduction_size * 0),
                            *(b + reduction_size * 1),
                            0.0f, 0.0f
                        };
                        break;
                    case 3:
                        vb0 = (float32x4_t) {
                            *(b + reduction_size * 0),
                            *(b + reduction_size * 1),
                            *(b + reduction_size * 2),
                            0.0f
                        };
                        break;
                }
            }
            b += 1;
        } else {
            if (nr >= 4) {
                vb0 = vld1q_f32_aligned(b + 0);
                
                if (nr != 4) {
                    if (nr == 8) {
                        vb1 = vld1q_f32_aligned(b + 4);
                        
                    } else {
                        switch (nr) {
                            case 5:
                                vb1 = (float32x4_t) {
                                    *(b + 4),
                                    0.0f, 0.0f, 0.0f
                                };
                                break;
                            case 6:
                                vb1 = (float32x4_t) {
                                    *(b + 4),
                                    *(b + 5),
                                    0.0f, 0.0f
                                };
                                break;
                            case 7:
                                vb1 = (float32x4_t) {
                                    *(b + 4),
                                    *(b + 5),
                                    *(b + 6),
                                    0.0f
                                };
                                break;
                        }
                    }
                }
            } else {
                switch (nr) {
                    case 1:
                        vb0 = (float32x4_t) {
                            *(b + 0),
                            0.0f, 0.0f, 0.0f
                        };
                        break;
                    case 2:
                        vb0 = (float32x4_t) {
                            *(b + 0),
                            *(b + 1),
                            0.0f, 0.0f
                        };
                        break;
                    case 3:
                        vb0 = (float32x4_t) {
                            *(b + 0),
                            *(b + 1),
                            *(b + 2),
                            0.0f
                        };
                        break;
                }
            }
            b += output_col;
        }
        
        const float32x4_t va0 = trans_a ? vld1q_dup_f32(a + 0) : vld1q_dup_f32(a + reduction_size * 0);
        vc00 = vmuladdq_f32(vc00, va0, vb0);
        vc01 = vmuladdq_f32(vc01, va0, vb1);
        
        if (mr > 1) {
            const float32x4_t va1 = trans_a ? vld1q_dup_f32(a + 1) : vld1q_dup_f32(a + reduction_size * 1);
            vc10 = vmuladdq_f32(vc10, va1, vb0);
            vc11 = vmuladdq_f32(vc11, va1, vb1);
            
            if (mr > 2) {
                const float32x4_t va2 = trans_a ? vld1q_dup_f32(a + 2) : vld1q_dup_f32(a + reduction_size * 2);
                vc20 = vmuladdq_f32(vc20, va2, vb0);
                vc21 = vmuladdq_f32(vc21, va2, vb1);
                
                if (mr > 3) {
                    const float32x4_t va3 = trans_a ? vld1q_dup_f32(a + 3) : vld1q_dup_f32(a + reduction_size * 3);
                    vc30 = vmuladdq_f32(vc30, va3, vb0);
                    vc31 = vmuladdq_f32(vc31, va3, vb1);
                    
                    if (mr > 4) {
                        const float32x4_t va4 = trans_a ? vld1q_dup_f32(a + 4) : vld1q_dup_f32(a + reduction_size * 4);
                        vc40 = vmuladdq_f32(vc40, va4, vb0);
                        vc41 = vmuladdq_f32(vc41, va4, vb1);
                        
                        if (mr > 5) {
                            const float32x4_t va5 = trans_a ? vld1q_dup_f32(a + 5) : vld1q_dup_f32(a + reduction_size * 5);
                            vc50 = vmuladdq_f32(vc50, va5, vb0);
                            vc51 = vmuladdq_f32(vc51, va5, vb1);
                            
                            if (mr > 6) {
                                const float32x4_t va6 = trans_a ? vld1q_dup_f32(a + 6) : vld1q_dup_f32(a + reduction_size * 6);
                                vc60 = vmuladdq_f32(vc60, va6, vb0);
                                vc61 = vmuladdq_f32(vc61, va6, vb1);
                                
                                if (mr > 7) {
                                    const float32x4_t va7 = trans_a ? vld1q_dup_f32(a + 7) : vld1q_dup_f32(a + reduction_size * 7);
                                    vc70 = vmuladdq_f32(vc70, va7, vb0);
                                    vc71 = vmuladdq_f32(vc71, va7, vb1);
                                }
                            }
                        }
                    }
                }
            }
        }
        a += trans_a ? output_row : 1;
        
    } while (--k);
    
    // c = alpha * a * b + beta * c
    float32x4_t alpha_v4 = vdupq_n_f32(alpha);
    float32x2_t alpha_v2 = vdup_n_f32(alpha);
    float32x4_t beta_v4 = vdupq_n_f32(beta);
    float32x2_t beta_v2 = vdup_n_f32(beta);
    
    if (update) {
        float32x4_t vc0n = vc00;
        size_t nr0 = nr;
        float* c0n = c;
        if (nr0 > 4) {
            vst1q_f32(c0n, vaddq_f32(vld1q_f32_aligned(c0n), vmulq_f32(vc0n, alpha_v4)));
            c0n += 4;
            nr0 -= 4;
            vc0n = vc01;
        }
        switch (nr0) {
            case 4:
                vst1q_f32(c0n, vaddq_f32(vld1q_f32_aligned(c0n), vmulq_f32(vc0n, alpha_v4)));
                break;
            case 3:
                vst1_lane_f32(c0n + 2, vadd_f32(vld1_dup_f32(c0n + 2), vmul_f32(vget_high_f32(vc0n), alpha_v2)), 0);
            case 2:
                vst1_f32(c0n, vadd_f32(vld1_f32(c0n), vmul_f32(vget_low_f32(vc0n), alpha_v2)));
                break;
            case 1:
                vst1_lane_f32(c0n, vadd_f32(vld1_dup_f32(c0n), vmul_f32(vget_low_f32(vc0n), alpha_v2)), 0);
                break;
            default:
                break;
        }
        if (mr > 1) {
            c += output_col;
            float32x4_t vc1n = vc10;
            size_t nr1 = nr;
            float* c1n = c;
            if (nr1 > 4) {
                vst1q_f32(c1n, vaddq_f32(vld1q_f32_aligned(c1n), vmulq_f32(vc1n, alpha_v4)));
                c1n += 4;
                nr1 -= 4;
                vc1n = vc11;
            }
            switch (nr1) {
                case 4:
                    vst1q_f32(c1n, vaddq_f32(vld1q_f32_aligned(c1n), vmulq_f32(vc1n, alpha_v4)));
                    break;
                case 3:
                    vst1_lane_f32(c1n + 2, vadd_f32(vld1_dup_f32(c1n + 2), vmul_f32(vget_high_f32(vc1n), alpha_v2)), 0);
                case 2:
                    vst1_f32(c1n, vadd_f32(vld1_f32(c1n), vmul_f32(vget_low_f32(vc1n), alpha_v2)));
                    break;
                case 1:
                    vst1_lane_f32(c1n, vadd_f32(vld1_dup_f32(c1n), vmul_f32(vget_low_f32(vc1n), alpha_v2)), 0);
                    break;
                default:
                    break;
            }
            if (mr > 2) {
                c += output_col;
                float32x4_t vc2n = vc20;
                size_t nr2 = nr;
                float* c2n = c;
                if (nr2 > 4) {
                    vst1q_f32(c2n, vaddq_f32(vld1q_f32_aligned(c2n), vmulq_f32(vc2n, alpha_v4)));
                    c2n += 4;
                    nr2 -= 4;
                    vc2n = vc21;
                }
                switch (nr2) {
                    case 4:
                        vst1q_f32(c2n, vaddq_f32(vld1q_f32_aligned(c2n), vmulq_f32(vc2n, alpha_v4)));
                        break;
                    case 3:
                        vst1_lane_f32(c2n + 2, vadd_f32(vld1_dup_f32(c2n + 2), vmul_f32(vget_high_f32(vc2n), alpha_v2)), 0);
                    case 2:
                        vst1_f32(c2n, vadd_f32(vld1_f32(c2n), vmul_f32(vget_low_f32(vc2n), alpha_v2)));
                        break;
                    case 1:
                        vst1_lane_f32(c2n, vadd_f32(vld1_dup_f32(c2n), vmul_f32(vget_low_f32(vc2n), alpha_v2)), 0);
                        break;
                    default:
                        break;
                }
                if (mr > 3) {
                    c += output_col;
                    float32x4_t vc3n = vc30;
                    size_t nr3 = nr;
                    float* c3n = c;
                    if (nr3 > 4) {
                        vst1q_f32(c3n, vaddq_f32(vld1q_f32_aligned(c3n), vmulq_f32(vc3n, alpha_v4)));
                        c3n += 4;
                        nr3 -= 4;
                        vc3n = vc31;
                    }
                    switch (nr3) {
                        case 4:
                            vst1q_f32(c3n, vaddq_f32(vld1q_f32_aligned(c3n), vmulq_f32(vc3n, alpha_v4)));
                            break;
                        case 3:
                            vst1_lane_f32(c3n + 2, vadd_f32(vld1_dup_f32(c3n + 2), vmul_f32(vget_high_f32(vc3n), alpha_v2)), 0);
                        case 2:
                            vst1_f32(c3n, vadd_f32(vld1_f32(c3n), vmul_f32(vget_low_f32(vc3n), alpha_v2)));
                            break;
                        case 1:
                            vst1_lane_f32(c3n, vadd_f32(vld1_dup_f32(c3n), vmul_f32(vget_low_f32(vc3n), alpha_v2)), 0);
                            break;
                        default:
                            break;
                    }
                    if (mr > 4) {
                        c += output_col;
                        float32x4_t vc4n = vc40;
                        size_t nr4 = nr;
                        float* c4n = c;
                        if (nr4 > 4) {
                            vst1q_f32(c4n, vaddq_f32(vld1q_f32_aligned(c4n), vmulq_f32(vc4n, alpha_v4)));
                            c4n += 4;
                            nr4 -= 4;
                            vc4n = vc41;
                        }
                        switch (nr4) {
                            case 4:
                                vst1q_f32(c4n, vaddq_f32(vld1q_f32_aligned(c4n), vmulq_f32(vc4n, alpha_v4)));
                                break;
                            case 3:
                                vst1_lane_f32(c4n + 2, vadd_f32(vld1_dup_f32(c4n + 2), vmul_f32(vget_high_f32(vc4n), alpha_v2)), 0);
                            case 2:
                                vst1_f32(c4n, vadd_f32(vld1_f32(c4n), vmul_f32(vget_low_f32(vc4n), alpha_v2)));
                                break;
                            case 1:
                                vst1_lane_f32(c4n, vadd_f32(vld1_dup_f32(c4n), vmul_f32(vget_low_f32(vc4n), alpha_v2)), 0);
                                break;
                            default:
                                break;
                        }
                        if (mr > 5) {
                            c += output_col;
                            float32x4_t vc5n = vc50;
                            size_t nr5 = nr;
                            float* c5n = c;
                            if (nr5 > 4) {
                                vst1q_f32(c5n, vaddq_f32(vld1q_f32_aligned(c5n), vmulq_f32(vc5n, alpha_v4)));
                                c5n += 4;
                                nr5 -= 4;
                                vc5n = vc51;
                            }
                            switch (nr5) {
                                case 4:
                                    vst1q_f32(c5n, vaddq_f32(vld1q_f32_aligned(c5n), vmulq_f32(vc5n, alpha_v4)));
                                    break;
                                case 3:
                                    vst1_lane_f32(c5n + 2, vadd_f32(vld1_dup_f32(c5n + 2), vmul_f32(vget_high_f32(vc5n), alpha_v2)), 0);
                                case 2:
                                    vst1_f32(c5n, vadd_f32(vld1_f32(c5n), vmul_f32(vget_low_f32(vc5n), alpha_v2)));
                                    break;
                                case 1:
                                    vst1_lane_f32(c5n, vadd_f32(vld1_dup_f32(c5n), vmul_f32(vget_low_f32(vc5n), alpha_v2)), 0);
                                    break;
                                default:
                                    break;
                            }
                            if (mr > 6) {
                                c += output_col;
                                float32x4_t vc6n = vc60;
                                size_t nr6 = nr;
                                float* c6n = c;
                                if (nr6 > 4) {
                                    vst1q_f32(c6n, vaddq_f32(vld1q_f32_aligned(c6n), vmulq_f32(vc6n, alpha_v4)));
                                    c6n += 4;
                                    nr6 -= 4;
                                    vc6n = vc61;
                                }
                                switch (nr6) {
                                    case 4:
                                        vst1q_f32(c6n, vaddq_f32(vld1q_f32_aligned(c6n), vmulq_f32(vc6n, alpha_v4)));
                                        break;
                                    case 3:
                                        vst1_lane_f32(c6n + 2, vadd_f32(vld1_dup_f32(c6n + 2), vmul_f32(vget_high_f32(vc6n), alpha_v2)), 0);
                                    case 2:
                                        vst1_f32(c6n, vadd_f32(vld1_f32(c6n), vmul_f32(vget_low_f32(vc6n), alpha_v2)));
                                        break;
                                    case 1:
                                        vst1_lane_f32(c6n, vadd_f32(vld1_dup_f32(c6n), vmul_f32(vget_low_f32(vc6n), alpha_v2)), 0);
                                        break;
                                    default:
                                        break;
                                }
                                if (mr > 7) {
                                    c += output_col;
                                    float32x4_t vc7n = vc70;
                                    size_t nr7 = nr;
                                    float* c7n = c;
                                    if (nr7 > 4) {
                                        vst1q_f32(c7n, vaddq_f32(vld1q_f32_aligned(c7n), vmulq_f32(vc7n, alpha_v4)));
                                        c7n += 4;
                                        nr7 -= 4;
                                        vc7n = vc71;
                                    }
                                    switch (nr7) {
                                        case 4:
                                            vst1q_f32(c7n, vaddq_f32(vld1q_f32_aligned(c7n), vmulq_f32(vc7n, alpha_v4)));
                                            break;
                                        case 3:
                                            vst1_lane_f32(c7n + 2, vadd_f32(vld1_dup_f32(c7n + 2), vmul_f32(vget_high_f32(vc7n), alpha_v2)), 0);
                                        case 2:
                                            vst1_f32(c7n, vadd_f32(vld1_f32(c7n), vmul_f32(vget_low_f32(vc7n), alpha_v2)));
                                            break;
                                        case 1:
                                            vst1_lane_f32(c7n, vadd_f32(vld1_dup_f32(c7n), vmul_f32(vget_low_f32(vc7n), alpha_v2)), 0);
                                            break;
                                        default:
                                            break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        float32x4_t vc0n = vc00;
        size_t nr0 = nr;
        float* c0n = c;
        if (nr0 > 4) {
            vst1q_f32(c0n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c0n), beta_v4), vmulq_f32(vc0n, alpha_v4)));
            c0n += 4;
            nr0 -= 4;
            vc0n = vc01;
        }
        switch (nr0) {
            case 4:
                vst1q_f32(c0n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c0n), beta_v4), vmulq_f32(vc0n, alpha_v4)));
                break;
            case 3:
                vst1_lane_f32(c0n + 2, vadd_f32(vmul_f32(vld1_f32(c0n + 2), beta_v2), vmul_f32(vget_high_f32(vc0n), alpha_v2)), 0);
            case 2:
                vst1_f32(c0n, vadd_f32(vmul_f32(vld1_f32(c0n), beta_v2), vmul_f32(vget_low_f32(vc0n), alpha_v2)));
                break;
            case 1:
                vst1_lane_f32(c0n, vadd_f32(vmul_f32(vld1_f32(c0n), beta_v2), vmul_f32(vget_low_f32(vc0n), alpha_v2)), 0);
                break;
            default:
                break;
        }
        if (mr > 1) {
            c += output_col;
            float32x4_t vc1n = vc10;
            size_t nr1 = nr;
            float* c1n = c;
            if (nr1 > 4) {
                vst1q_f32(c1n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c1n), beta_v4), vmulq_f32(vc1n, alpha_v4)));
                c1n += 4;
                nr1 -= 4;
                vc1n = vc11;
            }
            switch (nr1) {
                case 4:
                    vst1q_f32(c1n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c1n), beta_v4), vmulq_f32(vc1n, alpha_v4)));
                    break;
                case 3:
                    vst1_lane_f32(c1n + 2, vadd_f32(vmul_f32(vld1_f32(c1n + 2), beta_v2), vmul_f32(vget_high_f32(vc1n), alpha_v2)), 0);
                case 2:
                    vst1_f32(c1n, vadd_f32(vmul_f32(vld1_f32(c1n), beta_v2), vmul_f32(vget_low_f32(vc1n), alpha_v2)));
                    break;
                case 1:
                    vst1_lane_f32(c1n, vadd_f32(vmul_f32(vld1_f32(c1n), beta_v2), vmul_f32(vget_low_f32(vc1n), alpha_v2)), 0);
                    break;
                default:
                    break;
            }
            if (mr > 2) {
                c += output_col;
                float32x4_t vc2n = vc20;
                size_t nr2 = nr;
                float* c2n = c;
                if (nr2 > 4) {
                    vst1q_f32(c2n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c2n), beta_v4), vmulq_f32(vc2n, alpha_v4)));
                    c2n += 4;
                    nr2 -= 4;
                    vc2n = vc21;
                }
                switch (nr2) {
                    case 4:
                        vst1q_f32(c2n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c2n), beta_v4), vmulq_f32(vc2n, alpha_v4)));
                        break;
                    case 3:
                        vst1_lane_f32(c2n + 2, vadd_f32(vmul_f32(vld1_f32(c2n + 2), beta_v2), vmul_f32(vget_high_f32(vc2n), alpha_v2)), 0);
                    case 2:
                        vst1_f32(c2n, vadd_f32(vmul_f32(vld1_f32(c2n), beta_v2), vmul_f32(vget_low_f32(vc2n), alpha_v2)));
                        break;
                    case 1:
                        vst1_lane_f32(c2n, vadd_f32(vmul_f32(vld1_f32(c2n), beta_v2), vmul_f32(vget_low_f32(vc2n), alpha_v2)), 0);
                        break;
                    default:
                        break;
                }
                if (mr > 3) {
                    c += output_col;
                    float32x4_t vc3n = vc30;
                    size_t nr3 = nr;
                    float* c3n = c;
                    if (nr3 > 4) {
                        vst1q_f32(c3n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c3n), beta_v4), vmulq_f32(vc3n, alpha_v4)));
                        c3n += 4;
                        nr3 -= 4;
                        vc3n = vc31;
                    }
                    switch (nr3) {
                        case 4:
                            vst1q_f32(c3n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c3n), beta_v4), vmulq_f32(vc3n, alpha_v4)));
                            break;
                        case 3:
                            vst1_lane_f32(c3n + 2, vadd_f32(vmul_f32(vld1_f32(c3n + 2), beta_v2), vmul_f32(vget_high_f32(vc3n), alpha_v2)), 0);
                        case 2:
                            vst1_f32(c3n, vadd_f32(vmul_f32(vld1_f32(c3n), beta_v2), vmul_f32(vget_low_f32(vc3n), alpha_v2)));
                            break;
                        case 1:
                            vst1_lane_f32(c3n, vadd_f32(vmul_f32(vld1_f32(c3n), beta_v2), vmul_f32(vget_low_f32(vc3n), alpha_v2)), 0);
                            break;
                        default:
                            break;
                    }
                    if (mr > 4) {
                        c += output_col;
                        float32x4_t vc4n = vc40;
                        size_t nr4 = nr;
                        float* c4n = c;
                        if (nr4 > 4) {
                            vst1q_f32(c4n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c4n), beta_v4), vmulq_f32(vc4n, alpha_v4)));
                            c4n += 4;
                            nr4 -= 4;
                            vc4n = vc41;
                        }
                        switch (nr4) {
                            case 4:
                                vst1q_f32(c4n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c4n), beta_v4), vmulq_f32(vc4n, alpha_v4)));
                                break;
                            case 3:
                                vst1_lane_f32(c4n + 2, vadd_f32(vmul_f32(vld1_f32(c4n + 2), beta_v2), vmul_f32(vget_high_f32(vc4n), alpha_v2)), 0);
                            case 2:
                                vst1_f32(c4n, vadd_f32(vmul_f32(vld1_f32(c4n), beta_v2), vmul_f32(vget_low_f32(vc4n), alpha_v2)));
                                break;
                            case 1:
                                vst1_lane_f32(c4n, vadd_f32(vmul_f32(vld1_f32(c4n), beta_v2), vmul_f32(vget_low_f32(vc4n), alpha_v2)), 0);
                                break;
                            default:
                                break;
                        }
                        if (mr > 5) {
                            c += output_col;
                            float32x4_t vc5n = vc50;
                            size_t nr5 = nr;
                            float* c5n = c;
                            if (nr5 > 4) {
                                vst1q_f32(c5n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c5n), beta_v4), vmulq_f32(vc5n, alpha_v4)));
                                c5n += 4;
                                nr5 -= 4;
                                vc5n = vc51;
                            }
                            switch (nr5) {
                                case 4:
                                    vst1q_f32(c5n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c5n), beta_v4), vmulq_f32(vc5n, alpha_v4)));
                                    break;
                                case 3:
                                    vst1_lane_f32(c5n + 2, vadd_f32(vmul_f32(vld1_f32(c5n + 2), beta_v2), vmul_f32(vget_high_f32(vc5n), alpha_v2)), 0);
                                case 2:
                                    vst1_f32(c5n, vadd_f32(vmul_f32(vld1_f32(c5n), beta_v2), vmul_f32(vget_low_f32(vc5n), alpha_v2)));
                                    break;
                                case 1:
                                    vst1_lane_f32(c5n, vadd_f32(vmul_f32(vld1_f32(c5n), beta_v2), vmul_f32(vget_low_f32(vc5n), alpha_v2)), 0);
                                    break;
                                default:
                                    break;
                            }
                            if (mr > 6) {
                                c += output_col;
                                float32x4_t vc6n = vc60;
                                size_t nr6 = nr;
                                float* c6n = c;
                                if (nr6 > 4) {
                                    vst1q_f32(c6n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c6n), beta_v4), vmulq_f32(vc6n, alpha_v4)));
                                    c6n += 4;
                                    nr6 -= 4;
                                    vc6n = vc61;
                                }
                                switch (nr6) {
                                    case 4:
                                        vst1q_f32(c6n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c6n), beta_v4), vmulq_f32(vc6n, alpha_v4)));
                                        break;
                                    case 3:
                                        vst1_lane_f32(c6n + 2, vadd_f32(vmul_f32(vld1_f32(c6n + 2), beta_v2), vmul_f32(vget_high_f32(vc6n), alpha_v2)), 0);
                                    case 2:
                                        vst1_f32(c6n, vadd_f32(vmul_f32(vld1_f32(c6n), beta_v2), vmul_f32(vget_low_f32(vc6n), alpha_v2)));
                                        break;
                                    case 1:
                                        vst1_lane_f32(c6n, vadd_f32(vmul_f32(vld1_f32(c6n), beta_v2), vmul_f32(vget_low_f32(vc6n), alpha_v2)), 0);
                                        break;
                                    default:
                                        break;
                                }
                                if (mr > 7) {
                                    c += output_col;
                                    float32x4_t vc7n = vc70;
                                    size_t nr7 = nr;
                                    float* c7n = c;
                                    if (nr7 > 4) {
                                        vst1q_f32(c7n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c7n), beta_v4), vmulq_f32(vc7n, alpha_v4)));
                                        c7n += 4;
                                        nr7 -= 4;
                                        vc7n = vc71;
                                    }
                                    switch (nr7) {
                                        case 4:
                                            vst1q_f32(c7n, vaddq_f32(vmulq_f32(vld1q_f32_aligned(c7n), beta_v4), vmulq_f32(vc7n, alpha_v4)));
                                            break;
                                        case 3:
                                            vst1_lane_f32(c7n + 2, vadd_f32(vmul_f32(vld1_f32(c7n + 2), beta_v2), vmul_f32(vget_high_f32(vc7n), alpha_v2)), 0);
                                        case 2:
                                            vst1_f32(c7n, vadd_f32(vmul_f32(vld1_f32(c7n), beta_v2), vmul_f32(vget_low_f32(vc7n), alpha_v2)));
                                            break;
                                        case 1:
                                            vst1_lane_f32(c7n, vadd_f32(vmul_f32(vld1_f32(c7n), beta_v2), vmul_f32(vget_low_f32(vc7n), alpha_v2)), 0);
                                            break;
                                        default:
                                            break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// modified from https://github.com/Tencent/ncnn/blob/master/src/layer/arm/innerproduct_arm.cpp
void nnp_sgemm_1x1(size_t m,
                   size_t n,
                   size_t k,
                   const bool trans_b,
                   const float alpha,
                   const float beta,
                   const float *a,
                   const float *b,
                   float *c)
{
    float sum = 0.0f;
    float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
    
    size_t nn = k >> 3;
    size_t remain = k & 7;
    
    for (; nn > 0; nn--) {
        float32x4_t m1 = vld1q_f32(a);
        float32x4_t m2 = vld1q_f32(b);
        sum0 = vmuladdq_f32(sum0, m1, m2);

        m1 = vld1q_f32(a + 4);
        m2 = vld1q_f32(b + 4);
        sum1 = vmuladdq_f32(sum1, m1, m2);
        
        a += 8;
        b += 8;
    }
    
    for (; remain > 0; remain--) sum += *a++ * *b++;
    
    sum0 = vaddq_f32(sum0, sum1);
    
#if __aarch64__
    sum += vaddvq_f32(sum0);
#else
    float32x2_t sumss = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
    sumss = vpadd_f32(sumss, sumss);
    sum += vget_lane_f32(sumss, 0);
#endif
    
    *c = sum * alpha + *c * beta;
}
