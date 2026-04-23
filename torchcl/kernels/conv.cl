// ═══════════════════════════════════════════════════════════════════
// OjasX V2 — Convolution Kernels
// im2col-based Conv2d, BatchNorm, MaxPool2d, Broadcasting
// ═══════════════════════════════════════════════════════════════════

__kernel void im2col_f32(__global const float* input,
                         __global float* col,
                         const int C, const int H, const int W,
                         const int kH, const int kW,
                         const int padH, const int padW,
                         const int strH, const int strW,
                         const int outH, const int outW) {
    int gid = get_global_id(0);
    int total = C * kH * kW * outH * outW;
    if (gid >= total) return;

    int ow = gid % outW;
    int oh = (gid / outW) % outH;
    int kw = (gid / (outW * outH)) % kW;
    int kh = (gid / (outW * outH * kW)) % kH;
    int c  = gid / (outW * outH * kW * kH);

    int ih = oh * strH - padH + kh;
    int iw = ow * strW - padW + kw;

    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        val = input[c * H * W + ih * W + iw];

    col[(c * kH * kW + kh * kW + kw) * (outH * outW) + oh * outW + ow] = val;
}

__kernel void bias_add_f32(__global float* output,
                           __global const float* bias,
                           const int channels, const int spatial) {
    int gid = get_global_id(0);
    if (gid >= channels * spatial) return;
    output[gid] += bias[gid / spatial];
}

__kernel void maxpool2d_f32(__global const float* input,
                            __global float* output,
                            __global int* indices,
                            const int C, const int H, const int W,
                            const int kH, const int kW,
                            const int strH, const int strW,
                            const int outH, const int outW) {
    int gid = get_global_id(0);
    if (gid >= C * outH * outW) return;

    int ow = gid % outW;
    int oh = (gid / outW) % outH;
    int c  = gid / (outW * outH);

    float max_val = -INFINITY;
    int max_idx = 0;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int ih = oh * strH + kh;
            int iw = ow * strW + kw;
            if (ih < H && iw < W) {
                int idx = c * H * W + ih * W + iw;
                float val = input[idx];
                if (val > max_val) { max_val = val; max_idx = idx; }
            }
        }
    }
    output[gid] = max_val;
    indices[gid] = max_idx;
}

__kernel void add_broadcast_f32(__global const float* a,
                                __global const float* b,
                                __global float* c,
                                const int M, const int N) {
    int gid = get_global_id(0);
    if (gid >= M * N) return;
    c[gid] = a[gid] + b[gid % N];
}
