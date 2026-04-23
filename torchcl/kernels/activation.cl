// ═══════════════════════════════════════════════════════════════════
// TorchCL — Activation Function Kernels
// ReLU, Sigmoid, Tanh, GELU, LeakyReLU, SiLU, Softmax helper
// ═══════════════════════════════════════════════════════════════════

__kernel void relu_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = fmax(a[gid], 0.0f);
}

__kernel void relu_backward_f32(__global const float* grad_out,
                                __global const float* input,
                                __global float* grad_in,
                                const int n) {
    int gid = get_global_id(0);
    if (gid < n) grad_in[gid] = (input[gid] > 0.0f) ? grad_out[gid] : 0.0f;
}

__kernel void sigmoid_f32(__global const float* a,
                          __global float* c,
                          const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = 1.0f / (1.0f + exp(-a[gid]));
}

__kernel void sigmoid_backward_f32(__global const float* grad_out,
                                   __global const float* output,
                                   __global float* grad_in,
                                   const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float s = output[gid];
        grad_in[gid] = grad_out[gid] * s * (1.0f - s);
    }
}

__kernel void tanh_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = tanh(a[gid]);
}

__kernel void tanh_backward_f32(__global const float* grad_out,
                                __global const float* output,
                                __global float* grad_in,
                                const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float t = output[gid];
        grad_in[gid] = grad_out[gid] * (1.0f - t * t);
    }
}

__kernel void gelu_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float x = a[gid];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        c[gid] = x * cdf;
    }
}

__kernel void leaky_relu_f32(__global const float* a,
                             const float neg_slope,
                             __global float* c,
                             const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = (a[gid] >= 0.0f) ? a[gid] : neg_slope * a[gid];
}

__kernel void silu_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = a[gid] / (1.0f + exp(-a[gid]));
}
