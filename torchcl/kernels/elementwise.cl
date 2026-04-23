// ═══════════════════════════════════════════════════════════════════
// TorchCL — Element-wise OpenCL Kernels
// Supports: add, sub, mul, div, neg, abs, exp, log, sqrt, clamp
// Each kernel operates on flat 1-D buffers (tensors are flattened).
// ═══════════════════════════════════════════════════════════════════

// ── Binary operations (a ⊕ b → c) ──────────────────────────────────

__kernel void add_f32(__global const float* a,
                      __global const float* b,
                      __global float* c,
                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = a[gid] + b[gid];
}

__kernel void sub_f32(__global const float* a,
                      __global const float* b,
                      __global float* c,
                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = a[gid] - b[gid];
}

__kernel void mul_f32(__global const float* a,
                      __global const float* b,
                      __global float* c,
                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = a[gid] * b[gid];
}

__kernel void div_f32(__global const float* a,
                      __global const float* b,
                      __global float* c,
                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = a[gid] / b[gid];
}

// ── Scalar operations (a ⊕ scalar → c) ─────────────────────────────

__kernel void add_scalar_f32(__global const float* a,
                             const float scalar,
                             __global float* c,
                             const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = a[gid] + scalar;
}

__kernel void mul_scalar_f32(__global const float* a,
                             const float scalar,
                             __global float* c,
                             const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = a[gid] * scalar;
}

// ── Unary operations (a → c) ────────────────────────────────────────

__kernel void neg_f32(__global const float* a,
                      __global float* c,
                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = -a[gid];
}

__kernel void abs_f32(__global const float* a,
                      __global float* c,
                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = fabs(a[gid]);
}

__kernel void exp_f32(__global const float* a,
                      __global float* c,
                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = exp(a[gid]);
}

__kernel void log_f32(__global const float* a,
                      __global float* c,
                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = log(a[gid]);
}

__kernel void sqrt_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = sqrt(a[gid]);
}

__kernel void clamp_f32(__global const float* a,
                        const float min_val,
                        const float max_val,
                        __global float* c,
                        const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = clamp(a[gid], min_val, max_val);
}

// ── Fill operations ─────────────────────────────────────────────────

__kernel void fill_f32(__global float* c,
                       const float val,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = val;
}

__kernel void copy_f32(__global const float* src,
                       __global float* dst,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) dst[gid] = src[gid];
}

// ── Comparison operations (a ? b → c as float: 1.0 or 0.0) ─────────

__kernel void eq_f32(__global const float* a,
                     __global const float* b,
                     __global float* c,
                     const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = (a[gid] == b[gid]) ? 1.0f : 0.0f;
}

__kernel void gt_f32(__global const float* a,
                     __global const float* b,
                     __global float* c,
                     const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = (a[gid] > b[gid]) ? 1.0f : 0.0f;
}

__kernel void lt_f32(__global const float* a,
                     __global const float* b,
                     __global float* c,
                     const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = (a[gid] < b[gid]) ? 1.0f : 0.0f;
}

__kernel void where_f32(__global const float* cond,
                        __global const float* x,
                        __global const float* y,
                        __global float* out,
                        const int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = (cond[gid] != 0.0f) ? x[gid] : y[gid];
}
