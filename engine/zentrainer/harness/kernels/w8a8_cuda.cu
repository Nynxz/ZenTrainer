// The W8A8 input gradient of a frozen int8 ConvRot linear, as two CUDA kernels.
//
//   quantize_rows: gs = grad * s (s: the weight's per-output-row scale), each row's step =
//                  max|gs| / 127, gq = floor(gs / step + U[0, 1)) (stochastic rounding), written
//                  with each 16 columns in the order the GEMM's B fragments come out of ldmatrix.
//                  Single-pass register-cached variants cover N <= 16384 without re-reading grad
//                  from global memory; larger N uses the two-pass kernel.
//   gemm:          out = H((gq @ q) * step) (+ residual), int8 tensor cores (mma m16n8k32), int32 accumulation;
//                  q is the weight as stored, (out, in) row-major: its reduction dim is strided, read
//                  with ldmatrix.trans in 16-bit pairs and split into bytes with two prmt each.
//                  H: ConvRot's rotation, the regular Hadamard H4 x H4 x H4 x H4 / 16 on each 256
//                  output columns (a CTA's tile), as a fast transform in fp32 (padded shared memory
//                  for conflict-free digit-3 exchange, named half-CTA barriers, hardware bf16x2
//                  conversion, and coalesced global residual/output access); then bf16 once.
//                  Two M-tiles: BM=128 (3- or 4-stage, 256 threads) for large token counts, plus
//                  lock-free two-phase Stream-K to eliminate tail wave quantization on high-SM GPUs,
//                  and BM=64 (2-stage, 128 threads, 2 CTAs/SM) for very small token counts.
//
// Compiled at run time with NVRTC (kernels/w8a8_cuda.py): no compiler needed beside PyTorch's.
// The 16-column order: ldmatrix.trans hands each lane two adjacent weight columns at reductions
// 2t, 2t+1 (t = lane % 4), and the next matrix 8 rows down; one prmt picks a column's 4 bytes out of
// the pair of registers, so a fragment's reductions are {2t, 2t+1, 2t+8, 2t+9}, not 4t..4t+3. The
// gradient's quantized row is stored in that order (position 4t + j holds column 2t + (j & 1) +
// 8 (j >> 1)), so its fragments match without a transpose of either side. The two columns a lane
// holds land in two mma tiles (even and odd columns): the epilogue writes them back in place.

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned short u16;

#define BK 256
#define BR 64
#define GROUP 8

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(u32 dst, const void* src, bool pred) {
    int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// Non-volatile register-only MMA so ptxas can overlap PRMT and ldmatrix with IMMA issue.
__device__ __forceinline__ void mma_s8(int* c, const u32* a, u32 b0, u32 b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// H4 regular: out_k = sum - 2 x_(3-k)
__device__ __forceinline__ void h4(float& a, float& b, float& c, float& d) {
    float s = a + b + c + d;
    float na = s - 2.f * d, nb = s - 2.f * c, nc = s - 2.f * b, nd = s - 2.f * a;
    a = na; b = nb; c = nc; d = nd;
}

__device__ __forceinline__ u32 hash32(u32 x) {  // lowbias32 (C. Wellons)
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float bf16_to_f(u16 v) { return __uint_as_float(((u32)v) << 16); }
// Hardware sm_80+ round-to-nearest-even conversion of two floats to packed bf16x2 (lo = a, hi = b).
__device__ __forceinline__ u32 f2_to_bf16x2(float a, float b) {
    u32 out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ uint4 pack_quant16(const float x[16], float inv, u32 seed, int c0) {
    u32 out[4];
#pragma unroll
    for (int p = 0; p < 16; p++) {  // stored position p = 4t + j holds column 2t + (j & 1) + 8 (j >> 1)
        const int t = p >> 2, j = p & 3;
        const int c = 2 * t + (j & 1) + 8 * (j >> 1);
        const float u = (float)(hash32(seed ^ (u32)(c0 + c) * 0x85ebca6bU) >> 8) * (1.f / 16777216.f);
        float r = fminf(fmaxf(floorf(x[c] * inv + u), -127.f), 127.f);
        const u32 b = (u32)(u8)(signed char)(int)r;
        if (j == 0) out[t] = b; else out[t] |= b << (8 * j);
    }
    return make_uint4(out[0], out[1], out[2], out[3]);
}

// single-pass register-cached quantizer for N <= THREADS * 16 * MAX_CHUNKS
template <int THREADS, int MAX_CHUNKS>
__device__ __forceinline__ void quantize_rows_1pass(
        const u16* __restrict__ grad, const float* __restrict__ scale, signed char* __restrict__ q,
        float* __restrict__ steps, const int* __restrict__ seed_p, int N, int stride_g) {
    const int row = blockIdx.x;
    const u16* g = grad + (long long)row * stride_g;
    constexpr int WARPS = THREADS / 32;
    __shared__ float red[WARPS];
    float cache[MAX_CHUNKS][16];
    float amax = 0.f;
#pragma unroll
    for (int ch = 0; ch < MAX_CHUNKS; ch++) {
        const int c0 = (ch * THREADS + threadIdx.x) * 16;
        if (c0 < N) {
#pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                uint4 v = *reinterpret_cast<const uint4*>(g + c0 + 8 * h2);
                const u16* h = reinterpret_cast<const u16*>(&v);
                float4 s0 = *reinterpret_cast<const float4*>(scale + c0 + 8 * h2);
                float4 s1 = *reinterpret_cast<const float4*>(scale + c0 + 8 * h2 + 4);
                float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    float val = bf16_to_f(h[j]) * sv[j];
                    cache[ch][8 * h2 + j] = val;
                    amax = fmaxf(amax, fabsf(val));
                }
            }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < WARPS; w++) amax = fmaxf(amax, red[w]);
    const float step = fmaxf(amax, 1e-30f) / 127.f, inv = 1.f / step;
    if (threadIdx.x == 0) steps[row] = step;
    const u32 seed = hash32((u32)seed_p[0] ^ hash32((u32)row * 0x9E3779B9U));
#pragma unroll
    for (int ch = 0; ch < MAX_CHUNKS; ch++) {
        const int c0 = (ch * THREADS + threadIdx.x) * 16;
        if (c0 < N) {
            *reinterpret_cast<uint4*>(q + (long long)row * N + c0) = pack_quant16(cache[ch], inv, seed, c0);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) zt_quantize_rows_2k(
        const u16* __restrict__ grad, const float* __restrict__ scale, signed char* __restrict__ q,
        float* __restrict__ steps, const int* __restrict__ seed_p, int N, int stride_g) {
    quantize_rows_1pass<128, 1>(grad, scale, q, steps, seed_p, N, stride_g);
}
extern "C" __global__ void __launch_bounds__(256) zt_quantize_rows_4k(
        const u16* __restrict__ grad, const float* __restrict__ scale, signed char* __restrict__ q,
        float* __restrict__ steps, const int* __restrict__ seed_p, int N, int stride_g) {
    quantize_rows_1pass<256, 1>(grad, scale, q, steps, seed_p, N, stride_g);
}
extern "C" __global__ void __launch_bounds__(256) zt_quantize_rows_8k(
        const u16* __restrict__ grad, const float* __restrict__ scale, signed char* __restrict__ q,
        float* __restrict__ steps, const int* __restrict__ seed_p, int N, int stride_g) {
    quantize_rows_1pass<256, 2>(grad, scale, q, steps, seed_p, N, stride_g);
}
extern "C" __global__ void __launch_bounds__(512) zt_quantize_rows_16k(
        const u16* __restrict__ grad, const float* __restrict__ scale, signed char* __restrict__ q,
        float* __restrict__ steps, const int* __restrict__ seed_p, int N, int stride_g) {
    quantize_rows_1pass<512, 2>(grad, scale, q, steps, seed_p, N, stride_g);
}

// two-pass fallback for arbitrary N > 16384 (and direct test entry point)
extern "C" __global__ void __launch_bounds__(256) zt_quantize_rows(
        const u16* __restrict__ grad, const float* __restrict__ scale, signed char* __restrict__ q,
        float* __restrict__ steps, const int* __restrict__ seed_p, int N, int stride_g) {  // ints: the launcher passes C ints
    const int row = blockIdx.x;
    const u16* g = grad + (long long)row * stride_g;
    __shared__ float red[8];
    float amax = 0.f;
    for (int c0 = threadIdx.x * 8; c0 < N; c0 += 256 * 8) {
        uint4 v = *reinterpret_cast<const uint4*>(g + c0);
        const u16* h = reinterpret_cast<const u16*>(&v);
        float4 s0 = *reinterpret_cast<const float4*>(scale + c0), s1 = *reinterpret_cast<const float4*>(scale + c0 + 4);
        float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
        for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(bf16_to_f(h[j]) * sv[j]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < 8; w++) amax = fmaxf(amax, red[w]);
    const float step = fmaxf(amax, 1e-30f) / 127.f, inv = 1.f / step;
    if (threadIdx.x == 0) steps[row] = step;
    const u32 seed = hash32((u32)seed_p[0] ^ hash32((u32)row * 0x9E3779B9U));
    for (int c0 = threadIdx.x * 16; c0 < N; c0 += 256 * 16) {
        float x[16];
#pragma unroll
        for (int h2 = 0; h2 < 2; h2++) {
            uint4 v = *reinterpret_cast<const uint4*>(g + c0 + 8 * h2);
            const u16* h = reinterpret_cast<const u16*>(&v);
            float4 s0 = *reinterpret_cast<const float4*>(scale + c0 + 8 * h2);
            float4 s1 = *reinterpret_cast<const float4*>(scale + c0 + 8 * h2 + 4);
            float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) x[8 * h2 + j] = bf16_to_f(h[j]) * sv[j];
        }
        *reinterpret_cast<uint4*>(q + (long long)row * N + c0) = pack_quant16(x, inv, seed, c0);
    }
}

// Grouped launch order: GROUP row bands at a time walk the weight's column tiles together,
// so tiles in flight share the weight in L2.
__device__ __forceinline__ void tile_coords(int tile, int n_m, int n_k, int BM, int& m0, int& k0) {
    const int per = GROUP * n_k;
    const int first = (tile / per) * GROUP;
    const int rows = min(n_m - first, GROUP);
    const int rem = tile % per;
    m0 = (first + rem % rows) * BM;
    k0 = (rem / rows) * BK;
}

// Software-pipelined GEMM K-loop over reduction tiles [k_start, k_end).
// Double-buffers A fragments across ks=0/1 and streams B's ldmatrix.trans into a 2-slot ring
// so every SMEM load and byte-permute overlaps the 8 independent mma_s8 instructions of the prior slot.
template <int BM, int WARPS_M, int STAGES>
__device__ __forceinline__ void gemm_kloop(
        const signed char* __restrict__ A, const signed char* __restrict__ B,
        u8* sA, u8* sB, int acc[4][8][4],
        int m0, int k0, int k_start, int k_end, int M, int N, int K,
        int tid, int lane, int wm, int wn) {
    constexpr int A_STAGE = BM * BR;
    constexpr int B_STAGE = BR * BK;
    constexpr int THREADS = WARPS_M * 4 * 32;
    constexpr int MI = (BM / WARPS_M) / 16;  // 4 (each warp holds 64 x 64)
    constexpr int A_ITERS = (BM * (BR / 16)) / THREADS;
    constexpr int B_ITERS = (BR * (BK / 16)) / THREADS;

    auto load_stage = [&](int stage, int kt) {
        const int r0 = kt * BR;
        u8* a = sA + stage * A_STAGE;
        u8* b = sB + stage * B_STAGE;
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {  // A: BM rows x 4 chunks of 16 B
            const int c = tid + i * THREADS, row = c >> 2, ch = c & 3;
            const int gm = m0 + row;
            const signed char* src = A + (long long)(gm < M ? gm : 0) * N + r0 + ch * 16;
            cp_async16(smem_u32(a + row * BR + ((ch ^ ((row >> 1) & 3)) << 4)), src, gm < M);
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; i++) {  // B: 64 rows x 16 chunks of 16 B
            const int c = tid + i * THREADS, row = c >> 4, ch = c & 15;
            const signed char* src = B + (long long)(r0 + row) * K + k0 + ch * 16;
            cp_async16(smem_u32(b + row * BK + ((ch ^ (row & 7)) << 4)), src, true);
        }
    };

    const int n_iters = k_end - k_start;
    if (n_iters <= 0) return;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < n_iters) load_stage(s, k_start + s);
        cp_commit();
    }

    const int mi_l = lane >> 3, l7 = lane & 7;
    u32 a_off0[MI], b_off0[4];
#pragma unroll
    for (int i = 0; i < MI; i++) {
        const int row = wm * 64 + i * 16 + (mi_l & 1) * 8 + l7;
        const int ch = mi_l >> 1;
        a_off0[i] = (u32)(row * BR + ((ch ^ ((row >> 1) & 3)) << 4));
    }
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const int row = mi_l * 8 + l7;
        const int ch = wn * 4 + q;
        b_off0[q] = (u32)(row * BK + ((ch ^ (row & 7)) << 4));
    }

    const u32 sA_base = smem_u32(sA);
    const u32 sB_base = smem_u32(sB);

    cp_wait<STAGES - 2>();
    __syncthreads();

    u32 af[2][MI][4];
    u32 br_raw[2][4];
#pragma unroll
    for (int i = 0; i < MI; i++) ldsm_x4(sA_base + a_off0[i], af[0][i][0], af[0][i][1], af[0][i][2], af[0][i][3]);
    ldsm_x4_t(sB_base + b_off0[0], br_raw[0][0], br_raw[0][1], br_raw[0][2], br_raw[0][3]);

    for (int it = 0; it < n_iters; it++) {
        const int cur_s = it % STAGES;
        const u32 a_s = sA_base + cur_s * A_STAGE;
        const u32 b_s = sB_base + cur_s * B_STAGE;

        const int nxt = it + STAGES - 1;
        if (nxt < n_iters) load_stage(nxt % STAGES, k_start + nxt);
        cp_commit();

        // ks=1 A chunk is ks=0's chunk XOR 2 -> byte offset XOR 32
#pragma unroll
        for (int i = 0; i < MI; i++) ldsm_x4(a_s + (a_off0[i] ^ 32u), af[1][i][0], af[1][i][1], af[1][i][2], af[1][i][3]);

        // ks = 0: overlap each q's MMAs with the next q's ldmatrix.trans
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int cur = q & 1, nxt_b = cur ^ 1;
            if (q < 3) {
                ldsm_x4_t(b_s + b_off0[q + 1], br_raw[nxt_b][0], br_raw[nxt_b][1], br_raw[nxt_b][2], br_raw[nxt_b][3]);
            } else {
                ldsm_x4_t(b_s + b_off0[0] + 32 * BK, br_raw[nxt_b][0], br_raw[nxt_b][1], br_raw[nxt_b][2], br_raw[nxt_b][3]);
            }
            const u32 e0 = __byte_perm(br_raw[cur][0], br_raw[cur][1], 0x6420);  // even column, k 0-15
            const u32 e1 = __byte_perm(br_raw[cur][2], br_raw[cur][3], 0x6420);  // even column, k 16-31
            const u32 o0 = __byte_perm(br_raw[cur][0], br_raw[cur][1], 0x7531);  // odd column
            const u32 o1 = __byte_perm(br_raw[cur][2], br_raw[cur][3], 0x7531);
#pragma unroll
            for (int i = 0; i < MI; i++) {
                mma_s8(acc[i][2 * q], af[0][i], e0, e1);
                mma_s8(acc[i][2 * q + 1], af[0][i], o0, o1);
            }
        }

        // ks = 1: preload next stage's ks=0 right before the final q=3 MMAs
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int cur = q & 1, nxt_b = cur ^ 1;
            if (q < 3) {
                ldsm_x4_t(b_s + b_off0[q + 1] + 32 * BK, br_raw[nxt_b][0], br_raw[nxt_b][1], br_raw[nxt_b][2], br_raw[nxt_b][3]);
            }
            const u32 e0 = __byte_perm(br_raw[cur][0], br_raw[cur][1], 0x6420);
            const u32 e1 = __byte_perm(br_raw[cur][2], br_raw[cur][3], 0x6420);
            const u32 o0 = __byte_perm(br_raw[cur][0], br_raw[cur][1], 0x7531);
            const u32 o1 = __byte_perm(br_raw[cur][2], br_raw[cur][3], 0x7531);
            if (q == 3 && it + 1 < n_iters) {
                cp_wait<STAGES - 2>();
                __syncthreads();
                const int ns = (it + 1) % STAGES;
                const u32 na_s = sA_base + ns * A_STAGE;
                const u32 nb_s = sB_base + ns * B_STAGE;
#pragma unroll
                for (int i = 0; i < MI; i++) ldsm_x4(na_s + a_off0[i], af[0][i][0], af[0][i][1], af[0][i][2], af[0][i][3]);
                ldsm_x4_t(nb_s + b_off0[0], br_raw[0][0], br_raw[0][1], br_raw[0][2], br_raw[0][3]);
            }
#pragma unroll
            for (int i = 0; i < MI; i++) {
                mma_s8(acc[i][2 * q], af[1][i], e0, e1);
                mma_s8(acc[i][2 * q + 1], af[1][i], o0, o1);
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
}

// Epilogue: 16 rows per strip (i = 0..3) keeps live floats to 32 regs instead of 128;
// +4 floats padding per SMEM row removes the 8-way bank conflict in digit 3 and stages
// coalesced global residual loads and output stores. Named half-CTA barriers decouple wm=0/1.
template <int BM, int WARPS_M, bool ROTATE>
__device__ __forceinline__ void gemm_epilogue(
        int acc[4][8][4], u8* smem, const float* __restrict__ steps,
        u16* __restrict__ out, const u16* __restrict__ R, int has_res,
        int m0, int k0, int M, int K, int lane, int wm, int wn) {
    constexpr int MI = (BM / WARPS_M) / 16;  // 4
    const int g = lane >> 2, tig = lane & 3;
    const int half_w = lane >> 4, l16 = lane & 15;
    constexpr int STRIDE = 260;
    float* x = reinterpret_cast<float*>(smem);  // [WARPS_M][16][260] (33.3 KB for BM=128, 16.6 KB for BM=64)
    auto half_sync = [&]() {
        if constexpr (WARPS_M == 2) {
            asm volatile("bar.sync %0, 128;" :: "r"(wm + 1) : "memory");
        } else {
            __syncthreads();
        }
    };

#pragma unroll
    for (int i = 0; i < MI; i++) {
        float v[4][2][4];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int gm = m0 + wm * 64 + i * 16 + g + 8 * h;
            const float st = (gm < M ? steps[gm] : 0.f) * (ROTATE ? 1.f / 16.f : 1.f);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                v[q][h][0] = (float)acc[i][2 * q][2 * h] * st;
                v[q][h][1] = (float)acc[i][2 * q + 1][2 * h] * st;
                v[q][h][2] = (float)acc[i][2 * q][2 * h + 1] * st;
                v[q][h][3] = (float)acc[i][2 * q + 1][2 * h + 1] * st;
            }
        }
        if constexpr (ROTATE) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
#pragma unroll
                for (int q = 0; q < 4; q++) h4(v[q][h][0], v[q][h][1], v[q][h][2], v[q][h][3]);  // digit 0
#pragma unroll
                for (int d = 0; d < 4; d++) h4(v[0][h][d], v[1][h][d], v[2][h][d], v[3][h][d]);  // digit 2
#pragma unroll
                for (int q = 0; q < 4; q++)
#pragma unroll
                    for (int d = 0; d < 4; d++) {  // digit 1: across the quad's lanes
                        const float val = v[q][h][d];
                        float s = val + __shfl_xor_sync(0xffffffffu, val, 1);
                        s += __shfl_xor_sync(0xffffffffu, s, 2);
                        v[q][h][d] = s - 2.f * __shfl_xor_sync(0xffffffffu, val, 3);
                    }
            }
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int q = 0; q < 4; q++)
                    *reinterpret_cast<float4*>(x + (wm * 16 + g + 8 * h) * STRIDE + wn * 64 + q * 16 + tig * 4) =
                        make_float4(v[q][h][0], v[q][h][1], v[q][h][2], v[q][h][3]);
            half_sync();
            // digit 3 + coalesced residual/output: 16 lanes walk 64 cols (x4 quarters = 256 cols), 2 rows/warp
#pragma unroll
            for (int iter = 0; iter < 2; iter++) {
                const int lr = iter * 8 + wn * 2 + half_w;
                const int gm = m0 + wm * 64 + i * 16 + lr;
                if (gm < M) {
                    const float* p = x + (wm * 16 + lr) * STRIDE + l16 * 4;
                    const float4 a = *reinterpret_cast<const float4*>(p);
                    const float4 b = *reinterpret_cast<const float4*>(p + 64);
                    const float4 c = *reinterpret_cast<const float4*>(p + 128);
                    const float4 d = *reinterpret_cast<const float4*>(p + 192);
                    const float4 sum = make_float4(a.x + b.x + c.x + d.x, a.y + b.y + c.y + d.y,
                                                   a.z + b.z + c.z + d.z, a.w + b.w + c.w + d.w);
                    float4 o[4] = {
                        make_float4(sum.x - 2.f * d.x, sum.y - 2.f * d.y, sum.z - 2.f * d.z, sum.w - 2.f * d.w),
                        make_float4(sum.x - 2.f * c.x, sum.y - 2.f * c.y, sum.z - 2.f * c.z, sum.w - 2.f * c.w),
                        make_float4(sum.x - 2.f * b.x, sum.y - 2.f * b.y, sum.z - 2.f * b.z, sum.w - 2.f * b.w),
                        make_float4(sum.x - 2.f * a.x, sum.y - 2.f * a.y, sum.z - 2.f * a.z, sum.w - 2.f * a.w),
                    };
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const int gk = k0 + q * 64 + l16 * 4;
                        if (has_res) {
                            const uint2 rv = *reinterpret_cast<const uint2*>(R + (long long)gm * K + gk);
                            o[q].x += bf16_to_f((u16)(rv.x & 0xffffu));
                            o[q].y += bf16_to_f((u16)(rv.x >> 16));
                            o[q].z += bf16_to_f((u16)(rv.y & 0xffffu));
                            o[q].w += bf16_to_f((u16)(rv.y >> 16));
                        }
                        uint2 ov;
                        ov.x = f2_to_bf16x2(o[q].x, o[q].y);
                        ov.y = f2_to_bf16x2(o[q].z, o[q].w);
                        *reinterpret_cast<uint2*>(out + (long long)gm * K + gk) = ov;
                    }
                }
            }
            if (i < MI - 1) half_sync();
        } else {
            // stage through SMEM for coalesced 128-byte stores (32 lanes x 8 bf16 = 512 B/warp)
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int q = 0; q < 4; q++)
                    *reinterpret_cast<float4*>(x + (wm * 16 + g + 8 * h) * STRIDE + wn * 64 + q * 16 + tig * 4) =
                        make_float4(v[q][h][0], v[q][h][1], v[q][h][2], v[q][h][3]);
            half_sync();
#pragma unroll
            for (int iter = 0; iter < 4; iter++) {
                const int lr = wn * 4 + iter;
                const int gm = m0 + wm * 64 + i * 16 + lr;
                if (gm < M) {
                    const int gk = k0 + lane * 8;
                    const float4 f0 = *reinterpret_cast<const float4*>(x + (wm * 16 + lr) * STRIDE + lane * 8);
                    const float4 f1 = *reinterpret_cast<const float4*>(x + (wm * 16 + lr) * STRIDE + lane * 8 + 4);
                    float vals[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
                    if (has_res) {
                        const uint4 rv = *reinterpret_cast<const uint4*>(R + (long long)gm * K + gk);
                        const u16* rh = reinterpret_cast<const u16*>(&rv);
#pragma unroll
                        for (int j = 0; j < 8; j++) vals[j] += bf16_to_f(rh[j]);
                    }
                    uint4 ov;
                    ov.x = f2_to_bf16x2(vals[0], vals[1]);
                    ov.y = f2_to_bf16x2(vals[2], vals[3]);
                    ov.z = f2_to_bf16x2(vals[4], vals[5]);
                    ov.w = f2_to_bf16x2(vals[6], vals[7]);
                    *reinterpret_cast<uint4*>(out + (long long)gm * K + gk) = ov;
                }
            }
            if (i < MI - 1) half_sync();
        }
    }
}

// out (M, K) bf16 = H((A @ B) * step) + R: A (M, N) int8 from zt_quantize_rows, B (N, K) int8 row-major,
// R (M, K) bf16 when has_res. N % 64 == 0, K % 256 == 0.
template <int BM, int WARPS_M, int STAGES, bool ROTATE>
__device__ __forceinline__ void gemm_body(const signed char* __restrict__ A, const signed char* __restrict__ B,
                                          const float* __restrict__ steps, u16* __restrict__ out,
                                          const u16* __restrict__ R, int has_res, int M, int N, int K) {
    constexpr int A_STAGE = BM * BR;
    extern __shared__ __align__(128) u8 smem[];
    u8* sA = smem;
    u8* sB = smem + STAGES * A_STAGE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const int n_m = (M + BM - 1) / BM, n_k = K / BK;
    int m0, k0;
    tile_coords(blockIdx.x, n_m, n_k, BM, m0, k0);
    int acc[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
    gemm_kloop<BM, WARPS_M, STAGES>(A, B, sA, sB, acc, m0, k0, 0, N / BR, M, N, K, tid, lane, wm, wn);
    gemm_epilogue<BM, WARPS_M, ROTATE>(acc, smem, steps, out, R, has_res, m0, k0, M, K, lane, wm, wn);
}

// Two-phase lock-free Stream-K GEMM (no spin-wait, no memset, bit-exact int32 reduction):
//   Phase 1: dp_tiles (a multiple of gridDim.x) run standard zero-overhead data-parallel tiles.
//   Phase 2: remaining sk_tiles are partitioned evenly across all gridDim.x CTAs at BR=64 granularity.
//   Requires sk_tiles * (N / BR) >= gridDim.x so every CTA owns >= 1 iteration and peers is exact.
//   Each split tile's CTAs store their registers to partials[cta * 2 + (k_start == 0)] and increment
//   done[sk_idx]; the last finisher resets done[sk_idx] = 0, sums peers' int32 partials, and runs
//   the normal Hadamard + residual + bf16 epilogue.
template <int BM, int WARPS_M, int STAGES, bool ROTATE>
__device__ __forceinline__ void gemm_sk_body(const signed char* __restrict__ A, const signed char* __restrict__ B,
                                             const float* __restrict__ steps, u16* __restrict__ out,
                                             const u16* __restrict__ R, int has_res, int M, int N, int K,
                                             int* __restrict__ partials, int* __restrict__ done, int dp_tiles) {
    constexpr int A_STAGE = BM * BR;
    constexpr int THREADS = WARPS_M * 4 * 32;
    constexpr int TILE_INTS = THREADS * 128;  // 32768 ints (128 KB) per BM=128 tile
    extern __shared__ __align__(128) u8 smem[];
    u8* sA = smem;
    u8* sB = smem + STAGES * A_STAGE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const int cta = blockIdx.x, num_ctas = gridDim.x;
    const int n_m = (M + BM - 1) / BM, n_k = K / BK;
    const int total_tiles = n_m * n_k;
    const int KT = N / BR;

    for (int tile = cta; tile < dp_tiles; tile += num_ctas) {
        int m0, k0;
        tile_coords(tile, n_m, n_k, BM, m0, k0);
        int acc[4][8][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
        gemm_kloop<BM, WARPS_M, STAGES>(A, B, sA, sB, acc, m0, k0, 0, KT, M, N, K, tid, lane, wm, wn);
        gemm_epilogue<BM, WARPS_M, ROTATE>(acc, smem, steps, out, R, has_res, m0, k0, M, K, lane, wm, wn);
        __syncthreads();
    }

    const int sk_tiles = total_tiles - dp_tiles;
    if (sk_tiles <= 0) return;
    const long long total_iters = (long long)sk_tiles * KT;
    const long long iter_start = ((long long)cta * total_iters) / num_ctas;
    const long long iter_end = ((long long)(cta + 1) * total_iters) / num_ctas;
    __shared__ int s_arrived;

    long long it = iter_start;
    while (it < iter_end) {
        const int sk_idx = (int)(it / KT);
        const int tile = dp_tiles + sk_idx;
        const int k_start = (int)(it % KT);
        const long long tile_end_iter = (long long)(sk_idx + 1) * KT;
        const int k_end = (int)(min(iter_end, tile_end_iter) - (long long)sk_idx * KT);

        int m0, k0;
        tile_coords(tile, n_m, n_k, BM, m0, k0);
        int acc[4][8][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

        gemm_kloop<BM, WARPS_M, STAGES>(A, B, sA, sB, acc, m0, k0, k_start, k_end, M, N, K, tid, lane, wm, wn);

        if (k_start == 0 && k_end == KT) {
            gemm_epilogue<BM, WARPS_M, ROTATE>(acc, smem, steps, out, R, has_res, m0, k0, M, K, lane, wm, wn);
            if (it + (k_end - k_start) < iter_end) __syncthreads();
        } else {
            const long long t_lo = (long long)sk_idx * KT;
            const long long t_hi = t_lo + KT - 1;
            const int first_cta = (int)(((t_lo + 1) * num_ctas - 1) / total_iters);
            const int last_cta = (int)(((t_hi + 1) * num_ctas - 1) / total_iters);
            const int peers = last_cta - first_cta + 1;

            const int my_slot = cta * 2 + (k_start == 0 ? 1 : 0);
            int4* dst = reinterpret_cast<int4*>(partials + (long long)my_slot * TILE_INTS);
            int idx = 0;
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    dst[idx * THREADS + tid] = make_int4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                    idx++;
                }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                int prev;
                asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                             : "=r"(prev) : "l"(done + sk_idx) : "memory");
                s_arrived = prev + 1;
            }
            __syncthreads();
            if (s_arrived == peers) {
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                if (tid == 0) done[sk_idx] = 0;
                for (int p = first_cta; p <= last_cta; p++) {
                    if (p == cta) continue;
                    const int p_slot = p * 2 + (p == first_cta ? 1 : 0);
                    const int4* src = reinterpret_cast<const int4*>(partials + (long long)p_slot * TILE_INTS);
                    int sidx = 0;
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            int v0, v1, v2, v3;
                            asm volatile("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                                         : "l"(src + sidx * THREADS + tid));
                            acc[i][j][0] += v0;
                            acc[i][j][1] += v1;
                            acc[i][j][2] += v2;
                            acc[i][j][3] += v3;
                            sidx++;
                        }
                }
                gemm_epilogue<BM, WARPS_M, ROTATE>(acc, smem, steps, out, R, has_res, m0, k0, M, K, lane, wm, wn);
                if (it + (k_end - k_start) < iter_end) __syncthreads();
            }
        }
        it += (k_end - k_start);
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm_rot(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K) {
    gemm_body<128, 2, 3, true>(A, B, steps, out, R, has_res, M, N, K);
}
extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K) {
    gemm_body<128, 2, 3, false>(A, B, steps, out, R, has_res, M, N, K);
}
extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm_s4_rot(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K) {
    gemm_body<128, 2, 4, true>(A, B, steps, out, R, has_res, M, N, K);
}
extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm_s4(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K) {
    gemm_body<128, 2, 4, false>(A, B, steps, out, R, has_res, M, N, K);
}
extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm_sk_rot(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K,
        int* partials, int* done, int dp_tiles) {
    gemm_sk_body<128, 2, 3, true>(A, B, steps, out, R, has_res, M, N, K, partials, done, dp_tiles);
}
extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm_sk(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K,
        int* partials, int* done, int dp_tiles) {
    gemm_sk_body<128, 2, 3, false>(A, B, steps, out, R, has_res, M, N, K, partials, done, dp_tiles);
}
extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm_s4_sk_rot(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K,
        int* partials, int* done, int dp_tiles) {
    gemm_sk_body<128, 2, 4, true>(A, B, steps, out, R, has_res, M, N, K, partials, done, dp_tiles);
}
extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm_s4_sk(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K,
        int* partials, int* done, int dp_tiles) {
    gemm_sk_body<128, 2, 4, false>(A, B, steps, out, R, has_res, M, N, K, partials, done, dp_tiles);
}
extern "C" __global__ void __launch_bounds__(128, 2) zt_w8a8_gemm64_rot(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K) {
    gemm_body<64, 1, 2, true>(A, B, steps, out, R, has_res, M, N, K);
}
extern "C" __global__ void __launch_bounds__(128, 2) zt_w8a8_gemm64(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K) {
    gemm_body<64, 1, 2, false>(A, B, steps, out, R, has_res, M, N, K);
}
