// The W8A8 input gradient of a frozen int8 ConvRot linear, as two CUDA kernels.
//
//   quantize_rows: gs = grad * s (s: the weight's per-output-row scale), each row's step =
//                  max|gs| / 127, gq = floor(gs / step + U[0, 1)) (stochastic rounding), written
//                  with each 16 columns in the order the GEMM's B fragments come out of ldmatrix
//   gemm:          out = H((gq @ q) * step) (+ residual), int8 tensor cores (mma m16n8k32), int32 accumulation;
//                  q is the weight as stored, (out, in) row-major: its reduction dim is strided, read
//                  with ldmatrix.trans in 16-bit pairs and split into bytes with two prmt each.
//                  H: ConvRot's rotation, the regular Hadamard H4 x H4 x H4 x H4 / 16 on each 256
//                  output columns (a CTA's tile), as a fast transform in fp32; then the residual
//                  (the LoRA's input gradient beside it: no separate add) and bf16 once.
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

#define BM 128
#define BK 256
#define BR 64
#define STAGES 3
#define GROUP 8
#define A_STAGE (BM * BR)
#define B_STAGE (BR * BK)

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
__device__ __forceinline__ void mma_s8(int* c, const u32* a, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
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
__device__ __forceinline__ u16 f_to_bf16(float f) {  // round to nearest even
    u32 u = __float_as_uint(f);
    u += 0x7fffU + ((u >> 16) & 1U);
    return (u16)(u >> 16);
}

// one block a row; each thread 16 columns at a time
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
        *reinterpret_cast<uint4*>(q + (long long)row * N + c0) = make_uint4(out[0], out[1], out[2], out[3]);
    }
}

// out (M, K) bf16 = H((A @ B) * step) + R: A (M, N) int8 from zt_quantize_rows, B (N, K) int8 row-major,
// R (M, K) bf16 when has_res.
// N % 64 == 0, K % 256 == 0. 256 threads: 2 x 4 warps of 64 x 64.
template <bool ROTATE>
__device__ __forceinline__ void gemm_body(const signed char* __restrict__ A, const signed char* __restrict__ B,
                                          const float* __restrict__ steps, u16* __restrict__ out,
                                          const u16* __restrict__ R, int has_res, int M, int N, int K) {
    extern __shared__ __align__(128) u8 smem[];
    u8* sA = smem;
    u8* sB = smem + STAGES * A_STAGE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    // grouped launch order: GROUP row bands at a time walk the weight's column tiles together, so
    // the tiles in flight share the weight in L2 (a band at a time re-read all of it per band)
    const int n_m = (M + BM - 1) / BM, n_k = K / BK, per = GROUP * n_k;
    const int pid = blockIdx.x, first = (pid / per) * GROUP, rows = min(n_m - first, GROUP);
    const int m0 = (first + (pid % per) % rows) * BM, k0 = ((pid % per) / rows) * BK;
    const int KT = N / BR;

    auto load = [&](int stage, int kt) {
        const int r0 = kt * BR;
        u8* a = sA + stage * A_STAGE;
        u8* b = sB + stage * B_STAGE;
#pragma unroll
        for (int i = 0; i < 2; i++) {  // A: 128 rows x 4 chunks of 16 B
            const int c = tid + i * 256, row = c >> 2, ch = c & 3;
            const int gm = m0 + row;
            const signed char* src = A + (long long)(gm < M ? gm : 0) * N + r0 + ch * 16;
            cp_async16(smem_u32(a + row * BR + ((ch ^ ((row >> 1) & 3)) << 4)), src, gm < M);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {  // B: 64 rows x 16 chunks of 16 B
            const int c = tid + i * 256, row = c >> 4, ch = c & 15;
            const signed char* src = B + (long long)(r0 + row) * K + k0 + ch * 16;
            cp_async16(smem_u32(b + row * BK + ((ch ^ (row & 7)) << 4)), src, true);
        }
    };

    int acc[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < KT) load(s, s);
        cp_commit();
    }

    for (int kt = 0; kt < KT; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int nxt = kt + STAGES - 1;
        if (nxt < KT) load(nxt % STAGES, nxt);
        cp_commit();
        const u8* a = sA + (kt % STAGES) * A_STAGE;
        const u8* b = sB + (kt % STAGES) * B_STAGE;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            u32 af[4][4], bf[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int mi = lane >> 3;
                const int row = wm * 64 + i * 16 + (mi & 1) * 8 + (lane & 7);
                const int ch = ks * 2 + (mi >> 1);
                ldsm_x4(smem_u32(a + row * BR + ((ch ^ ((row >> 1) & 3)) << 4)), af[i][0], af[i][1], af[i][2], af[i][3]);
            }
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int row = ks * 32 + (lane >> 3) * 8 + (lane & 7);
                const int ch = wn * 4 + q;
                u32 r0, r1, r2, r3;
                ldsm_x4_t(smem_u32(b + row * BK + ((ch ^ (row & 7)) << 4)), r0, r1, r2, r3);
                bf[q][0] = __byte_perm(r0, r1, 0x6420);  // even column, k 0-15
                bf[q][1] = __byte_perm(r2, r3, 0x6420);  // even column, k 16-31
                bf[q][2] = __byte_perm(r0, r1, 0x7531);  // odd column
                bf[q][3] = __byte_perm(r2, r3, 0x7531);
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    mma_s8(acc[i][2 * q], af[i], bf[q][0], bf[q][1]);
                    mma_s8(acc[i][2 * q + 1], af[i], bf[q][2], bf[q][3]);
                }
        }
    }
    cp_wait<0>();
    __syncthreads();

    // v[i][q][h][d]: row m0 + wm*64 + i*16 + g + 8h, column k0 + wn*64 + q*16 + tig*4 + d
    const int g = lane >> 2, tig = lane & 3;
    float v[4][4][2][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int gm = m0 + wm * 64 + i * 16 + g + 8 * h;
            const float st = (gm < M ? steps[gm] : 0.f) * (ROTATE ? 1.f / 16.f : 1.f);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                v[i][q][h][0] = (float)acc[i][2 * q][2 * h] * st;
                v[i][q][h][1] = (float)acc[i][2 * q + 1][2 * h] * st;
                v[i][q][h][2] = (float)acc[i][2 * q][2 * h + 1] * st;
                v[i][q][h][3] = (float)acc[i][2 * q + 1][2 * h + 1] * st;
            }
        }
    if (ROTATE) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
#pragma unroll
                for (int q = 0; q < 4; q++) h4(v[i][q][h][0], v[i][q][h][1], v[i][q][h][2], v[i][q][h][3]);  // digit 0
#pragma unroll
                for (int d = 0; d < 4; d++) h4(v[i][0][h][d], v[i][1][h][d], v[i][2][h][d], v[i][3][h][d]);  // digit 2
#pragma unroll
                for (int q = 0; q < 4; q++)
#pragma unroll
                    for (int d = 0; d < 4; d++) {  // digit 1: across the quad's lanes
                        const float x = v[i][q][h][d];
                        float s = x + __shfl_xor_sync(0xffffffffu, x, 1);
                        s += __shfl_xor_sync(0xffffffffu, s, 2);
                        v[i][q][h][d] = s - 2.f * __shfl_xor_sync(0xffffffffu, x, 3);
                    }
            }
        // digit 3: across the four warps of a row band, through shared memory, 16 rows at a time
        float* x = reinterpret_cast<float*>(smem);  // [2][16][256]
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int q = 0; q < 4; q++)
                    *reinterpret_cast<float4*>(x + (wm * 16 + g + 8 * h) * 256 + wn * 64 + q * 16 + tig * 4) =
                        make_float4(v[i][q][h][0], v[i][q][h][1], v[i][q][h][2], v[i][q][h][3]);
            __syncthreads();
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const float* p = x + (wm * 16 + g + 8 * h) * 256 + q * 16 + tig * 4;
                    const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 64);
                    const float4 c = *reinterpret_cast<const float4*>(p + 128), d = *reinterpret_cast<const float4*>(p + 192);
                    const float4 o = wn == 0 ? d : wn == 1 ? c : wn == 2 ? b : a;  // x_(3 - wn)
                    v[i][q][h][0] = a.x + b.x + c.x + d.x - 2.f * o.x;
                    v[i][q][h][1] = a.y + b.y + c.y + d.y - 2.f * o.y;
                    v[i][q][h][2] = a.z + b.z + c.z + d.z - 2.f * o.z;
                    v[i][q][h][3] = a.w + b.w + c.w + d.w - 2.f * o.w;
                }
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int gm = m0 + wm * 64 + i * 16 + g + 8 * h;
            if (gm >= M) continue;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int gk = k0 + wn * 64 + q * 16 + tig * 4;
                if (has_res) {
                    const uint2 r = *reinterpret_cast<const uint2*>(R + (long long)gm * K + gk);
                    v[i][q][h][0] += bf16_to_f((u16)(r.x & 0xffffu));
                    v[i][q][h][1] += bf16_to_f((u16)(r.x >> 16));
                    v[i][q][h][2] += bf16_to_f((u16)(r.y & 0xffffu));
                    v[i][q][h][3] += bf16_to_f((u16)(r.y >> 16));
                }
                uint2 o;
                o.x = (u32)f_to_bf16(v[i][q][h][0]) | ((u32)f_to_bf16(v[i][q][h][1]) << 16);
                o.y = (u32)f_to_bf16(v[i][q][h][2]) | ((u32)f_to_bf16(v[i][q][h][3]) << 16);
                *reinterpret_cast<uint2*>(out + (long long)gm * K + gk) = o;
            }
        }
}

extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm_rot(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K) {
    gemm_body<true>(A, B, steps, out, R, has_res, M, N, K);
}
extern "C" __global__ void __launch_bounds__(256, 1) zt_w8a8_gemm(const signed char* A, const signed char* B,
        const float* steps, u16* out, const u16* R, int has_res, int M, int N, int K) {
    gemm_body<false>(A, B, steps, out, R, has_res, M, N, K);
}
