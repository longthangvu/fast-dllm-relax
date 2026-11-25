"""Hand-written RDNA (gfx11, wave32) WMMA GEMM kernels in TVM TIR for the ROCm build.

Y[M, N] = X[M, K] @ W^T with fp32 accumulation (`llvm.amdgcn.wmma.f32.16x16x16.{f16,bf16}`):
- `wmma_gemm_f16` / `wmma_gemm_bf16`: fp16 / bf16 W [N, K]
- `wmma_gemm_w8` (+ `wmma_gemm_w8_pf`, register prefetch for long-K decode GEMMs): int8 W, uint32-packed
  [N, K/4], zero point 128, scales [N, K/G]
- `wmma_gemm_w4`: int4 W, uint32-packed [N, K/8], zero point 8, scales [N, K/G]

A block is 128 threads = 4 waves (2x2), each owning TI x TJ 16x16 WMMA tiles (block tile 32*TI x 32*TJ);
K advances by BK through LDS. Quantized B tiles are decoded into LDS once per block, so the decode
cost is shared by all rows. M is symbolic (rows beyond M are masked). Fragment layout (gfx11):
A lane l = row l % 16, B lane l = column l % 16, D lane l = rows 2i + l // 16 (i < 8), column l % 16.

Tile tables below were picked by cold-cache sweeps with numeric checks on gfx1151 (Radeon 8060S)
for the Fast-dLLM v2 1.5B projection shapes; other shapes fall back to generic tiles.
"""

from tvm.script import tirx as T

WMMA = "llvm.amdgcn.wmma.f32.16x16x16.f16"
WMMA_BF16 = "llvm.amdgcn.wmma.f32.16x16x16.bf16"  # operands are bf16 bit patterns as int16x16

# Tiles per wave (TI, TJ) for the Fast-dLLM v2 1.5B projections, good at both M = 128 and 512 (swept on gfx1151).
WMMA_W4_TILES: dict[tuple[int, int], tuple[int, int]] = {
    (2048, 1536): (2, 2),     # qkv
    (1536, 1536): (2, 4),     # o_proj
    (17920, 1536): (4, 2),    # gate_up
    (1536, 8960): (2, 2),     # down_proj
}


# Static decode M (8 / 32) int4 on gfx1151 (cold cache): WMMA with the int4 tile decoded once per block
# beats the tuned dlight GEMMs for gate_up / down (and qkv at M=32); None keeps the dlight path.
WMMA_W4_DECODE_TILES: dict[tuple[int, int, int], tuple[int, int] | None] = {
    (8, 2048, 1536): None, (32, 2048, 1536): (1, 4),        # qkv
    (8, 1536, 1536): None, (32, 1536, 1536): None,          # o_proj
    (8, 17920, 1536): (1, 1), (32, 17920, 1536): (1, 4),    # gate_up
    (8, 1536, 8960): (1, 1), (32, 1536, 8960): (1, 1),      # down_proj
}

# Static decode M (8 / 32) int8 (wmma_gemm_w8) on gfx1151, cold cache: (TI, TJ, BK).
WMMA_W8_DECODE_TILES: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (8, 2048, 1536): (1, 1, 64), (32, 2048, 1536): (1, 2, 128),       # qkv
    (8, 1536, 1536): (1, 2, 64), (32, 1536, 1536): (1, 1, 64),        # o_proj
    (8, 17920, 1536): (1, 1, 128), (32, 17920, 1536): (1, 1, 128),    # gate_up
    (8, 1536, 8960): (1, 1, 128), (32, 1536, 8960): (1, 2, 128),      # down_proj
}

# Static decode M (8 / 32) fp16 / bf16 (wmma_gemm_f16 / _bf16) on gfx1151, cold cache: (TI, TJ, BK).
WMMA_F16_DECODE_TILES: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (8, 2048, 1536): (1, 1, 128), (32, 2048, 1536): (1, 2, 128),      # qkv
    (8, 1536, 1536): (1, 1, 128), (32, 1536, 1536): (1, 1, 128),      # o_proj
    (8, 17920, 1536): (1, 1, 128), (32, 17920, 1536): (1, 1, 128),    # gate_up
    (8, 1536, 8960): (1, 1, 128), (32, 1536, 8960): (1, 1, 128),      # down_proj
}

# fp16 weights (swept on gfx1151, M = 128 / 512): qkv 2x4, gate_up 4x4, down 4x4; o_proj like qkv.
WMMA_F16_TILES: dict[tuple[int, int], tuple[int, int]] = {
    (2048, 1536): (2, 4),
    (1536, 1536): (2, 4),
    (17920, 1536): (4, 4),
    (1536, 8960): (4, 4),
}


def _check_chunks(bm: int, bn: int, kc: int) -> None:
    # The 128 threads copy 16-value chunks cooperatively; a tile with fewer chunks than threads
    # (or a remainder) would silently skip part of its loads.
    if (bm * kc) % 128 or (bn * kc) % 128 or bm * kc < 128 or bn * kc < 128:
        raise ValueError(f"WMMA tile {bm}x{bn} with BK={16 * kc} does not split into whole 128-thread loads")


def wmma_tiles(n: int, k: int, table: dict | None = None) -> tuple[int, int]:
    for ti, tj in ((table if table is not None else WMMA_W4_TILES).get((n, k), (2, 2)), (2, 2), (1, 1)):
        if n % (32 * tj) == 0 and k % 64 == 0:
            return ti, tj
    raise ValueError(f"no WMMA tiling for N={n}, K={k}")


def wmma_gemm_w4(N: int, K: int, TI: int = 4, TJ: int = 4, BK: int = 64, PAD: int = 8, G: int = 128):
    """W4A16 variant: W given as packed uint32 [N, K/8] (nibble i of word j = q[n, 8j+i], zero point 8)
    plus fp16 scales [N, K/G]; each B chunk of 16 k-values is decoded to fp16 while it is written to
    LDS, so the decode is amortized over all BM rows of the block."""
    BM, BN = 32 * TI, 32 * TJ
    assert N % BN == 0 and K % BK == 0 and BK % 16 == 0 and G % 16 == 0
    KS, LD, KC = BK // 16, BK + PAD, BK // 16
    A_CH, B_CH = BM * KC // 128, BN * KC // 128
    _check_chunks(BM, BN, KC)

    @T.prim_func(private=True)
    def gemm(x_h: T.handle, p_h: T.handle, s_h: T.handle, y_h: T.handle):
        T.func_attr({"tirx.noalias": True, "tirx.is_scheduled": True})
        m = T.int64()
        X = T.match_buffer(x_h, (m, K), "float16")
        P = T.match_buffer(p_h, (N, K // 8), "uint32")
        S = T.match_buffer(s_h, (N, K // G), "float16")
        Y = T.match_buffer(y_h, (m, N), "float16")
        for by in T.thread_binding((m + BM - 1) // BM, thread="blockIdx.y"):
            for bx in T.thread_binding(N // BN, thread="blockIdx.x"):
                for tx in T.thread_binding(128, thread="threadIdx.x"):
                    As = T.alloc_buffer((BM * LD,), "float16", scope="shared")
                    Bs = T.alloc_buffer((BN * LD,), "float16", scope="shared")
                    acc = T.alloc_buffer((TI * TJ * 8,), "float32", scope="local")
                    a = T.alloc_buffer((TI * 16,), "float16", scope="local")
                    b = T.alloc_buffer((TJ * 16,), "float16", scope="local")
                    wm = tx // 64
                    wn = (tx // 32) % 2
                    lane = tx % 32
                    for i in T.serial(TI * TJ * 8):
                        acc[i] = T.float32(0)
                    for k0 in T.serial(K // BK):
                        for ch in T.serial(A_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            gr = by * BM + r
                            if gr < m:
                                As[T.ramp(r * LD + c, 1, 16)] = X[gr, T.ramp(k0 * BK + c, 1, 16)]
                            else:
                                As[T.ramp(r * LD + c, 1, 16)] = T.Broadcast(T.float16(0), 16)
                        for ch in T.serial(B_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            n = bx * BN + r
                            kg = k0 * BK + c
                            sc = S[n, kg // G]
                            for h in T.serial(2):
                                wd = P[n, kg // 8 + h]
                                for j in T.serial(8):
                                    Bs[r * LD + c + h * 8 + j] = (
                                        T.Cast("float16", (wd >> T.Cast("uint32", j * 4)) & T.uint32(15)) - T.float16(8)
                                    ) * sc
                        T.tvm_storage_sync("shared")
                        for kk in T.serial(KS):
                            for t in T.serial(TI):
                                a[T.ramp(t * 16, 1, 16)] = As[T.ramp((wm * 16 * TI + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for t in T.serial(TJ):
                                b[T.ramp(t * 16, 1, 16)] = Bs[T.ramp((wn * 16 * TJ + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for ti in T.serial(TI):
                                for tj in T.serial(TJ):
                                    acc[T.ramp((ti * TJ + tj) * 8, 1, 8)] = T.call_llvm_pure_intrin(
                                        "float32x8", WMMA, a[T.ramp(ti * 16, 1, 16)], b[T.ramp(tj * 16, 1, 16)],
                                        acc[T.ramp((ti * TJ + tj) * 8, 1, 8)])
                        T.tvm_storage_sync("shared")
                    for ti in T.serial(TI):
                        for tj in T.serial(TJ):
                            for i in T.serial(8):
                                row = by * BM + wm * 16 * TI + ti * 16 + 2 * i + lane // 16
                                col = bx * BN + wn * 16 * TJ + tj * 16 + lane % 16
                                if row < m:
                                    Y[row, col] = T.Cast("float16", acc[(ti * TJ + tj) * 8 + i])

    return gemm


def wmma_gemm_w8(N: int, K: int, TI: int = 4, TJ: int = 4, BK: int = 64, PAD: int = 8, G: int = 128):
    """W8A16 variant: W given as packed uint32 [N, K/4] (byte i of word j = q[n, 4j+i], zero point 128)
    plus fp16 scales [N, K/G]; each B chunk of 16 k-values is decoded to fp16 while it is written to
    LDS, so the decode is amortized over all BM rows of the block."""
    BM, BN = 32 * TI, 32 * TJ
    assert N % BN == 0 and K % BK == 0 and BK % 16 == 0 and G % 16 == 0
    KS, LD, KC = BK // 16, BK + PAD, BK // 16
    A_CH, B_CH = BM * KC // 128, BN * KC // 128
    _check_chunks(BM, BN, KC)

    @T.prim_func(private=True)
    def gemm(x_h: T.handle, p_h: T.handle, s_h: T.handle, y_h: T.handle):
        T.func_attr({"tirx.noalias": True, "tirx.is_scheduled": True})
        m = T.int64()
        X = T.match_buffer(x_h, (m, K), "float16")
        P = T.match_buffer(p_h, (N, K // 4), "uint32")
        S = T.match_buffer(s_h, (N, K // G), "float16")
        Y = T.match_buffer(y_h, (m, N), "float16")
        for by in T.thread_binding((m + BM - 1) // BM, thread="blockIdx.y"):
            for bx in T.thread_binding(N // BN, thread="blockIdx.x"):
                for tx in T.thread_binding(128, thread="threadIdx.x"):
                    As = T.alloc_buffer((BM * LD,), "float16", scope="shared")
                    Bs = T.alloc_buffer((BN * LD,), "float16", scope="shared")
                    acc = T.alloc_buffer((TI * TJ * 8,), "float32", scope="local")
                    a = T.alloc_buffer((TI * 16,), "float16", scope="local")
                    b = T.alloc_buffer((TJ * 16,), "float16", scope="local")
                    wm = tx // 64
                    wn = (tx // 32) % 2
                    lane = tx % 32
                    for i in T.serial(TI * TJ * 8):
                        acc[i] = T.float32(0)
                    for k0 in T.serial(K // BK):
                        for ch in T.serial(A_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            gr = by * BM + r
                            if gr < m:
                                As[T.ramp(r * LD + c, 1, 16)] = X[gr, T.ramp(k0 * BK + c, 1, 16)]
                            else:
                                As[T.ramp(r * LD + c, 1, 16)] = T.Broadcast(T.float16(0), 16)
                        for ch in T.serial(B_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            n = bx * BN + r
                            kg = k0 * BK + c
                            sc = S[n, kg // G]
                            for h in T.serial(4):
                                wd = P[n, kg // 4 + h]
                                for j in T.serial(4):
                                    Bs[r * LD + c + h * 4 + j] = (
                                        T.Cast("float16", (wd >> T.Cast("uint32", j * 8)) & T.uint32(255)) - T.float16(128)
                                    ) * sc
                        T.tvm_storage_sync("shared")
                        for kk in T.serial(KS):
                            for t in T.serial(TI):
                                a[T.ramp(t * 16, 1, 16)] = As[T.ramp((wm * 16 * TI + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for t in T.serial(TJ):
                                b[T.ramp(t * 16, 1, 16)] = Bs[T.ramp((wn * 16 * TJ + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for ti in T.serial(TI):
                                for tj in T.serial(TJ):
                                    acc[T.ramp((ti * TJ + tj) * 8, 1, 8)] = T.call_llvm_pure_intrin(
                                        "float32x8", WMMA, a[T.ramp(ti * 16, 1, 16)], b[T.ramp(tj * 16, 1, 16)],
                                        acc[T.ramp((ti * TJ + tj) * 8, 1, 8)])
                        T.tvm_storage_sync("shared")
                    for ti in T.serial(TI):
                        for tj in T.serial(TJ):
                            for i in T.serial(8):
                                row = by * BM + wm * 16 * TI + ti * 16 + 2 * i + lane // 16
                                col = bx * BN + wn * 16 * TJ + tj * 16 + lane % 16
                                if row < m:
                                    Y[row, col] = T.Cast("float16", acc[(ti * TJ + tj) * 8 + i])

    return gemm


def wmma_gemm_w8_pf(N: int, K: int, TI: int = 1, TJ: int = 1, BK: int = 128, PAD: int = 8, G: int = 128):
    """wmma_gemm_w8 with register prefetch: the packed weights (and scales) of k-step k0 + 1 are
    loaded into registers before the WMMAs of step k0, so global loads overlap compute (decode M).
    W8A16 variant: W given as packed uint32 [N, K/4] (byte i of word j = q[n, 4j+i], zero point 128)
    plus fp16 scales [N, K/G]; each B chunk of 16 k-values is decoded to fp16 while it is written to
    LDS, so the decode is amortized over all BM rows of the block."""
    BM, BN = 32 * TI, 32 * TJ
    assert N % BN == 0 and K % BK == 0 and BK % 16 == 0 and G % 16 == 0
    KS, LD, KC = BK // 16, BK + PAD, BK // 16
    A_CH, B_CH = BM * KC // 128, BN * KC // 128
    _check_chunks(BM, BN, KC)

    @T.prim_func(private=True)
    def gemm(x_h: T.handle, p_h: T.handle, s_h: T.handle, y_h: T.handle):
        T.func_attr({"tirx.noalias": True, "tirx.is_scheduled": True})
        m = T.int64()
        X = T.match_buffer(x_h, (m, K), "float16")
        P = T.match_buffer(p_h, (N, K // 4), "uint32")
        S = T.match_buffer(s_h, (N, K // G), "float16")
        Y = T.match_buffer(y_h, (m, N), "float16")
        for by in T.thread_binding((m + BM - 1) // BM, thread="blockIdx.y"):
            for bx in T.thread_binding(N // BN, thread="blockIdx.x"):
                for tx in T.thread_binding(128, thread="threadIdx.x"):
                    As = T.alloc_buffer((BM * LD,), "float16", scope="shared")
                    Bs = T.alloc_buffer((BN * LD,), "float16", scope="shared")
                    acc = T.alloc_buffer((TI * TJ * 8,), "float32", scope="local")
                    a = T.alloc_buffer((TI * 16,), "float16", scope="local")
                    b = T.alloc_buffer((TJ * 16,), "float16", scope="local")
                    wm = tx // 64
                    wn = (tx // 32) % 2
                    lane = tx % 32
                    pw = T.alloc_buffer((B_CH * 4,), "uint32", scope="local")
                    ps = T.alloc_buffer((B_CH,), "float16", scope="local")
                    for i in T.serial(TI * TJ * 8):
                        acc[i] = T.float32(0)
                    for ch in T.serial(B_CH):
                        idx = ch * 128 + tx
                        n = bx * BN + idx // KC
                        kg = (idx % KC) * 16
                        ps[ch] = S[n, kg // G]
                        for h in T.serial(4):
                            pw[ch * 4 + h] = P[n, kg // 4 + h]
                    for k0 in T.serial(K // BK):
                        for ch in T.serial(A_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            gr = by * BM + r
                            if gr < m:
                                As[T.ramp(r * LD + c, 1, 16)] = X[gr, T.ramp(k0 * BK + c, 1, 16)]
                            else:
                                As[T.ramp(r * LD + c, 1, 16)] = T.Broadcast(T.float16(0), 16)
                        for ch in T.serial(B_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            for h in T.serial(4):
                                for j in T.serial(4):
                                    Bs[r * LD + c + h * 4 + j] = (
                                        T.Cast("float16", (pw[ch * 4 + h] >> T.Cast("uint32", j * 8)) & T.uint32(255))
                                        - T.float16(128)
                                    ) * ps[ch]
                        T.tvm_storage_sync("shared")
                        if k0 + 1 < K // BK:
                            for ch in T.serial(B_CH):
                                idx = ch * 128 + tx
                                n = bx * BN + idx // KC
                                kg = (k0 + 1) * BK + (idx % KC) * 16
                                ps[ch] = S[n, kg // G]
                                for h in T.serial(4):
                                    pw[ch * 4 + h] = P[n, kg // 4 + h]
                        for kk in T.serial(KS):
                            for t in T.serial(TI):
                                a[T.ramp(t * 16, 1, 16)] = As[T.ramp((wm * 16 * TI + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for t in T.serial(TJ):
                                b[T.ramp(t * 16, 1, 16)] = Bs[T.ramp((wn * 16 * TJ + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for ti in T.serial(TI):
                                for tj in T.serial(TJ):
                                    acc[T.ramp((ti * TJ + tj) * 8, 1, 8)] = T.call_llvm_pure_intrin(
                                        "float32x8", WMMA, a[T.ramp(ti * 16, 1, 16)], b[T.ramp(tj * 16, 1, 16)],
                                        acc[T.ramp((ti * TJ + tj) * 8, 1, 8)])
                        T.tvm_storage_sync("shared")
                    for ti in T.serial(TI):
                        for tj in T.serial(TJ):
                            for i in T.serial(8):
                                row = by * BM + wm * 16 * TI + ti * 16 + 2 * i + lane // 16
                                col = bx * BN + wn * 16 * TJ + tj * 16 + lane % 16
                                if row < m:
                                    Y[row, col] = T.Cast("float16", acc[(ti * TJ + tj) * 8 + i])

    return gemm



def wmma_gemm_f16(N: int, K: int, TI: int = 2, TJ: int = 2, BK: int = 64, PAD: int = 8):
    """v3: 128 threads = 4 waves in a 2x2 grid; each wave owns TI x TJ WMMA 16x16 tiles, so the block
    tile is BM = 32*TI by BN = 32*TJ. Vectorized 32-byte global loads (row guard on X), vectorized
    LDS fragment reads, BK step, LDS rows padded by PAD halves."""
    BM, BN = 32 * TI, 32 * TJ
    assert N % BN == 0 and K % BK == 0 and BK % 16 == 0
    KS, LD, KC = BK // 16, BK + PAD, BK // 16
    A_CH, B_CH = BM * KC // 128, BN * KC // 128
    _check_chunks(BM, BN, KC)
    assert BM * KC % 128 == 0 and BN * KC % 128 == 0

    @T.prim_func(private=True)
    def gemm(x_h: T.handle, w_h: T.handle, y_h: T.handle):
        T.func_attr({"tirx.noalias": True, "tirx.is_scheduled": True})
        m = T.int64()
        X = T.match_buffer(x_h, (m, K), "float16")
        W = T.match_buffer(w_h, (N, K), "float16")
        Y = T.match_buffer(y_h, (m, N), "float16")
        for by in T.thread_binding((m + BM - 1) // BM, thread="blockIdx.y"):
            for bx in T.thread_binding(N // BN, thread="blockIdx.x"):
                for tx in T.thread_binding(128, thread="threadIdx.x"):
                    As = T.alloc_buffer((BM * LD,), "float16", scope="shared")
                    Bs = T.alloc_buffer((BN * LD,), "float16", scope="shared")
                    acc = T.alloc_buffer((TI * TJ * 8,), "float32", scope="local")
                    a = T.alloc_buffer((TI * 16,), "float16", scope="local")
                    b = T.alloc_buffer((TJ * 16,), "float16", scope="local")
                    wm = tx // 64
                    wn = (tx // 32) % 2
                    lane = tx % 32
                    for i in T.serial(TI * TJ * 8):
                        acc[i] = T.float32(0)
                    for k0 in T.serial(K // BK):
                        for ch in T.serial(A_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            gr = by * BM + r
                            if gr < m:
                                As[T.ramp(r * LD + c, 1, 16)] = X[gr, T.ramp(k0 * BK + c, 1, 16)]
                            else:
                                As[T.ramp(r * LD + c, 1, 16)] = T.Broadcast(T.float16(0), 16)
                        for ch in T.serial(B_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            Bs[T.ramp(r * LD + c, 1, 16)] = W[bx * BN + r, T.ramp(k0 * BK + c, 1, 16)]
                        T.tvm_storage_sync("shared")
                        for kk in T.serial(KS):
                            for t in T.serial(TI):
                                a[T.ramp(t * 16, 1, 16)] = As[T.ramp((wm * 16 * TI + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for t in T.serial(TJ):
                                b[T.ramp(t * 16, 1, 16)] = Bs[T.ramp((wn * 16 * TJ + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for ti in T.serial(TI):
                                for tj in T.serial(TJ):
                                    acc[T.ramp((ti * TJ + tj) * 8, 1, 8)] = T.call_llvm_pure_intrin(
                                        "float32x8", WMMA, a[T.ramp(ti * 16, 1, 16)], b[T.ramp(tj * 16, 1, 16)],
                                        acc[T.ramp((ti * TJ + tj) * 8, 1, 8)])
                        T.tvm_storage_sync("shared")
                    for ti in T.serial(TI):
                        for tj in T.serial(TJ):
                            for i in T.serial(8):
                                row = by * BM + wm * 16 * TI + ti * 16 + 2 * i + lane // 16
                                col = bx * BN + wn * 16 * TJ + tj * 16 + lane % 16
                                if row < m:
                                    Y[row, col] = T.Cast("float16", acc[(ti * TJ + tj) * 8 + i])

    return gemm


def wmma_gemm_bf16(N: int, K: int, TI: int = 2, TJ: int = 2, BK: int = 64, PAD: int = 8):
    """bf16 variant of wmma_gemm_f16 (gfx11 v_wmma_f32_16x16x16_bf16; LLVM takes bf16 as i16 bits).
    v3: 128 threads = 4 waves in a 2x2 grid; each wave owns TI x TJ WMMA 16x16 tiles, so the block
    tile is BM = 32*TI by BN = 32*TJ. Vectorized 32-byte global loads (row guard on X), vectorized
    LDS fragment reads, BK step, LDS rows padded by PAD halves."""
    BM, BN = 32 * TI, 32 * TJ
    assert N % BN == 0 and K % BK == 0 and BK % 16 == 0
    KS, LD, KC = BK // 16, BK + PAD, BK // 16
    A_CH, B_CH = BM * KC // 128, BN * KC // 128
    _check_chunks(BM, BN, KC)
    assert BM * KC % 128 == 0 and BN * KC % 128 == 0

    @T.prim_func(private=True)
    def gemm(x_h: T.handle, w_h: T.handle, y_h: T.handle):
        T.func_attr({"tirx.noalias": True, "tirx.is_scheduled": True})
        m = T.int64()
        X = T.match_buffer(x_h, (m, K), "bfloat16")
        W = T.match_buffer(w_h, (N, K), "bfloat16")
        Y = T.match_buffer(y_h, (m, N), "bfloat16")
        for by in T.thread_binding((m + BM - 1) // BM, thread="blockIdx.y"):
            for bx in T.thread_binding(N // BN, thread="blockIdx.x"):
                for tx in T.thread_binding(128, thread="threadIdx.x"):
                    As = T.alloc_buffer((BM * LD,), "int16", scope="shared")
                    Bs = T.alloc_buffer((BN * LD,), "int16", scope="shared")
                    acc = T.alloc_buffer((TI * TJ * 8,), "float32", scope="local")
                    a = T.alloc_buffer((TI * 16,), "int16", scope="local")
                    b = T.alloc_buffer((TJ * 16,), "int16", scope="local")
                    wm = tx // 64
                    wn = (tx // 32) % 2
                    lane = tx % 32
                    for i in T.serial(TI * TJ * 8):
                        acc[i] = T.float32(0)
                    for k0 in T.serial(K // BK):
                        for ch in T.serial(A_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            gr = by * BM + r
                            if gr < m:
                                As[T.ramp(r * LD + c, 1, 16)] = T.reinterpret("int16x16", X[gr, T.ramp(k0 * BK + c, 1, 16)])
                            else:
                                As[T.ramp(r * LD + c, 1, 16)] = T.Broadcast(T.int16(0), 16)
                        for ch in T.serial(B_CH):
                            idx = ch * 128 + tx
                            r = idx // KC
                            c = (idx % KC) * 16
                            Bs[T.ramp(r * LD + c, 1, 16)] = T.reinterpret("int16x16", W[bx * BN + r, T.ramp(k0 * BK + c, 1, 16)])
                        T.tvm_storage_sync("shared")
                        for kk in T.serial(KS):
                            for t in T.serial(TI):
                                a[T.ramp(t * 16, 1, 16)] = As[T.ramp((wm * 16 * TI + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for t in T.serial(TJ):
                                b[T.ramp(t * 16, 1, 16)] = Bs[T.ramp((wn * 16 * TJ + t * 16 + lane % 16) * LD + kk * 16, 1, 16)]
                            for ti in T.serial(TI):
                                for tj in T.serial(TJ):
                                    acc[T.ramp((ti * TJ + tj) * 8, 1, 8)] = T.call_llvm_pure_intrin(
                                        "float32x8", WMMA_BF16, a[T.ramp(ti * 16, 1, 16)], b[T.ramp(tj * 16, 1, 16)],
                                        acc[T.ramp((ti * TJ + tj) * 8, 1, 8)])
                        T.tvm_storage_sync("shared")
                    for ti in T.serial(TI):
                        for tj in T.serial(TJ):
                            for i in T.serial(8):
                                row = by * BM + wm * 16 * TI + ti * 16 + 2 * i + lane // 16
                                col = bx * BN + wn * 16 * TJ + tj * 16 + lane % 16
                                if row < m:
                                    Y[row, col] = T.Cast("bfloat16", acc[(ti * TJ + tj) * 8 + i])

    return gemm
