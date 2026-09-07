"""Gather directly from separate exact/history tensors into FA2 varlen storage."""
import triton
import triton.language as tl


@triton.jit
def gather_queries(Q, Indices, Out, N, T, H: tl.constexpr, D: tl.constexpr,
                   SB, ST, SH, SD,
                   R: tl.constexpr, C: tl.constexpr):
    row = tl.program_id(0)*R + tl.arange(0, R)
    col = tl.arange(0, C)
    index = tl.load(Indices+row, row < N, 0)
    head, token, batch = index % H, (index//H) % T, index//(H*T)
    address = batch[:, None]*SB + token[:, None]*ST + head[:, None]*SH + col[None, :]*SD
    value = tl.load(Q+address, (row[:, None] < N) & (col[None, :] < D), 0)
    tl.store(Out+row[:, None]*D+col[None, :], value, (row[:, None] < N) & (col[None, :] < D))


@triton.jit
def gather_separate_kv(EK, EV, HK, HV, Indices, OutK, OutV,
                       N, E, U, H: tl.constexpr, D: tl.constexpr,
                       EKB, EKT, EKH, EKD,
                       EVB, EVT, EVH, EVD,
                       HKB, HKT, HKH, HKD,
                       HVB, HVT, HVH, HVD,
                       R: tl.constexpr, C: tl.constexpr):
    row = tl.program_id(0)*R + tl.arange(0, R)
    col = tl.arange(0, C)
    index = tl.load(Indices+row, row < N, 0)
    head, token, batch = index % H, (index//H) % (E+U), index//(H*(E+U))
    valid = (row[:, None] < N) & (col[None, :] < D)
    exact = token[:, None] < E
    ek_address = batch[:, None]*EKB + token[:, None]*EKT + head[:, None]*EKH + col[None, :]*EKD
    ev_address = batch[:, None]*EVB + token[:, None]*EVT + head[:, None]*EVH + col[None, :]*EVD
    hk_address = batch[:, None]*HKB + (token[:, None]-E)*HKT + head[:, None]*HKH + col[None, :]*HKD
    hv_address = batch[:, None]*HVB + (token[:, None]-E)*HVT + head[:, None]*HVH + col[None, :]*HVD
    key = tl.where(exact, tl.load(EK+ek_address, valid & exact, 0), tl.load(HK+hk_address, valid & ~exact, 0))
    value = tl.where(exact, tl.load(EV+ev_address, valid & exact, 0), tl.load(HV+hv_address, valid & ~exact, 0))
    tl.store(OutK+row[:, None]*D+col[None, :], key, valid)
    tl.store(OutV+row[:, None]*D+col[None, :], value, valid)
