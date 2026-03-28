"""custom adjoint definitions for MuJoCo Warp autodifferentiation.

This module centralizes all ``@wp.func_grad`` registrations and the
implicit differentiation adjoint for the constraint solver.

Import this module via ``grad.py`` dont import it directly
"""

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_func
from mujoco_warp._src.warp_util import cache_kernel

# ---------------------------------------------------------------------------
# Phase 3: efc-level gradient kernels for collision chain
# ---------------------------------------------------------------------------


@wp.kernel
def _efc_J_grad_kernel(
  # In:
  v: wp.array2d(dtype=float),
  efc_force: wp.array2d(dtype=float),
  nefc: wp.array(dtype=int),
  nv: int,
  njmax: int,
  # Out:
  efc_J_grad_out: wp.array3d(dtype=float),
):
  """Compute adj_efc_J[i, j] = v[j] * efc_force[i].

  From KKT: F(qacc) = M*qacc - qfrc_smooth - J^T*f = 0
  The derivative of J^T*f w.r.t. J[i,j] is f[i] * delta, and the
  adjoint vector v gives the sensitivity: adj_J[i,j] = v[j] * f[i].
  """
  worldid, efcid, dofid = wp.tid()
  if efcid < nefc[worldid] and dofid < nv:
    efc_J_grad_out[worldid, efcid, dofid] = v[worldid, dofid] * efc_force[worldid, efcid]


@wp.kernel
def _efc_pos_grad_kernel(
  # In:
  efc_aref_grad: wp.array2d(dtype=float),
  contact_solref: wp.array(dtype=wp.vec2),
  contact_solimp: wp.array(dtype=types.vec5),
  contact_includemargin: wp.array(dtype=float),
  contact_dist: wp.array(dtype=float),
  contact_efc_address: wp.array2d(dtype=int),
  contact_worldid: wp.array(dtype=int),
  contact_type: wp.array(dtype=int),
  nacon: wp.array(dtype=int),
  opt_timestep: wp.array(dtype=float),
  opt_disableflags: int,
  # Out:
  efc_pos_grad_out: wp.array2d(dtype=float),
):
  """Compute adj_efc_pos from adj_efc_aref.

  From efc_aref = -k * imp * pos - b * vel, d(aref)/d(pos) = -k*imp.
  So adj_efc_pos = adj_efc_aref * (-k * imp).
  We iterate over contacts and their first dimension (normal direction).
  """
  conid = wp.tid()
  if conid >= nacon[0]:
    return
  if not (contact_type[conid] & 1):  # ContactType.CONSTRAINT
    return

  efcid = contact_efc_address[conid, 0]
  if efcid < 0:
    return

  worldid = contact_worldid[conid]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  solref = contact_solref[conid]
  solimp = contact_solimp[conid]
  includemargin = contact_includemargin[conid]
  pos_val = contact_dist[conid] - includemargin

  # Recompute k and imp (same as _efc_row)
  timeconst = solref[0]
  dampratio = solref[1]
  dmin = solimp[0]
  dmax = solimp[1]
  width = solimp[2]
  mid = solimp[3]
  power = solimp[4]

  if not (opt_disableflags & types.DisableBit.REFSAFE):
    timeconst = wp.max(timeconst, 2.0 * timestep)

  dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(types.MJ_MINVAL, width)
  mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, power)

  dmax_sq = dmax * dmax
  k = 1.0 / (dmax_sq * timeconst * timeconst * dampratio * dampratio)
  k = wp.where(solref[0] <= 0.0, -solref[0] / dmax_sq, k)

  imp_x = wp.abs(pos_val) / width
  imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
  imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
  imp_y = wp.where(imp_x < mid, imp_a, imp_b)
  imp = dmin + imp_y * (dmax - dmin)
  imp = wp.clamp(imp, dmin, dmax)
  imp = wp.where(imp_x > 1.0, dmax, imp)

  # d(aref)/d(pos) = -k * imp
  daref_dpos = -k * imp

  adj_aref = efc_aref_grad[worldid, efcid]
  efc_pos_grad_out[worldid, efcid] = adj_aref * daref_dpos


@wp.func_grad(math.quat_integrate)
def _quat_integrate_grad(q: wp.quat, v: wp.vec3, dt: float, adj_ret: wp.quat):
  """Custom adjoint avoiding gradient singularity at |v|=0."""
  EPS = float(1e-10)
  norm_v = wp.length(v)
  norm_v_sq = norm_v * norm_v
  half_angle = dt * norm_v * 0.5

  # sinc-safe rotation quaternion construction
  if norm_v > EPS:
    s_over_nv = wp.sin(half_angle) / norm_v  # sin(dt|v|/2) / |v|
    c = wp.cos(half_angle)
    # d(s_over_nv)/dv_j = ds_coeff * v_j
    ds_coeff = (c * dt * 0.5 - s_over_nv) / norm_v_sq
  else:
    s_over_nv = dt * 0.5
    c = 1.0
    # Taylor limit: (c*dt/2 - s_over_nv) / |v|^2 -> -dt^3/24
    ds_coeff = -dt * dt * dt / 24.0

  q_rot = wp.quat(
    c,
    s_over_nv * v[0],
    s_over_nv * v[1],
    s_over_nv * v[2],
  )

  # recompute forward intermediates
  q_len = wp.length(q)
  q_inv_len = 1.0 / wp.max(q_len, EPS)
  q_n = wp.quat(
    q[0] * q_inv_len,
    q[1] * q_inv_len,
    q[2] * q_inv_len,
    q[3] * q_inv_len,
  )

  q_res = math.mul_quat(q_n, q_rot)
  res_len = wp.length(q_res)
  res_inv = 1.0 / wp.max(res_len, EPS)

  # result = normalize(q_res)
  # adj_q_res_k = adj_ret_k / |q_res| - q_res_k * dot(adj_ret, q_res) / |q_res|^3
  dot_ar = adj_ret[0] * q_res[0] + adj_ret[1] * q_res[1] + adj_ret[2] * q_res[2] + adj_ret[3] * q_res[3]
  res_inv3 = res_inv * res_inv * res_inv
  adj_qr = wp.quat(
    adj_ret[0] * res_inv - q_res[0] * dot_ar * res_inv3,
    adj_ret[1] * res_inv - q_res[1] * dot_ar * res_inv3,
    adj_ret[2] * res_inv - q_res[2] * dot_ar * res_inv3,
    adj_ret[3] * res_inv - q_res[3] * dot_ar * res_inv3,
  )

  # q_res = mul_quat(q_n, q_rot)
  # adj_q_n  = mul_quat(adj_qr, conj(q_rot))
  # adj_q_rot = mul_quat(conj(q_n), adj_qr)
  q_rot_conj = wp.quat(q_rot[0], -q_rot[1], -q_rot[2], -q_rot[3])
  adj_qn = math.mul_quat(adj_qr, q_rot_conj)

  q_n_conj = wp.quat(q_n[0], -q_n[1], -q_n[2], -q_n[3])
  adj_q_rot = math.mul_quat(q_n_conj, adj_qr)

  # q_rot = (c, s_over_nv * v)
  # d(c)/dv_j = -s_over_nv * dt/2 * v_j
  # d(s_over_nv * v_i)/dv_j = ds_coeff * v_j * v_i + s_over_nv * delta_ij
  sv_dot = adj_q_rot[1] * v[0] + adj_q_rot[2] * v[1] + adj_q_rot[3] * v[2]
  common = -s_over_nv * dt * 0.5 * adj_q_rot[0] + ds_coeff * sv_dot
  adj_v_val = wp.vec3(
    common * v[0] + s_over_nv * adj_q_rot[1],
    common * v[1] + s_over_nv * adj_q_rot[2],
    common * v[2] + s_over_nv * adj_q_rot[3],
  )

  # adj_dt from q_rot dependency on dt
  # d(c)/d(dt)            = -sin(half_angle) * norm_v / 2
  # d(s_over_nv * v_i)/dt = (c / 2) * v_i
  adj_dt_val = adj_q_rot[0] * (-wp.sin(half_angle) * norm_v * 0.5)
  adj_dt_val += sv_dot * c * 0.5

  # q_n = normalize(q)
  # adj_q_k = adj_qn_k / |q| - q_k * dot(adj_qn, q) / |q|^3
  dot_aqn = adj_qn[0] * q[0] + adj_qn[1] * q[1] + adj_qn[2] * q[2] + adj_qn[3] * q[3]
  q_inv_len3 = q_inv_len * q_inv_len * q_inv_len
  adj_q_val = wp.quat(
    adj_qn[0] * q_inv_len - q[0] * dot_aqn * q_inv_len3,
    adj_qn[1] * q_inv_len - q[1] * dot_aqn * q_inv_len3,
    adj_qn[2] * q_inv_len - q[2] * dot_aqn * q_inv_len3,
    adj_qn[3] * q_inv_len - q[3] * dot_aqn * q_inv_len3,
  )

  # accumulate adjoints
  wp.adjoint[q] += adj_q_val
  wp.adjoint[v] += adj_v_val
  wp.adjoint[dt] += adj_dt_val


# ---------------------------------------------------------------------------
# Solver implicit differentiation adjoint
# ---------------------------------------------------------------------------

_BLOCK_CHOLESKY_DIM = 32


@wp.kernel
def _copy_grad_kernel(
  # In:
  src: wp.array2d(dtype=float),
  # Out:
  dst: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  dst[worldid, dofid] = src[worldid, dofid]


@cache_kernel
def _adjoint_cholesky_tile(nv: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    H: wp.array3d(dtype=float),
    b: wp.array2d(dtype=float),
    # Out:
    out: wp.array2d(dtype=float),
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(nv)
    H_tile = wp.tile_load(H[worldid], shape=(TILE_SIZE, TILE_SIZE))
    b_tile = wp.tile_load(b[worldid], shape=(TILE_SIZE,))
    L = wp.tile_cholesky(H_tile)
    x = wp.tile_cholesky_solve(L, b_tile)
    wp.tile_store(out[worldid], x)

  return kernel


@cache_kernel
def _adjoint_cholesky_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    hfactor: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    nv_runtime: int,
    # Out:
    out: wp.array3d(dtype=float),
  ):
    worldid = wp.tid()
    wp.static(create_blocked_cholesky_solve_func(tile_size, matrix_size))(
      hfactor[worldid], b[worldid], nv_runtime, out[worldid]
    )

  return kernel


@cache_kernel
def _adjoint_cholesky_full_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    H: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    nv_runtime: int,
    hfactor_tmp: wp.array3d(dtype=float),
    # Out:
    out: wp.array3d(dtype=float),
  ):
    worldid = wp.tid()
    wp.static(create_blocked_cholesky_func(tile_size))(H[worldid], nv_runtime, hfactor_tmp[worldid])
    wp.static(create_blocked_cholesky_solve_func(tile_size, matrix_size))(
      hfactor_tmp[worldid], b[worldid], nv_runtime, out[worldid]
    )

  return kernel


@wp.kernel
def _padding_h_adjoint(
  nv: int,
  H_out: wp.array3d(dtype=float),
):
  worldid, elementid = wp.tid()
  dofid = nv + elementid
  H_out[worldid, dofid, dofid] = 1.0


def _solve_hessian_system(m: types.Model, d: types.Data, b, out):
  """Solve H * x = b using stored solver Hessian."""
  if m.nv <= _BLOCK_CHOLESKY_DIM:
    wp.launch_tiled(
      _adjoint_cholesky_tile(m.nv),
      dim=d.nworld,
      inputs=[d.solver_h, b],
      outputs=[out],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  else:
    b_3d = b.reshape((d.nworld, m.nv_pad, 1))
    out_3d = out.reshape((d.nworld, m.nv_pad, 1))

    if d.solver_hfactor.shape[1] > 0:
      # Solve-only using stored Cholesky factor
      wp.launch_tiled(
        _adjoint_cholesky_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[d.solver_hfactor, b_3d, m.nv],
        outputs=[out_3d],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )
    else:
      # Full factorize + solve (no stored factor)
      # Pad diagonal for stability
      if m.nv_pad > m.nv:
        wp.launch(
          _padding_h_adjoint,
          dim=(d.nworld, m.nv_pad - m.nv),
          inputs=[m.nv],
          outputs=[d.solver_h],
        )
      hfactor_tmp = wp.zeros((d.nworld, m.nv_pad, m.nv_pad), dtype=float)
      wp.launch_tiled(
        _adjoint_cholesky_full_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[d.solver_h, b_3d, m.nv, hfactor_tmp],
        outputs=[out_3d],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )


def solver_implicit_adjoint(m: types.Model, d: types.Data):
  """Implicit differentiation adjoint for constraint solver.

  Called during tape backward. Reads d.qacc.grad (set by downstream),
  solves H*v = adj_qacc, writes d.qacc_smooth.grad = M*v.
  """
  nv = m.nv
  if nv == 0:
    return

  if d.njmax == 0:
    # Solver was identity (qacc = qacc_smooth), copy adjoint through
    wp.launch(
      _copy_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[d.qacc.grad],
      outputs=[d.qacc_smooth.grad],
    )
    return

  if m.opt.solver != types.SolverType.NEWTON:
    # CG solver: no Hessian stored, fall back to identity
    wp.launch(
      _copy_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[d.qacc.grad],
      outputs=[d.qacc_smooth.grad],
    )
    return

  # Solve H * v = adj_qacc
  v = wp.zeros((d.nworld, m.nv_pad), dtype=float)
  _solve_hessian_system(m, d, d.qacc.grad, v)

  # adj_qacc_smooth = M * v
  support.mul_m(m, d, d.qacc_smooth.grad, v)

  # Phase 3: compute efc-level gradients for collision chain
  if d.njmax > 0:
    efc_J = d.efc.J
    if hasattr(efc_J, "grad") and efc_J.grad is not None:
      wp.launch(
        _efc_J_grad_kernel,
        dim=(d.nworld, d.njmax_pad, m.nv_pad),
        inputs=[v, d.efc.force, d.nefc, m.nv, d.njmax],
        outputs=[efc_J.grad],
      )

    efc_aref = d.efc.aref
    efc_pos = d.efc.pos
    if hasattr(efc_aref, "grad") and efc_aref.grad is not None and hasattr(efc_pos, "grad") and efc_pos.grad is not None:
      wp.launch(
        _efc_pos_grad_kernel,
        dim=d.naconmax,
        inputs=[
          efc_aref.grad,
          d.contact.solref,
          d.contact.solimp,
          d.contact.includemargin,
          d.contact.dist,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.nacon,
          m.opt.timestep,
          m.opt.disableflags,
        ],
        outputs=[efc_pos.grad],
      )
