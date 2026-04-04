# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Optional

import warp as wp

from mujoco_warp._src import collision_driver
from mujoco_warp._src import collision_smooth
from mujoco_warp._src import constraint
from mujoco_warp._src import derivative
from mujoco_warp._src import island
from mujoco_warp._src import math
from mujoco_warp._src import passive
from mujoco_warp._src import sensor
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.support import next_act
from mujoco_warp._src.support import xfrc_accumulate
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import TileSet
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import vec10f
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": True})


@wp.kernel
def _next_position(
  # Model:
  opt_timestep: wp.array(dtype=float),
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  qvel_in: wp.array2d(dtype=float),
  # In:
  qvel_scale_in: float,
  # Data out:
  qpos_out: wp.array2d(dtype=float),
):
  worldid, jntid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  jnttype = jnt_type[jntid]
  qpos_adr = jnt_qposadr[jntid]
  dof_adr = jnt_dofadr[jntid]
  qpos = qpos_in[worldid]
  qpos_next = qpos_out[worldid]
  qvel = qvel_in[worldid]

  if jnttype == JointType.FREE:
    qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
    qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

    qpos_new = qpos_pos + timestep * qvel_lin

    qpos_quat = wp.quat(
      qpos[qpos_adr + 3],
      qpos[qpos_adr + 4],
      qpos[qpos_adr + 5],
      qpos[qpos_adr + 6],
    )
    qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_next[qpos_adr + 0] = qpos_new[0]
    qpos_next[qpos_adr + 1] = qpos_new[1]
    qpos_next[qpos_adr + 2] = qpos_new[2]
    qpos_next[qpos_adr + 3] = qpos_quat_new[0]
    qpos_next[qpos_adr + 4] = qpos_quat_new[1]
    qpos_next[qpos_adr + 5] = qpos_quat_new[2]
    qpos_next[qpos_adr + 6] = qpos_quat_new[3]

  elif jnttype == JointType.BALL:
    qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
    qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_next[qpos_adr + 0] = qpos_quat_new[0]
    qpos_next[qpos_adr + 1] = qpos_quat_new[1]
    qpos_next[qpos_adr + 2] = qpos_quat_new[2]
    qpos_next[qpos_adr + 3] = qpos_quat_new[3]

  else:  # if jnt_type in (JointType.HINGE, JointType.SLIDE):
    qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in


@wp.kernel
def _next_velocity(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_in: wp.array2d(dtype=float),
  # In:
  qacc_scale_in: float,
  # Data out:
  qvel_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep


@wp.kernel
def _next_activation(
  # Model:
  opt_timestep: wp.array(dtype=float),
  actuator_dyntype: wp.array(dtype=int),
  actuator_actadr: wp.array(dtype=int),
  actuator_actnum: wp.array(dtype=int),
  actuator_actlimited: wp.array(dtype=bool),
  actuator_dynprm: wp.array2d(dtype=vec10f),
  actuator_actrange: wp.array2d(dtype=wp.vec2),
  # Data in:
  act_in: wp.array2d(dtype=float),
  act_dot_in: wp.array2d(dtype=float),
  # In:
  act_dot_scale: float,
  limit: bool,
  # Data out:
  act_out: wp.array2d(dtype=float),
):
  worldid, uid = wp.tid()
  opt_timestep_id = worldid % opt_timestep.shape[0]
  actuator_dynprm_id = worldid % actuator_dynprm.shape[0]
  actuator_actrange_id = worldid % actuator_actrange.shape[0]
  actadr = actuator_actadr[uid]
  actnum = actuator_actnum[uid]
  for j in range(actadr, actadr + actnum):
    act = next_act(
      opt_timestep[opt_timestep_id],
      actuator_dyntype[uid],
      actuator_dynprm[actuator_dynprm_id, uid],
      actuator_actrange[actuator_actrange_id, uid],
      act_in[worldid, j],
      act_dot_in[worldid, j],
      act_dot_scale,
      limit and actuator_actlimited[uid],
    )
    act_out[worldid, j] = act


@wp.kernel
def _next_time(
  # Model:
  opt_timestep: wp.array(dtype=float),
  is_sparse: bool,
  # Data in:
  nefc_in: wp.array(dtype=int),
  time_in: wp.array(dtype=float),
  efc_J_rownnz_in: wp.array2d(dtype=int),
  efc_J_rowadr_in: wp.array2d(dtype=int),
  nworld_in: int,
  naconmax_in: int,
  njmax_in: int,
  njmax_nnz_in: int,
  nacon_in: wp.array(dtype=int),
  ncollision_in: wp.array(dtype=int),
  # Data out:
  time_out: wp.array(dtype=float),
):
  worldid = wp.tid()
  time_out[worldid] = time_in[worldid] + opt_timestep[worldid % opt_timestep.shape[0]]
  nefc = nefc_in[worldid]

  if nefc > njmax_in:
    wp.printf("nefc overflow - please increase njmax to %u\n", nefc)
  elif nefc > 0 and is_sparse:
    efcid = wp.min(nefc, njmax_in) - 1
    efc_nnz = efc_J_rowadr_in[worldid, efcid] + efc_J_rownnz_in[worldid, efcid]
    if efc_nnz > njmax_nnz_in:
      wp.printf("njmax_nnz overflow - please increase njmax_nnz to %u\n", efc_nnz)

  if worldid == 0:
    ncollision = ncollision_in[0]
    if ncollision > naconmax_in:
      nconmax = int(wp.ceil(float(ncollision) / float(nworld_in)))
      wp.printf("broadphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, ncollision)

    if nacon_in[0] > naconmax_in:
      nconmax = int(wp.ceil(float(nacon_in[0]) / float(nworld_in)))
      wp.printf("narrowphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, nacon_in[0])


def _advance(m: Model, d: Data, qacc: wp.array, qvel: Optional[wp.array] = None):
  """Advance state and time given activation derivatives and acceleration."""
  # TODO(team): can we assume static timesteps?

  # Clone arrays used as both input and output so that Warp's tape retains the
  # original values for correct reverse-mode AD.  Guard with requires_grad so
  # non-AD paths pay zero overhead.
  act_in = wp.clone(d.act) if d.act.requires_grad else d.act
  qvel_prev = wp.clone(d.qvel) if d.qvel.requires_grad else d.qvel
  qpos_prev = wp.clone(d.qpos) if d.qpos.requires_grad else d.qpos

  # advance activations
  wp.launch(
    _next_activation,
    dim=(d.nworld, m.nu),
    inputs=[
      m.opt.timestep,
      m.actuator_dyntype,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_actlimited,
      m.actuator_dynprm,
      m.actuator_actrange,
      act_in,
      d.act_dot,
      1.0,
      True,
    ],
    outputs=[d.act],
  )

  wp.launch(
    _next_velocity,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.timestep, qvel_prev, qacc, 1.0],
    outputs=[d.qvel],
  )

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  qvel_in = qvel or d.qvel

  wp.launch(
    _next_position,
    dim=(d.nworld, m.njnt),
    inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, qpos_prev, qvel_in, 1.0],
    outputs=[d.qpos],
  )

  wp.launch(
    _next_time,
    dim=d.nworld,
    inputs=[
      m.opt.timestep,
      m.is_sparse,
      d.nefc,
      d.time,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.nworld,
      d.naconmax,
      d.njmax,
      d.njmax_nnz,
      d.nacon,
      d.ncollision,
    ],
    outputs=[d.time],
  )

  # Use _nograd_copy: warmstart is a numerical hint, not a gradient path.
  # wp.copy would be tracked on the tape and create cross-substep gradient
  # leaks through the shared d.qacc_warmstart array.
  wp.launch(
    support._nograd_copy, dim=(d.nworld, qacc.shape[1]),
    inputs=[qacc], outputs=[d.qacc_warmstart],
  )


@wp.kernel
def _euler_damp_qfrc_sparse(
  # Model:
  opt_timestep: wp.array(dtype=float),
  dof_Madr: wp.array(dtype=int),
  dof_damping: wp.array2d(dtype=float),
  # Out:
  qM_integration_out: wp.array3d(dtype=float),
):
  worldid, tid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  adr = dof_Madr[tid]
  qM_integration_out[worldid, 0, adr] += timestep * dof_damping[worldid % dof_damping.shape[0], tid]


@wp.kernel
def _euler_damp_qfrc_dense(
  # Model:
  opt_timestep: wp.array(dtype=float),
  dof_damping: wp.array2d(dtype=float),
  # Out:
  qM_integration_out: wp.array3d(dtype=float),
):
  """Add dt * damping to diagonal of dense (nworld, nv, nv) mass matrix."""
  worldid, tid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  qM_integration_out[worldid, tid, tid] += timestep * dof_damping[worldid % dof_damping.shape[0], tid]


@cache_kernel
def _tile_euler_dense(tile: TileSet):
  @wp.kernel(module="unique", enable_backward=False)
  def euler_dense(
    # Model:
    opt_timestep: wp.array(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    # Data in:
    qM_in: wp.array3d(dtype=float),
    efc_Ma_in: wp.array2d(dtype=float),
    # In:
    adr_in: wp.array(dtype=int),
    # Data out:
    qacc_out: wp.array2d(dtype=float),
  ):
    worldid, nodeid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    TILE_SIZE = wp.static(tile.size)

    dofid = adr_in[nodeid]
    M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))
    damping_tile = wp.tile_load(dof_damping[worldid % dof_damping.shape[0]], shape=(TILE_SIZE,), offset=(dofid,))
    damping_scaled = damping_tile * timestep
    qm_integration_tile = wp.tile_diag_add(M_tile, damping_scaled)

    Ma_tile = wp.tile_load(efc_Ma_in[worldid], shape=(TILE_SIZE,), offset=(dofid,))
    L_tile = wp.tile_cholesky(qm_integration_tile)
    qacc_tile = wp.tile_cholesky_solve(L_tile, Ma_tile)
    wp.tile_store(qacc_out[worldid], qacc_tile, offset=(dofid))

  return euler_dense


@event_scope
def euler(m: Model, d: Data):
  """Euler integrator, semi-implicit in velocity."""
  # integrate damping implicitly
  if not (m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER)):
    ad_active = d.qpos.requires_grad
    qacc = wp.empty((d.nworld, m.nv), dtype=float, requires_grad=ad_active)
    if m.is_sparse:
      qM = wp.clone(d.qM)
      qLD = wp.empty((d.nworld, 1, m.nC), dtype=float, requires_grad=ad_active)
      qLDiagInv = wp.empty((d.nworld, m.nv), dtype=float, requires_grad=ad_active)
      wp.launch(
        _euler_damp_qfrc_sparse,
        dim=(d.nworld, m.nv),
        inputs=[m.opt.timestep, m.dof_Madr, m.dof_damping],
        outputs=[qM],
      )
      smooth.factor_solve_i(m, d, qM, qLD, qLDiagInv, qacc, d.efc.Ma)
    else:
      for tile in m.qM_tiles:
        wp.launch_tiled(
          _tile_euler_dense(tile),
          dim=(d.nworld, tile.adr.size),
          inputs=[m.opt.timestep, m.dof_damping, d.qM, d.efc.Ma, tile.adr],
          outputs=[qacc],
          block_dim=m.block_dim.euler_dense,
        )
    _record_solver_adjoint(m, d, qacc_array=qacc)
    _record_euler_damp_adjoint(m, d, qacc)
    _advance(m, d, qacc)
  else:
    _record_solver_adjoint(m, d, qacc_array=d.qacc)
    _advance(m, d, d.qacc)


def _rk_perturb_state(
  m: Model,
  d: Data,
  scale: float,
  qpos_t0: wp.array2d(dtype=float),
  qvel_t0: wp.array2d(dtype=float),
  act_t0: Optional[wp.array] = None,
):
  # position
  wp.launch(
    _next_position,
    dim=(d.nworld, m.njnt),
    inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, qpos_t0, d.qvel, scale],
    outputs=[d.qpos],
  )

  # velocity
  wp.launch(
    _next_velocity,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.timestep, qvel_t0, d.qacc, scale],
    outputs=[d.qvel],
  )

  # activation
  if m.na and act_t0 is not None:
    wp.launch(
      _next_activation,
      dim=(d.nworld, m.nu),
      inputs=[
        m.opt.timestep,
        m.actuator_dyntype,
        m.actuator_actadr,
        m.actuator_actnum,
        m.actuator_actlimited,
        m.actuator_dynprm,
        m.actuator_actrange,
        act_t0,
        d.act_dot,
        scale,
        False,
      ],
      outputs=[d.act],
    )


@wp.kernel
def _rk_accumulate_velocity_acceleration(
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_in: wp.array2d(dtype=float),
  # In:
  scale: float,
  # Data out:
  qvel_out: wp.array2d(dtype=float),
  qacc_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]
  qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]


@wp.kernel
def _rk_accumulate_activation_velocity(
  # Data in:
  act_dot_in: wp.array2d(dtype=float),
  # In:
  scale: float,
  # Data out:
  act_dot_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()
  act_dot_out[worldid, actid] += scale * act_dot_in[worldid, actid]


def _rk_accumulate(
  m: Model,
  d: Data,
  scale: float,
  qvel_rk: wp.array2d(dtype=float),
  qacc_rk: wp.array2d(dtype=float),
  act_dot_rk: Optional[wp.array] = None,
):
  """Computes one term of 1/6 k_1 + 1/3 k_2 + 1/3 k_3 + 1/6 k_4."""
  wp.launch(
    _rk_accumulate_velocity_acceleration,
    dim=(d.nworld, m.nv),
    inputs=[d.qvel, d.qacc, scale],
    outputs=[qvel_rk, qacc_rk],
  )

  if m.na and act_dot_rk is not None:
    wp.launch(
      _rk_accumulate_activation_velocity,
      dim=(d.nworld, m.na),
      inputs=[d.act_dot, scale],
      outputs=[act_dot_rk],
    )


@event_scope
def rungekutta4(m: Model, d: Data):
  """Runge-Kutta explicit order 4 integrator."""
  # RK4 tableau
  A = [0.5, 0.5, 1.0]  # diagonal only
  B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

  ad_active = d.qpos.requires_grad
  qpos_t0 = wp.clone(d.qpos)
  qvel_t0 = wp.clone(d.qvel)
  qvel_rk = wp.zeros((d.nworld, m.nv), dtype=float, requires_grad=ad_active)
  qacc_rk = wp.zeros((d.nworld, m.nv), dtype=float, requires_grad=ad_active)

  if m.na:
    act_t0 = wp.clone(d.act)
    act_dot_rk = wp.zeros((d.nworld, m.na), dtype=float, requires_grad=ad_active)
  else:
    act_t0 = None
    act_dot_rk = None

  _rk_accumulate(m, d, B[0], qvel_rk, qacc_rk, act_dot_rk)

  for i in range(3):
    a, b = float(A[i]), B[i + 1]
    _rk_perturb_state(m, d, a, qpos_t0, qvel_t0, act_t0)
    forward(m, d)
    _rk_accumulate(m, d, b, qvel_rk, qacc_rk, act_dot_rk)

  wp.copy(d.qpos, qpos_t0)
  wp.copy(d.qvel, qvel_t0)

  if m.na:
    wp.copy(d.act, act_t0)
    wp.copy(d.act_dot, act_dot_rk)

  _record_solver_adjoint(m, d, qacc_array=qacc_rk)
  _advance(m, d, qacc_rk, qvel_rk)


@event_scope
def implicit(m: Model, d: Data):
  """Integrates fully implicit in velocity."""
  if ~(m.opt.disableflags | ~(DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER)):
    ad_active = d.qpos.requires_grad
    if m.is_sparse:
      qDeriv = wp.empty((d.nworld, 1, m.nM), dtype=float)
      qLD = wp.empty((d.nworld, 1, m.nC), dtype=float)
    else:
      qDeriv = wp.empty(d.qM.shape, dtype=float)
      qLD = wp.empty(d.qM.shape, dtype=float)
    qLDiagInv = wp.empty((d.nworld, m.nv), dtype=float)
    derivative.deriv_smooth_vel(m, d, qDeriv)
    qacc = wp.empty((d.nworld, m.nv), dtype=float, requires_grad=ad_active)
    smooth.factor_solve_i(m, d, qDeriv, qLD, qLDiagInv, qacc, d.efc.Ma)
    _record_solver_adjoint(m, d, qacc_array=qacc)
    _advance(m, d, qacc)
  else:
    _record_solver_adjoint(m, d, qacc_array=d.qacc)
    _advance(m, d, d.qacc)


@event_scope
def fwd_position(m: Model, d: Data, factorize: bool = True):
  """Position-dependent computations.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize interia matrix.
  """
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.flex(m, d)
  smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  if factorize:
    smooth.factor_m(m, d)
  if m.opt.run_collision_detection:
    collision_driver.collision(m, d)
    # Phase 3: smooth collision recomputation for AD
    tape = wp._src.context.runtime.tape
    if tape is not None and d.qpos.requires_grad:
      collision_smooth.smooth_recompute_contacts(m, d)
  constraint.make_constraint(m, d)
  # Phase 3: differentiable constraint assembly for AD
  tape = wp._src.context.runtime.tape
  if tape is not None and d.qpos.requires_grad:
    collision_smooth.smooth_contact_to_efc(m, d)
  # TODO(team): remove False after island features are more complete
  if False and not (m.opt.disableflags & DisableBit.ISLAND):
    island.island(m, d)
  smooth.transmission(m, d)


@wp.kernel
def _actuator_velocity(
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  moment_rownnz_in: wp.array2d(dtype=int),
  moment_rowadr_in: wp.array2d(dtype=int),
  moment_colind_in: wp.array2d(dtype=int),
  actuator_moment_in: wp.array2d(dtype=float),
  # Data out:
  actuator_velocity_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  rownnz = moment_rownnz_in[worldid, actid]
  rowadr = moment_rowadr_in[worldid, actid]

  vel = float(0.0)
  for i in range(rownnz):
    sparseid = rowadr + i
    colind = moment_colind_in[worldid, sparseid]
    vel += actuator_moment_in[worldid, sparseid] * qvel_in[worldid, colind]

  actuator_velocity_out[worldid, actid] = vel


@wp.kernel
def _tendon_velocity(
  # Model:
  ten_J_rownnz: wp.array(dtype=int),
  ten_J_rowadr: wp.array(dtype=int),
  ten_J_colind: wp.array(dtype=int),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  ten_J_in: wp.array2d(dtype=float),
  # Data out:
  ten_velocity_out: wp.array2d(dtype=float),
):
  worldid, tenid = wp.tid()

  velocity = float(0.0)
  rownnz = ten_J_rownnz[tenid]
  rowadr = ten_J_rowadr[tenid]
  for i in range(rownnz):
    sparseid = rowadr + i
    J = ten_J_in[worldid, sparseid]
    if J != 0.0:
      colind = ten_J_colind[sparseid]
      velocity += J * qvel_in[worldid, colind]

  ten_velocity_out[worldid, tenid] = velocity


@event_scope
def fwd_velocity(m: Model, d: Data):
  """Velocity-dependent computations."""
  wp.launch(
    _actuator_velocity,
    dim=(d.nworld, m.nu),
    inputs=[d.qvel, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment],
    outputs=[d.actuator_velocity],
    block_dim=m.block_dim.actuator_velocity,
  )

  wp.launch(
    _tendon_velocity,
    dim=(d.nworld, m.ntendon),
    inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.qvel, d.ten_J],
    outputs=[d.ten_velocity],
  )

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)
  smooth.tendon_bias(m, d, d.qfrc_bias)


@wp.kernel
def _actuator_force(
  # Model:
  na: int,
  opt_timestep: wp.array(dtype=float),
  actuator_dyntype: wp.array(dtype=int),
  actuator_gaintype: wp.array(dtype=int),
  actuator_biastype: wp.array(dtype=int),
  actuator_actadr: wp.array(dtype=int),
  actuator_actnum: wp.array(dtype=int),
  actuator_ctrllimited: wp.array(dtype=bool),
  actuator_forcelimited: wp.array(dtype=bool),
  actuator_actlimited: wp.array(dtype=bool),
  actuator_dynprm: wp.array2d(dtype=vec10f),
  actuator_gainprm: wp.array2d(dtype=vec10f),
  actuator_biasprm: wp.array2d(dtype=vec10f),
  actuator_actearly: wp.array(dtype=bool),
  actuator_ctrlrange: wp.array2d(dtype=wp.vec2),
  actuator_forcerange: wp.array2d(dtype=wp.vec2),
  actuator_actrange: wp.array2d(dtype=wp.vec2),
  actuator_acc0: wp.array2d(dtype=float),
  actuator_lengthrange: wp.array2d(dtype=wp.vec2),
  # Data in:
  act_in: wp.array2d(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  actuator_length_in: wp.array2d(dtype=float),
  actuator_velocity_in: wp.array2d(dtype=float),
  # In:
  dsbl_clampctrl: int,
  # Data out:
  act_dot_out: wp.array2d(dtype=float),
  actuator_force_out: wp.array2d(dtype=float),
):
  worldid, uid = wp.tid()

  actuator_ctrlrange_id = worldid % actuator_ctrlrange.shape[0]

  ctrl = ctrl_in[worldid, uid]

  if actuator_ctrllimited[uid] and not dsbl_clampctrl:
    ctrlrange = actuator_ctrlrange[actuator_ctrlrange_id, uid]
    ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])
  ctrl_act = ctrl

  act_first = actuator_actadr[uid]
  if na and act_first >= 0:
    act_last = act_first + actuator_actnum[uid] - 1
    dyntype = actuator_dyntype[uid]
    dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]

    if dyntype == DynType.INTEGRATOR:
      act_dot = ctrl
    elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:
      act = act_in[worldid, act_last]
      act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)
    elif dyntype == DynType.MUSCLE:
      dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]
      act = act_in[worldid, act_last]
      act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)
    elif dyntype == DynType.USER:
      act_dot = 0.0  # set by act_dyn_callback
    else:  # DynType.NONE
      act_dot = 0.0

    act_dot_out[worldid, act_last] = act_dot

    if actuator_actearly[uid]:
      if dyntype == DynType.INTEGRATOR or dyntype == DynType.NONE:
        act = act_in[worldid, act_last]

      ctrl_act = next_act(
        opt_timestep[worldid % opt_timestep.shape[0]],
        dyntype,
        dynprm,
        actuator_actrange[worldid % actuator_actrange.shape[0], uid],
        act,
        act_dot,
        1.0,
        actuator_actlimited[uid],
      )
    else:
      ctrl_act = act_in[worldid, act_last]

  length = actuator_length_in[worldid, uid]
  velocity = actuator_velocity_in[worldid, uid]

  # gain
  gaintype = actuator_gaintype[uid]
  gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]

  gain = 0.0
  if gaintype == GainType.FIXED:
    gain = gainprm[0]
  elif gaintype == GainType.AFFINE:
    gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity
  elif gaintype == GainType.MUSCLE:
    acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]
    lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]
    gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)
  # GainType.USER: gain stays 0, modified by act_gain_callback

  # bias
  biastype = actuator_biastype[uid]
  biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]

  bias = 0.0  # BiasType.NONE or BiasType.USER (modified by act_bias_callback)
  if biastype == BiasType.AFFINE:
    bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity
  elif biastype == BiasType.MUSCLE:
    acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]
    lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]
    bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)

  force = gain * ctrl_act + bias

  if actuator_forcelimited[uid]:
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], uid]
    force = wp.clamp(force, forcerange[0], forcerange[1])

  actuator_force_out[worldid, uid] = force


@wp.kernel
def _tendon_actuator_force(
  # Model:
  actuator_trntype: wp.array(dtype=int),
  actuator_trnid: wp.array(dtype=wp.vec2i),
  # Data in:
  actuator_force_in: wp.array2d(dtype=float),
  # Out:
  ten_actfrc_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  if actuator_trntype[actid] == TrnType.TENDON:
    tenid = actuator_trnid[actid][0]
    # TODO(team): only compute for tendons with force limits?
    wp.atomic_add(ten_actfrc_out[worldid], tenid, actuator_force_in[worldid, actid])


@wp.kernel
def _tendon_actuator_force_clamp(
  # Model:
  tendon_actfrclimited: wp.array(dtype=bool),
  tendon_actfrcrange: wp.array2d(dtype=wp.vec2),
  actuator_trntype: wp.array(dtype=int),
  actuator_trnid: wp.array(dtype=wp.vec2i),
  # In:
  ten_actfrc_in: wp.array2d(dtype=float),
  # Data out:
  actuator_force_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  if actuator_trntype[actid] == TrnType.TENDON:
    tenid = actuator_trnid[actid][0]
    if tendon_actfrclimited[tenid]:
      ten_actfrc = ten_actfrc_in[worldid, tenid]
      actfrcrange = tendon_actfrcrange[worldid % tendon_actfrcrange.shape[0], tenid]

      if ten_actfrc < actfrcrange[0]:
        actuator_force_out[worldid, actid] = actuator_force_out[worldid, actid] * (actfrcrange[0] / ten_actfrc)
      elif ten_actfrc > actfrcrange[1]:
        actuator_force_out[worldid, actid] = actuator_force_out[worldid, actid] * (actfrcrange[1] / ten_actfrc)


@wp.kernel
def _qfrc_actuator(
  # Data in:
  moment_rownnz_in: wp.array2d(dtype=int),
  moment_rowadr_in: wp.array2d(dtype=int),
  moment_colind_in: wp.array2d(dtype=int),
  actuator_moment_in: wp.array2d(dtype=float),
  actuator_force_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_actuator_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  rownnz = moment_rownnz_in[worldid, actid]
  rowadr = moment_rowadr_in[worldid, actid]

  for i in range(rownnz):
    sparseid = rowadr + i
    colind = moment_colind_in[worldid, sparseid]
    qfrc = actuator_moment_in[worldid, sparseid] * actuator_force_in[worldid, actid]
    wp.atomic_add(qfrc_actuator_out[worldid], colind, qfrc)


@wp.kernel
def _qfrc_actuator_gravcomp_limits(
  # Model:
  ngravcomp: int,
  jnt_actfrclimited: wp.array(dtype=bool),
  jnt_actgravcomp: wp.array(dtype=int),
  jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
  dof_jntid: wp.array(dtype=int),
  # Data in:
  qfrc_gravcomp_in: wp.array2d(dtype=float),
  qfrc_actuator_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_actuator_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  jntid = dof_jntid[dofid]

  qfrc = qfrc_actuator_in[worldid, dofid]

  # actuator-level gravity compensation, skip if added as passive force
  if ngravcomp and jnt_actgravcomp[jntid]:
    qfrc += qfrc_gravcomp_in[worldid, dofid]

  # limits
  if jnt_actfrclimited[jntid]:
    frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]
    qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])

  qfrc_actuator_out[worldid, dofid] = qfrc


@event_scope
def fwd_actuation(m: Model, d: Data):
  """Actuation-dependent computations."""
  if not m.nu or (m.opt.disableflags & DisableBit.ACTUATION):
    d.act_dot.zero_()
    d.qfrc_actuator.zero_()
    return

  wp.launch(
    _actuator_force,
    dim=(d.nworld, m.nu),
    inputs=[
      m.na,
      m.opt.timestep,
      m.actuator_dyntype,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_ctrllimited,
      m.actuator_forcelimited,
      m.actuator_actlimited,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_actearly,
      m.actuator_ctrlrange,
      m.actuator_forcerange,
      m.actuator_actrange,
      m.actuator_acc0,
      m.actuator_lengthrange,
      d.act,
      d.ctrl,
      d.actuator_length,
      d.actuator_velocity,
      m.opt.disableflags & DisableBit.CLAMPCTRL,
    ],
    outputs=[d.act_dot, d.actuator_force],
  )

  if m.callback.act_dyn:
    m.callback.act_dyn(m, d)
  if m.callback.act_gain:
    m.callback.act_gain(m, d)
  if m.callback.act_bias:
    m.callback.act_bias(m, d)

  if m.ntendon:
    # total actuator force at tendon
    ten_actfrc = wp.zeros((d.nworld, m.ntendon), dtype=float)
    wp.launch(
      _tendon_actuator_force,
      dim=(d.nworld, m.nu),
      inputs=[m.actuator_trntype, m.actuator_trnid, d.actuator_force],
      outputs=[ten_actfrc],
    )

    wp.launch(
      _tendon_actuator_force_clamp,
      dim=(d.nworld, m.nu),
      inputs=[m.tendon_actfrclimited, m.tendon_actfrcrange, m.actuator_trntype, m.actuator_trnid, ten_actfrc],
      outputs=[d.actuator_force],
    )

  # TODO(team): optimize performance
  d.qfrc_actuator.zero_()
  wp.launch(
    _qfrc_actuator,
    dim=(d.nworld, m.nu),
    inputs=[
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.actuator_force,
    ],
    outputs=[d.qfrc_actuator],
  )
  # clone to break input/output aliasing for correct AD; skip when not
  # recording a backward tape to avoid unnecessary allocation + copy.
  qfrc_actuator_in = wp.clone(d.qfrc_actuator) if d.qfrc_actuator.requires_grad else d.qfrc_actuator
  wp.launch(
    _qfrc_actuator_gravcomp_limits,
    dim=(d.nworld, m.nv),
    inputs=[
      m.ngravcomp,
      m.jnt_actfrclimited,
      m.jnt_actgravcomp,
      m.jnt_actfrcrange,
      m.dof_jntid,
      d.qfrc_gravcomp,
      qfrc_actuator_in,
    ],
    outputs=[d.qfrc_actuator],
  )


@wp.kernel
def _qfrc_smooth(
  # Data in:
  qfrc_applied_in: wp.array2d(dtype=float),
  qfrc_bias_in: wp.array2d(dtype=float),
  qfrc_passive_in: wp.array2d(dtype=float),
  qfrc_actuator_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_smooth_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qfrc_smooth_out[worldid, dofid] = (
    qfrc_passive_in[worldid, dofid]
    - qfrc_bias_in[worldid, dofid]
    + qfrc_actuator_in[worldid, dofid]
    + qfrc_applied_in[worldid, dofid]
  )


@event_scope
def fwd_acceleration(m: Model, d: Data, factorize: bool = False):
  """Add up all non-constraint forces, compute qacc_smooth.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize inertia matrix.
  """
  wp.launch(
    _qfrc_smooth,
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_applied, d.qfrc_bias, d.qfrc_passive, d.qfrc_actuator],
    outputs=[d.qfrc_smooth],
  )
  xfrc_accumulate(m, d, d.qfrc_smooth)

  if factorize:
    smooth.factor_solve_i(m, d, d.qM, d.qLD, d.qLDiagInv, d.qacc_smooth, d.qfrc_smooth)
  else:
    smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)

  # Custom adjoint for M_inv solve on the dense path.
  # The tile Cholesky kernels have enable_backward=False, so the tape cannot
  # propagate qacc_smooth.grad -> qfrc_smooth.grad automatically.  We record
  # a callback that performs the VJP: qfrc_smooth.grad += M_inv * qacc_smooth.grad
  # (M is symmetric so M_inv^T = M_inv).
  _record_fwd_accel_adjoint(m, d)


def _record_fwd_accel_adjoint(m: Model, d: Data):
  """Record custom adjoint for the M_inv solve in fwd_acceleration.

  On the dense path, _tile_cholesky_factorize_solve has enable_backward=False.
  This record_func propagates qacc_smooth.grad -> qfrc_smooth.grad via M_inv,
  using the already-factored d.qLD from the forward pass.

  Array references are captured at record time (not through d) so that
  intermediate array cloning between substeps routes each substep's adjoint
  to the correct .grad memory.
  """
  tape = wp._src.context.runtime.tape
  if tape is not None and d.qpos.requires_grad:
    from mujoco_warp._src.adjoint import _accumulate_grad_kernel

    # Capture current array refs for correct gradient isolation across substeps
    qacc_smooth_ref = d.qacc_smooth
    qfrc_smooth_ref = d.qfrc_smooth

    def _adjoint(m=m, d=d, qacc_smooth=qacc_smooth_ref, qfrc_smooth=qfrc_smooth_ref):
      adj_qacc_smooth = qacc_smooth.grad
      if adj_qacc_smooth is None:
        return

      # qfrc_smooth.grad += M_inv * qacc_smooth.grad
      tmp = wp.zeros_like(qfrc_smooth)
      smooth.solve_m(m, d, tmp, adj_qacc_smooth)
      if qfrc_smooth.grad is None:
        qfrc_smooth.grad = tmp
      else:
        wp.launch(
          _accumulate_grad_kernel,
          dim=(d.nworld, m.nv),
          inputs=[tmp],
          outputs=[qfrc_smooth.grad],
        )

    tape.record_func(_adjoint, [qacc_smooth_ref, qfrc_smooth_ref])


def _record_solver_adjoint(m: Model, d: Data, qacc_array=None):
  """Record the solver implicit differentiation adjoint on the active tape.

  Args:
    qacc_array: The array whose .grad will receive the incoming adjoint from
                the integrator backward. Defaults to d.qacc (correct when
                the integrator uses d.qacc directly, e.g. eulerdamp disabled).
                Integrators that create a local qacc must pass it here.

  Array references are captured at record time so that intermediate array
  cloning between substeps routes each substep's adjoint correctly.
  """
  tape = wp._src.context.runtime.tape
  if tape is not None and d.qpos.requires_grad:
    from mujoco_warp._src.adjoint import solver_implicit_adjoint

    if qacc_array is None:
      qacc_array = d.qacc

    # Capture qacc_smooth ref at record time for gradient isolation
    qacc_smooth_ref = d.qacc_smooth

    tape.record_func(
      lambda m=m, d=d, qa=qacc_array, qs=qacc_smooth_ref: solver_implicit_adjoint(
        m, d, qacc_array=qa, qacc_smooth_ref=qs
      ),
      [qacc_array, qacc_smooth_ref],
    )


def _record_euler_damp_adjoint(m: Model, d: Data, qacc: wp.array):
  """Record euler-damping adjoint transformation on the active tape.

  During backward, transforms qacc.grad from the raw integrator adjoint
  into the correct adjoint that accounts for the (M+dt*D)^{-1}*M
  transformation in the euler implicit damping solve.

  Forward:  qacc_local = (M + dt*D)^{-1} * M * d.qacc
  Adjoint:  adj_d_qacc = M * (M + dt*D)^{-1} * adj_qacc_local

  This callback runs between _advance backward (which sets qacc.grad)
  and _record_solver_adjoint backward (which reads qacc.grad).
  """
  tape = wp._src.context.runtime.tape
  if tape is None or not d.qpos.requires_grad:
    return

  # Capture the forward-pass mass matrix reference at record time.
  # _isolate_intermediates_for_ad() allocates fresh d.qM each substep,
  # so this captures the correct per-substep mass matrix.
  qM_ref = d.qM
  qacc_ref = qacc

  def _adjoint(m=m, d=d, qM=qM_ref, qacc_arr=qacc_ref):
    adj_qacc = qacc_arr.grad
    if adj_qacc is None:
      return

    nv = m.nv

    # Step 1: Construct M_damp = M + dt*D
    qM_damp = wp.clone(qM)
    if m.is_sparse:
      wp.launch(
        _euler_damp_qfrc_sparse,
        dim=(d.nworld, nv),
        inputs=[m.opt.timestep, m.dof_Madr, m.dof_damping],
        outputs=[qM_damp],
      )
    else:
      wp.launch(
        _euler_damp_qfrc_dense,
        dim=(d.nworld, nv),
        inputs=[m.opt.timestep, m.dof_damping],
        outputs=[qM_damp],
      )

    # Step 2: Solve (M + dt*D) * tmp = adj_qacc
    qLD_tmp = wp.zeros_like(d.qLD)
    qLDiagInv_tmp = wp.zeros((d.nworld, nv), dtype=float)
    tmp = wp.zeros((d.nworld, nv), dtype=float)
    smooth.factor_solve_i(m, d, qM_damp, qLD_tmp, qLDiagInv_tmp, tmp, adj_qacc)

    # Step 3: result = M * tmp (using original undamped mass matrix)
    result = wp.zeros((d.nworld, nv), dtype=float)
    support.mul_m(m, d, result, tmp, M=qM)

    # Step 4: Overwrite qacc.grad with the corrected adjoint
    wp.copy(qacc_arr.grad, result)

  tape.record_func(_adjoint, [qacc_ref])


@event_scope
def forward(m: Model, d: Data, record_solver_adjoint: bool = True):
  """Forward dynamics.

  Args:
    record_solver_adjoint: If True, record the solver implicit differentiation
        adjoint on the tape. Set to False when called from step() since the
        integrator records its own adjoint at the correct tape position.
  """
  energy = m.opt.enableflags & EnableBit.ENERGY

  fwd_position(m, d, factorize=False)
  d.sensordata.zero_()
  sensor.sensor_pos(m, d)
  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)
  fwd_actuation(m, d)
  fwd_acceleration(m, d, factorize=True)

  solver.solve(m, d)

  # Record implicit differentiation adjoint on the active tape.
  # When called from step(), the integrator handles this instead (at the
  # correct tape position between factor_solve_i and _advance).
  if record_solver_adjoint:
    _record_solver_adjoint(m, d)

  sensor.sensor_acc(m, d)


def _isolate_intermediates_for_ad(m: Model, d: Data):
  """Allocate fresh intermediate arrays for per-substep gradient isolation.

  In tape-all mode (single wp.Tape over multiple step() calls), intermediate
  arrays like qfrc_smooth and qacc_smooth are overwritten each substep but
  share a single .grad array. This causes backward to accumulate adjoint
  contributions from ALL substeps into the same memory (~250,000x amplification
  for 16 substeps).

  By allocating fresh arrays at the start of each step(), each substep writes
  to its own memory. The tape records operations on these unique arrays, and
  backward routes each substep's adjoint to the correct .grad memory.

  Only called when AD is active (d.qpos.requires_grad).

  Every array here appears as an output of wp.launch in the pipeline. If
  shared across substeps, Warp's adjoint zeroes the output .grad during
  the later substep's backward, corrupting the earlier substep's gradient
  chain.  The allocation overhead is ~25KB/step (negligible for GPU).
  """
  nw = d.nworld
  nv = m.nv
  nu = m.nu

  # --- Force arrays ---
  d.qfrc_smooth = wp.zeros((nw, nv), dtype=float, requires_grad=True)
  d.qacc_smooth = wp.zeros((nw, nv), dtype=float, requires_grad=True)
  d.qfrc_actuator = wp.zeros((nw, nv), dtype=float, requires_grad=True)
  d.actuator_force = wp.zeros((nw, nu), dtype=float, requires_grad=True)
  d.qacc = wp.zeros((nw, nv), dtype=float, requires_grad=True)
  d.qfrc_bias = wp.zeros((nw, nv), dtype=float, requires_grad=True)
  d.qfrc_passive = wp.zeros((nw, nv), dtype=float, requires_grad=True)

  # --- Kinematics arrays ---
  # These use Warp vector/matrix dtypes (vec3, mat33, etc.) so use
  # zeros_like to match the exact dtype and shape from the existing arrays.
  d.xpos = wp.zeros_like(d.xpos, requires_grad=True)
  d.xmat = wp.zeros_like(d.xmat, requires_grad=True)
  d.xipos = wp.zeros_like(d.xipos, requires_grad=True)
  d.ximat = wp.zeros_like(d.ximat, requires_grad=True)
  d.subtree_com = wp.zeros_like(d.subtree_com, requires_grad=True)
  d.cinert = wp.zeros_like(d.cinert, requires_grad=True)
  d.cdof = wp.zeros_like(d.cdof, requires_grad=True)
  d.cdof_dot = wp.zeros_like(d.cdof_dot, requires_grad=True)
  d.cvel = wp.zeros_like(d.cvel, requires_grad=True)
  d.crb = wp.zeros_like(d.crb, requires_grad=True)
  d.cacc = wp.zeros_like(d.cacc, requires_grad=True)

  # --- Mass matrix ---
  # Shapes depend on sparse vs dense; zeros_like handles both.
  d.qM = wp.zeros_like(d.qM, requires_grad=True)
  d.qLD = wp.zeros_like(d.qLD, requires_grad=True)
  d.qLDiagInv = wp.zeros((nw, nv), dtype=float, requires_grad=True)

  # --- Geometry / joint kinematics ---
  d.geom_xpos = wp.zeros_like(d.geom_xpos, requires_grad=True)
  d.geom_xmat = wp.zeros_like(d.geom_xmat, requires_grad=True)
  d.xanchor = wp.zeros_like(d.xanchor, requires_grad=True)
  d.xaxis = wp.zeros_like(d.xaxis, requires_grad=True)
  d.subtree_linvel = wp.zeros_like(d.subtree_linvel, requires_grad=True)
  d.subtree_angmom = wp.zeros_like(d.subtree_angmom, requires_grad=True)

  # --- Actuator arrays ---
  d.actuator_velocity = wp.zeros((nw, nu), dtype=float, requires_grad=True)


@event_scope
def step(m: Model, d: Data):
  """Advance simulation."""
  # TODO(team): mj_checkPos
  # TODO(team): mj_checkVel

  # Allocate fresh intermediate arrays when AD is active to prevent
  # cross-substep gradient accumulation in tape-all mode.
  if d.qpos.requires_grad:
    _isolate_intermediates_for_ad(m, d)

  forward(m, d, record_solver_adjoint=False)
  # TODO(team): mj_checkAcc

  if m.opt.integrator == IntegratorType.EULER:
    euler(m, d)
  elif m.opt.integrator == IntegratorType.RK4:
    rungekutta4(m, d)
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    implicit(m, d)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")


@event_scope
def step1(m: Model, d: Data):
  """Advance simulation in two phases: before input is set by user."""
  energy = m.opt.enableflags & EnableBit.ENERGY
  # TODO(team): mj_checkPos
  # TODO(team): mj_checkVel
  fwd_position(m, d)
  d.sensordata.zero_()
  sensor.sensor_pos(m, d)

  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)


@event_scope
def step2(m: Model, d: Data):
  """Advance simulation in two phases: after input is set by user."""
  fwd_actuation(m, d)
  fwd_acceleration(m, d)
  solver.solve(m, d)

  # The solver adjoint record_func is handled by the integrator below,
  # NOT here — see euler()/implicit() for details.

  sensor.sensor_acc(m, d)
  # TODO(team): mj_checkAcc

  # integrate with Euler or implicitfast
  # TODO(team): implicit
  if m.opt.integrator == IntegratorType.IMPLICITFAST:
    implicit(m, d)
  else:
    # note: RK4 defaults to Euler
    euler(m, d)
