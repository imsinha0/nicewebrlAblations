import io
import inspect
import random
import sys
import time
from base64 import b64encode
from datetime import datetime
from typing import Any, Callable, Optional, Tuple, Union, get_type_hints

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from flax import serialization, struct
from flax.core import FrozenDict
from nicegui import app, ui
from PIL import Image

from nicewebrl.logging import get_logger

# Type definitions
TIMESTEP = Any
RENDER_FN = Callable[[TIMESTEP], jax.Array]

# Module-level variables
logger = get_logger(__name__)


def get_rng():
    """Initializes a jax.random number generator or gets the latest if already initialized."""
    app.storage.user["seed"] = app.storage.user.get("seed", random.getrandbits(32))
    app.storage.user["rng_splits"] = app.storage.user.get("rng_splits", 0)
    if "rng_key" in app.storage.user:
        rng_key = jnp.array(app.storage.user["rng_key"], dtype=jax.numpy.uint32)
        return rng_key
    else:
        rng_key = jax.random.PRNGKey(app.storage.user["seed"])
        app.storage.user["rng_key"] = rng_key.tolist()
        return rng_key


def new_rng():
    """Return a new jax.random number generator or make a new one if not initialized."""
    app.storage.user["seed"] = app.storage.user.get("seed", random.getrandbits(32))
    if "rng_key" in app.storage.user:
        rng_key = jnp.array(app.storage.user["rng_key"], dtype=jax.numpy.uint32)
        rng_key, rng = jax.random.split(rng_key)
        app.storage.user["rng_key"] = rng_key.tolist()
        app.storage.user["rng_splits"] = app.storage.user.get("rng_splits", 0) + 1
        return rng
    else:
        rng_key = jax.random.PRNGKey(app.storage.user["seed"])
        app.storage.user["rng_key"] = rng_key.tolist()
        return rng_key


def match_types(example: struct.PyTreeNode, data: struct.PyTreeNode):
    def match_types_(ex, d):
        if d is None:
            return None
        else:
            if hasattr(ex, "dtype"):
                return jax.numpy.array(d, dtype=ex.dtype)
            else:
                return d

    return jax.tree_map(match_types_, example, data)


def make_serializable(obj: Any):
    """Convert nested jax objects to serializable python objects"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert JAX array to list
    elif isinstance(obj, jnp.ndarray):
        obj = jax.tree_map(np.array, obj)
        return obj.tolist()  # Convert JAX array to list
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


def base64_npimage(image: np.ndarray):
    image = np.asarray(image)
    buffer = io.BytesIO()
    # Convert to uint8 if float, keep as is if already uint
    if image.dtype.kind == "f":
        image = (image * 255).clip(0, 255).astype("uint8")
    elif image.dtype != np.uint8:
        image = image.astype("uint8")
    Image.fromarray(image).save(buffer, format="JPEG")
    encoded_image = b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded_image


def get_size(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    return sum(
        leaf.nbytes if hasattr(leaf, "nbytes") else sys.getsizeof(leaf) for leaf in leaves
    )


class StepType(jnp.uint8):
    FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
    MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
    LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)


ENV_PARAMS = struct.PyTreeNode


class TimeStep(struct.PyTreeNode):
    state: struct.PyTreeNode

    step_type: StepType
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class TimestepWrapper(object):
    """."""

    def __init__(
        self,
        env,
        autoreset: bool = True,
        use_params: bool = True,
        num_leading_dims: int = 0,
        reset_w_batch_dim: bool = True,
    ):
        self._env = env
        self._autoreset = autoreset
        self._use_params = use_params
        self._num_leading_dims = num_leading_dims
        self._reset_w_batch_dim = reset_w_batch_dim

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(
        self, key: jax.random.PRNGKey, params: Optional[struct.PyTreeNode] = None
    ) -> Tuple[TimeStep, dict]:
        if self._use_params:
            obs, state = self._env.reset(key, params)
        else:
            obs, state = self._env.reset(key)
        # Get shape from first leaf of obs, assuming it's a batch dimension
        if self._reset_w_batch_dim:
            first_leaf = jax.tree_util.tree_leaves(obs)[0]
            shape = first_leaf.shape[: self._num_leading_dims]
        else:
            shape = ()
        timestep = TimeStep(
            state=state,
            observation=obs,
            discount=jnp.ones(shape, dtype=jnp.float32),
            reward=jnp.zeros(shape, dtype=jnp.float32),
            step_type=jnp.full(shape, StepType.FIRST, dtype=StepType.FIRST.dtype),
        )
        return timestep

    def step(
        self,
        key: jax.random.PRNGKey,
        prior_timestep: TimeStep,
        action: Union[int, float],
        params: Optional[struct.PyTreeNode] = None,
    ) -> Tuple[TimeStep, dict]:
        def env_step(prior_timestep_):
            if self._use_params:
                obs, state, reward, done, info = self._env.step(
                    key, prior_timestep_.state, action, params
                )
            else:
                obs, state, reward, done, info = self._env.step(
                    key, prior_timestep_.state, action
                )
            del info
            if type(done) == dict:  # multi-agent
                done = done["__all__"]
            if type(reward) == dict:  # multi-agent
                reward = reward["agent_0"].astype(jnp.float32)
            return TimeStep(
                state=state,
                observation=obs,
                discount=1.0 - jnp.asarray(done).astype(jnp.float32),
                reward=reward,
                step_type=jnp.where(done, StepType.LAST, StepType.MID),
            )

        if self._autoreset:
            if prior_timestep.last():
                timestep = self.reset(key, params)
            else:
                timestep = env_step(prior_timestep)
        else:
            timestep = env_step(prior_timestep)
        return timestep


def try_to_get_actions(env):
    if hasattr(env, "num_actions"):
        if callable(getattr(env, "num_actions")):
            num_actions = env.num_actions()
        else:
            num_actions = env.num_actions
    elif hasattr(env, "action_space"):
        if callable(getattr(env, "action_space")):
            num_actions = env.action_space().n
        else:
            num_actions = env.action_space.n
    else:
        raise ValueError(
            "Cannot determine number of actions for environment. please provide actions"
        )
    actions = jnp.arange(num_actions)
    return actions


class JaxWebEnv:
  def __init__(self, env, actions=None):
    """The main purpose of this class is to wrap env functions without JIT and avoid tracers reaching numpy code."""
    self.env = env
    assert hasattr(env, "reset"), "env needs reset function"
    assert hasattr(env, "step"), "env needs step function"

    if actions is None:
      actions = try_to_get_actions(env)

    # Ensure actions is a plain Python list of Python ints (so iterating won't create tracers)
    # This prevents the loop from receiving JAX tracers when called under jax transformations.
    actions_arr = np.asarray(actions)
    try:
      # convert to native Python ints
      self._action_list = [int(x) for x in actions_arr.ravel().tolist()]
    except Exception:
      # fallback - iterate and cast each
      self._action_list = [int(x) for x in list(actions_arr)]

    def next_steps(rng, timestep, env_params):
      """
      Compute next timesteps for each action WITHOUT using jax.vmap/jit.
      This uses a plain Python loop and stacks the results at the end into jnp arrays
      to preserve the same output types.
      """
      timesteps_list = []
      infos_list = []
      for a in self._action_list:
        # call env.step with native Python int action
        # keep the same signature env.step(rng, timestep, action, env_params) or whatever your env expects
        try:
          out = env.step(rng, timestep, a, env_params)
        except TypeError:
          # try without params if env.step doesn't accept env_params
          out = env.step(rng, timestep, a)
        # env.step is expected to return (timestep, info) or TimeStep; support both
        if isinstance(out, tuple) and len(out) == 2:
          ts, info = out
        else:
          ts, info = out, {}
        timesteps_list.append(ts)
        infos_list.append(info)

      # Stack the pytrees across the new leading axis (actions). Re-create a pytree where
      # each leaf is stacked along axis 0 so that shape matches previous vmap output.
      # We convert lists of leaves into arrays using jnp.stack to keep types consistent.
      # Helper that stacks corresponding leaves from all timesteps.
      def stack_leaves(*leaves):
        # leaves is a tuple with one leaf per action
        # convert each to array (if needed) and stack
        arrays = [jnp.asarray(l) for l in leaves]
        return jnp.stack(arrays, axis=0)

      stacked_timesteps = jax.tree_util.tree_map(lambda *l: stack_leaves(*l), *timesteps_list)

      # infos_list may contain dicts; keep as a Python list of dicts (or convert similarly if needed)
      return stacked_timesteps
    
    self.reset = env.reset
    self.next_steps = next_steps

  def precompile(self, dummy_env_params: Optional[struct.PyTreeNode] = None) -> None:
      """No-op: kept for API compatibility."""
      print("Skipping JIT precompilation (removed).")

  def precompile_vmap_render_fn(
      self, render_fn: RENDER_FN, dummy_env_params: struct.PyTreeNode
  ) -> RENDER_FN:
      """Just return a vmap’d render function without JIT."""
      vmap_render_fn = jax.vmap(render_fn)
      return vmap_render_fn
