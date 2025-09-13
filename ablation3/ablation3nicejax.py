# nicejax.py
import io
import inspect
import random
import sys
import time
from base64 import b64encode
from datetime import datetime
from typing import Any, Callable, Optional, Tuple, Union
import multiprocessing as mp
import pickle
import os
import warnings

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from flax import struct
from nicegui import app, ui
from PIL import Image

from currentNiceWebRL.logging import get_logger

# Type definitions
TIMESTEP = Any
RENDER_FN = Callable[[TIMESTEP], jax.Array]

EnvParams = struct.PyTreeNode

logger = get_logger(__name__)

# Global flag to track multiprocessing setup
_mp_initialized = False

def _ensure_multiprocessing_setup():
    """Ensure multiprocessing is properly configured for JAX compatibility."""
    global _mp_initialized
    if _mp_initialized:
        return
    
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            # No method set yet, we can set spawn
            mp.set_start_method("spawn", force=True)
            logger.info("Set multiprocessing start method to 'spawn'")
        elif current_method == "spawn":
            # Already set to spawn, which is what we want
            logger.info("Multiprocessing start method already set to 'spawn'")
        else:
            # Different method already set, warn but continue
            warnings.warn(f"Multiprocessing start method is {current_method}, not 'spawn'. This may cause issues with JAX.")
    except RuntimeError as e:
        # This happens if start method is already set and we can't change it
        warnings.warn(f"Cannot set multiprocessing start method: {e}")
    finally:
        _mp_initialized = True

# Initialize multiprocessing setup
_ensure_multiprocessing_setup()

# ----------------------------
# RNG helpers
# ----------------------------
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
        return obj.tolist()
    elif isinstance(obj, jnp.ndarray):
        obj = jax.tree_map(np.array, obj)
        return obj.tolist()
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
    """Wrapper around env providing reset/step proxies and autoreset behavior."""

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

    def __getattr__(self, name):
        if name == "_env":  # prevent recursion
          raise AttributeError
        return getattr(self._env, name)

    def reset(
        self, key: jax.random.PRNGKey, params: Optional[struct.PyTreeNode] = None
    ) -> Tuple[TimeStep, dict]:
        if self._use_params:
            obs, state = self._env.reset(key, params)
        else:
            obs, state = self._env.reset(key)
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
            if type(done) == dict:
                done = done["__all__"]
            if type(reward) == dict:
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


# ----------------------------
# Multiprocessing worker global and functions (top-level -> pickle-safe)
# ----------------------------
_WORKER_ENV = None  # set per-worker by initializer


def _worker_init_with_obj(env_obj):
    """
    Pool initializer that receives a pickled env object (if the env is picklable).
    The object will be unpickled and assigned to _WORKER_ENV in each worker.
    """
    global _WORKER_ENV
    _WORKER_ENV = env_obj


def _worker_init_with_ctor(env_ctor):
    """
    Pool initializer that receives a picklable callable env_ctor (module-level).
    Calls env_ctor() in each worker to create its own env instance.
    """
    global _WORKER_ENV
    try:
        _WORKER_ENV = env_ctor()
    except Exception as e:
        # If constructing fails, ensure _WORKER_ENV remains None and raise inside worker so that pool creation fails early.
        raise RuntimeError(f"env_ctor raised in worker initializer: {e}")


def _worker_step_one(args):
    """
    Worker function: uses process-global _WORKER_ENV and performs a single step.
    args: (rng, timestep, action, env_params)
    Returns (timestep, info) same as env.step.
    """
    global _WORKER_ENV
    if _WORKER_ENV is None:
        raise RuntimeError("Worker env not initialized (_WORKER_ENV is None).")
    rng, timestep, a, env_params = args
    env = _WORKER_ENV
    try:
        out = env.step(rng, timestep, a, env_params)
    except TypeError:
        out = env.step(rng, timestep, a)
    if isinstance(out, tuple) and len(out) == 2:
        ts, info = out
    else:
        ts, info = out, {}
    return ts, info


# ----------------------------
# JaxWebEnv class
# ----------------------------
class JaxWebEnv:
    def __init__(
        self,
        env,
        actions: Optional[Union[jnp.ndarray, list, tuple]] = None,
        use_multiprocessing: bool = True,
        processes: Optional[int] = None,
        env_ctor: Optional[Callable[[], Any]] = None,
    ):
        """
        env: the environment instance (may or may not be picklable).
        env_ctor: optional module-level callable that returns a fresh env instance (picklable). Recommended when env is not picklable (e.g., JAX/Flax wrappers).
        use_multiprocessing: whether to attempt to use multiprocessing (spawn) for parallel evaluation of actions.
        processes: number of worker processes (default: min(len(actions), cpu_count()))
        """
        self.env = env
        self._unwrapped_env = env  # keep a reference for sequential fallback
        self.use_multiprocessing = use_multiprocessing
        self.processes = processes
        self._pool = None
        self._mp_ctx = None
        self._env_ctor = env_ctor
        self._pool_creation_attempted = False

        assert hasattr(env, "reset"), "env needs reset function"
        assert hasattr(env, "step"), "env needs step function"

        if actions is None:
            actions = try_to_get_actions(env)

        actions_arr = np.asarray(actions)
        try:
            self._action_list = [int(x) for x in actions_arr.ravel().tolist()]
        except Exception:
            self._action_list = [int(x) for x in list(actions_arr)]

        # Don't create pool immediately - delay until first use
        # This avoids the bootstrapping phase issue

        # expose reset directly (keep original API)
        self.reset = env.reset

    # ---------------- pool lifecycle ----------------
    def _maybe_create_pool(self):
        """
        Create a persistent Pool using 'spawn' context (safe with JAX threads).
        Only attempt creation once to avoid repeated warnings.
        """
        if self._pool is not None or self._pool_creation_attempted:
            return

        self._pool_creation_attempted = True

        # Ensure multiprocessing is set up
        _ensure_multiprocessing_setup()

        # enforce 'spawn' start method for safety with JAX multithreading
        try:
            self._mp_ctx = mp.get_context("spawn")
        except Exception as e:
            warnings.warn(f"Failed to get 'spawn' context: {e}. Disabling multiprocessing.")
            self.use_multiprocessing = False
            self._mp_ctx = None
            return

        # choose initializer and initargs
        initfunc = None
        initargs = ()
        if self._env_ctor is not None:
            # make sure env_ctor is picklable (module-level or otherwise picklable)
            try:
                pickle.dumps(self._env_ctor)
                initfunc = _worker_init_with_ctor
                initargs = (self._env_ctor,)
            except Exception as e:
                warnings.warn(f"Provided env_ctor is not picklable: {e}. Disabling multiprocessing.")
                self.use_multiprocessing = False
                self._mp_ctx = None
                return
        else:
            # try to pickle env object directly
            try:
                pickle.dumps(self._unwrapped_env)
                initfunc = _worker_init_with_obj
                initargs = (self._unwrapped_env,)
            except Exception:
                # cannot pickle env; user must provide env_ctor
                warnings.warn(
                    "Environment object is not picklable and no env_ctor provided. "
                    "Disabling multiprocessing; falling back to sequential execution."
                )
                self.use_multiprocessing = False
                self._mp_ctx = None
                return

        # create pool
        try:
            n_proc = self.processes or min(len(self._action_list), (mp.cpu_count() or 1))
            # ensure at least 1 worker (we still map over actions; 1 worker is allowed)
            n_proc = max(1, n_proc)
            self._pool = self._mp_ctx.Pool(processes=n_proc, initializer=initfunc, initargs=initargs)
            logger.info(f"Created spawn Pool with {n_proc} workers for JaxWebEnv.")
        except Exception as e:
            warnings.warn(f"Failed to create Pool (spawn). Error: {e}. Falling back to sequential.")
            self._pool = None
            self.use_multiprocessing = False
            self._mp_ctx = None

    def _close_pool(self):
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                try:
                    self._pool.terminate()
                except Exception:
                    pass
            finally:
                self._pool = None

    def __del__(self):
        # attempt to clean up pool if still alive
        try:
            self._close_pool()
        except Exception:
            pass

    # ---------------- public API ----------------
    def next_steps(self, rng, timestep, env_params):
        """
        Compute next timesteps for each action.
        If multiprocessing (spawn-based) is active: submit lightweight args to workers.
        Otherwise: sequentially iterate actions in-process.
        """
        # ensure pool exists if MP desired
        if self.use_multiprocessing and self._pool is None:
            self._maybe_create_pool()

        if self.use_multiprocessing and self._pool is not None:
            # prepare lightweight args (spawn will pickle these)
            args_iter = [(rng, timestep, a, env_params) for a in self._action_list]
            try:
                results = self._pool.map(_worker_step_one, args_iter)
            except Exception as e:
                # if mapping fails, close pool and fallback to sequential
                warnings.warn(f"Multiprocessing map failed: {e}. Falling back to sequential.")
                self._close_pool()
                self.use_multiprocessing = False
                results = []
                env = self._unwrapped_env
                for a in self._action_list:
                    try:
                        out = env.step(rng, timestep, a, env_params)
                    except TypeError:
                        out = env.step(rng, timestep, a)
                    if isinstance(out, tuple) and len(out) == 2:
                        ts, info = out
                    else:
                        ts, info = out, {}
                    results.append((ts, info))
        else:
            # sequential fallback: call env.step directly
            results = []
            env = self._unwrapped_env
            for a in self._action_list:
                try:
                    out = env.step(rng, timestep, a, env_params)
                except TypeError:
                    out = env.step(rng, timestep, a)
                if isinstance(out, tuple) and len(out) == 2:
                    ts, info = out
                else:
                    ts, info = out, {}
                results.append((ts, info))

        if not results:
            # no actions => return an empty structure similar to previous behavior
            raise RuntimeError("No action results produced in next_steps()")

        timesteps_list, infos_list = zip(*results)

        # Stack the pytree leaves across action axis 0
        def stack_leaves(*leaves):
            arrays = [jnp.asarray(l) for l in leaves]
            return jnp.stack(arrays, axis=0)

        stacked_timesteps = jax.tree_util.tree_map(lambda *l: stack_leaves(*l), *timesteps_list)
        # infos_list is available if you want to aggregate; preserving previous behavior (ignored).
        return stacked_timesteps

    def precompile(self, dummy_env_params: Optional[struct.PyTreeNode] = None) -> None:
        print("Skipping JIT precompilation (removed).")

    def precompile_vmap_render_fn(self, render_fn: RENDER_FN, dummy_env_params: struct.PyTreeNode) -> RENDER_FN:
        return jax.vmap(render_fn)
