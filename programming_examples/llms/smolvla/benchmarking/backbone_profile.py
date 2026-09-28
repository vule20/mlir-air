"""Op-level profile of the SmolLM2-360M backbone's single fill forward
(past_key_values=None), the same method CPU_STAGES_REPROFILE.md used for the
action expert. Investigation only, not part of the shipping example.

    PATH must include .venv312_main/lib/python3.12/site-packages/mlir_aie/bin
    PEANO_INSTALL_DIR=~/peano_pinned/llvm-aie
    .venv312_main/bin/python benchmarking/backbone_profile.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SMOLVLA_CPU_BIND", "1")
os.environ.setdefault("SMOLVLA_CPU_THREADS", "8")
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from smolvla_inference import build_oracle_batch, fixed_noise, run_hybrid_forward  # noqa: E402


def main():
    torch.set_num_threads(int(os.environ.get("SMOLVLA_CPU_THREADS", "8")))

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()
    batch = build_oracle_batch(policy, n_cameras=3)
    noise = fixed_noise(policy)

    vwe = policy.model.vlm_with_expert
    orig_fwd = vwe.forward
    calls = {"backbone": 0, "expert": 0}
    wall = {"backbone": 0.0, "expert": 0.0}

    def timed_fwd(*a, **kw):
        key = "backbone" if kw.get("past_key_values") is None else "expert"
        t = time.perf_counter()
        r = orig_fwd(*a, **kw)
        wall[key] += (time.perf_counter() - t) * 1e3
        calls[key] += 1
        return r

    vwe.forward = timed_fwd

    # Warm up (compiles nothing on CPU, but touches every code path / cache).
    run_hybrid_forward(batch, policy=policy, noise=noise, npu_vision=False)
    calls["backbone"] = calls["expert"] = 0
    wall["backbone"] = wall["expert"] = 0.0

    # Isolate ONE backbone fill call: start/stop the profiler around just that
    # call (not the context-managed whole predict_action_chunk), so the CPU
    # vision encoder's own ops (which run before vwe.forward is ever called)
    # never enter the trace. Raise right after to skip the 10 expert steps too
    # -- nothing past the backbone call is wanted.
    class _StopAfterBackbone(Exception):
        pass

    prof = profile(activities=[ProfilerActivity.CPU], record_shapes=True)

    def guarded_fwd(*a, **kw):
        if kw.get("past_key_values") is not None:
            raise _StopAfterBackbone()
        prof.start()
        r = timed_fwd(*a, **kw)
        prof.stop()
        raise _StopAfterBackbone()

    vwe.forward = guarded_fwd
    try:
        run_hybrid_forward(batch, policy=policy, noise=noise, npu_vision=False)
    except _StopAfterBackbone:
        pass
    vwe.forward = orig_fwd

    print(f"\nBackbone fill: {wall['backbone']:.3f} ms wall ({calls['backbone']} call)\n")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=30
        )
    )
    print()
    print(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    )

    # A few more clean repeats for a stable median wall time.
    vwe.forward = timed_fwd
    reps = []
    for _ in range(10):
        calls["backbone"] = 0
        wall["backbone"] = 0.0
        run_hybrid_forward(batch, policy=policy, noise=noise, npu_vision=False)
        reps.append(wall["backbone"])
    vwe.forward = orig_fwd
    reps.sort()
    print(f"\nBackbone fill wall time, 10 reps (ms): {reps}")
    print(f"median: {reps[len(reps) // 2]:.3f} ms")


if __name__ == "__main__":
    main()
