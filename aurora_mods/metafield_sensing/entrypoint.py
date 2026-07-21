#!/usr/bin/env python3
"""
metafield_sensing / entrypoint.py

Initial skeleton for a MetaField sensing mod in Aurora.

Purpose:
- Periodically collect stats from a running MetaField instance
  (memory size, prediction loss, curvature signals, etc.)
- Expose them through Aurora's sensing / monitoring system

This is a minimal starting point. It will evolve to support:
- richer telemetry
- task scheduling of MetaField workloads
- curvature-aware compute allocation
"""

# Example of how Aurora mods typically register hooks.
# The exact API will depend on Aurora's hook_registry, but the pattern is usually:
#
# from scheduler.hook_registry import register_hook
#
# def on_sensing_tick(context):
#     # Pull stats from MetaField (via shared memory, API, or file)
#     stats = get_metafield_stats()
#     context.sensing.publish("metafield", stats)
#
# def register():
#     register_hook("on_sensing_tick", on_sensing_tick)
#     register_hook("on_node_status", on_node_status)
#
# if __name__ == "__main__":
#     register()


def get_metafield_stats():
    """
    Placeholder function.
    In a real implementation this would read from a running MetaField
    process (via shared memory, Redis, file, or direct import if co-located).
    """
    return {
        "memory_size": 0,
        "avg_priority": 0.0,
        "prediction_loss": None,
        "geometry_loss": None,
    }


def on_sensing_tick(context):
    """Called periodically by Aurora's sensing system."""
    stats = get_metafield_stats()
    # TODO: Publish stats into Aurora's sensing layer
    print(f"[metafield_sensing] Tick - stats: {stats}")


def on_node_status(context):
    """Optional hook for node-level status updates."""
    pass


def register():
    """
    Register this mod's hooks with Aurora.
    This function would normally be called by Aurora's mod loader.
    """
    # Example (pseudo-code):
    # from scheduler.hook_registry import register_hook
    # register_hook("on_sensing_tick", on_sensing_tick)
    # register_hook("on_node_status", on_node_status)
    print("[metafield_sensing] Mod registered (skeleton)")


if __name__ == "__main__":
    register()
