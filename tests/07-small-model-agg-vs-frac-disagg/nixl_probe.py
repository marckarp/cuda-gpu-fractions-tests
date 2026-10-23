import nixl, inspect
from nixl._api import nixl_agent, nixl_agent_config
print("nixl", getattr(nixl, "__version__", "?"))
a = nixl_agent("probe", nixl_agent_config(backends=["UCX"]))
for n in ["register_memory","get_reg_descs","get_xfer_descs","initialize_xfer","transfer","check_xfer_state",
          "get_agent_metadata","add_remote_agent","get_serialized_descs","deserialize_descs","prep_xfer_dlist",
          "make_prepped_xfer","release_xfer_handle","deregister_memory","get_xfer_telemetry"]:
    print(f"{n}: {hasattr(a, n)}")
for n in ["initialize_xfer","get_xfer_descs","register_memory","transfer","check_xfer_state","prep_xfer_dlist","make_prepped_xfer"]:
    if hasattr(a, n): print(n, inspect.signature(getattr(a, n)))
