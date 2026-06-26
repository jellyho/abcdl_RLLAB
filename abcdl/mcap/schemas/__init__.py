"""Load the ABC custom protobuf messages (RobotState/GripperState/Instructions/Annotation).

Built at import time from abc.proto via grpc_tools.protoc into a temp module,
falling back to a prebuilt abc_pb2 if present.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

_HERE = os.path.dirname(__file__)


def _build():
    try:
        return importlib.import_module("abcdl.mcap.schemas.abc_pb2")
    except ModuleNotFoundError:
        pass
    from grpc_tools import protoc  # provided by grpcio-tools

    out = tempfile.mkdtemp(prefix="abcdl_pb2_")
    rc = protoc.main([
        "protoc", f"-I{_HERE}",
        f"-I{os.path.dirname(protoc.__file__)}/_proto",  # bundled google/protobuf/*.proto
        f"--python_out={out}", os.path.join(_HERE, "abc.proto"),
    ])
    if rc != 0:
        raise RuntimeError("protoc failed to compile abc.proto")
    sys.path.insert(0, out)
    return importlib.import_module("abc_pb2")


_pb2 = _build()
RobotState = _pb2.RobotState
GripperState = _pb2.GripperState
Instructions = _pb2.Instructions
Annotation = _pb2.Annotation
