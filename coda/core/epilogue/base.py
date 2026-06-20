import cutlass
import cutlass.cute as cute
from typing import NamedTuple
from dataclasses import dataclass

from quack.gemm_sm90 import GemmSm90
from quack.cute_dsl_utils import mlir_namedtuple, ParamsBase


class EpiDecl(object):
    pass


class Epilogue(object):

    def declares(self) -> EpiDecl:
        return EpiDecl()

    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> cute.Tensor | None:
        return None

    def bind(self, name: str, gemm_cls: type) -> type:
        return _lower(self, name=name, gemm_cls=gemm_cls)
