import cutlass
import cutlass.cute as cute

from quack.gemm_sm90 import GemmSm90
from quack.cute_dsl_utils import mlir_namedtuple, ParamsBase


class Epilogue(object):

    def declares(self) -> tuple[EpiOp, ...]:
        return ()

    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        return ()

    def cache_key(self) -> tuple:
        return (type(self).__name__,)

    def bind(self, name: str, gemm_cls: type) -> type:
        return _lower(self, name=name, gemm_cls=gemm_cls)


class _Composite(Epilogue):

    def __init__(self, epilogues: Iterable["Epilogue"]) -> None:
        self._children = list(epilogues)

    def declares(self) -> tuple[EpiOp, ...]:
        return _normalize(child.declares() for child in self._children)

    def cache_key(self) -> tuple:
        return tuple(child.cache_key() for child in self._children)

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        tRS_rAuxOuts = []
        for child in self._children:
            tRS_rAuxOuts.extend(
                child.visit(
                    gemm=gemm,
                    params=params,
                    epi_loop_tensors=epi_loop_tensors,
                    tRS_rD=tRS_rD,
                    tRS_rC=tRS_rC,
                )
            )
        if cutlass.const_expr(len(tRS_rAuxOuts) > 1):
            raise NotImplementedError
        return tuple(tRS_rAuxOuts)


def compose(epilogues: Iterable["Epilogue"]) -> "Epilogue":
    return _Composite(list(epilogues))
