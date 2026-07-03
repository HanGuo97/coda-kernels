from quack.epi_ops import EpiOp, assume_stride_divisibility


class ColVecStore(EpiOp):

    dim = 0
    epi_m_major_preference = -1

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def epi_m_major_score(self, arg_tensor, gemm):
        return self.epi_m_major_preference

    def _tile_size(self, cta_tile_shape_mnk):
        return cta_tile_shape_mnk[self.dim]

    def _broadcast_stride(self):
        return (1, 0) if self.dim == 0 else (0, 1)

    def _reduce_dim(self):
        return 1 - self.dim
