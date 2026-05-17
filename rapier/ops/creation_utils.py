import cutlass
import cutlass.cute as cute
from rapier.ops.misc_utils import static_assert
from rapier.ops.layout_utils import make_ordered_layout


def allocate_tensor_from_shape(
    shape: tuple | cute.Shape,
    order: tuple | str,
    dtype: type[cute.Numeric],
    memspace: str,
    smem_allocator: cutlass.utils.SmemAllocator,
    **kwargs,
) -> cute.Tensor:
    layout = make_ordered_layout(
        shape=shape,
        order=order,
    )
    return allocate_tensor_from_layout(
        layout=layout,
        dtype=dtype,
        memspace=memspace,
        smem_allocator=smem_allocator,
        **kwargs,
    )


def allocate_tensor_from_layout(
    layout: cute.Layout,
    dtype: type[cute.Numeric],
    memspace: str,
    smem_allocator: cutlass.utils.SmemAllocator,
    **kwargs,
) -> cute.Tensor:
    if cutlass.const_expr(memspace == "rmem"):
        return cute.make_rmem_tensor(
            layout_or_shape=layout,
            dtype=dtype,
        )
    elif cutlass.const_expr(memspace == "smem"):
        return smem_allocator.allocate_tensor(
            element_type=dtype,
            layout=layout,
            **kwargs,
        )
    else:
        raise ValueError(f"`memspace` {memspace} not supported")


def allocate_tensor_from_recast_layout(
    layout: cute.Layout,
    new_type_bits: int,
    old_type_bits: int,
    dtype: type[cute.Numeric],
    memspace: str,
    smem_allocator: cutlass.utils.SmemAllocator,
    **kwargs,
) -> cute.Tensor:
    layout = cute.recast_layout(
        new_type_bits=new_type_bits,
        old_type_bits=old_type_bits,
        src_layout=layout,
    )
    return allocate_tensor_from_layout(
        layout=layout,
        dtype=dtype,
        memspace=memspace,
        smem_allocator=smem_allocator,
        **kwargs
    )


def allocate_tensor_like(
    tensor: cute.Tensor,
    memspace: str,
    smem_allocator: cutlass.utils.SmemAllocator,
    dtype: type[cute.Numeric] | None = None,
    **kwargs,
) -> cute.Tensor:
    if cutlass.const_expr(dtype is None):
        dtype = tensor.element_type
    if cutlass.const_expr(memspace == "rmem"):
        return cute.make_fragment_like(
            tensor,
            dtype=dtype,
        )
    return allocate_tensor_from_layout(
        layout=tensor.layout,
        dtype=dtype,
        memspace=memspace,
        smem_allocator=smem_allocator,
        **kwargs,
    )


def empty_like(
    tensor: cute.Tensor | cute.TensorSSA,
    dtype: type[cute.Numeric] | None = None,
    memspace: str | None = None,
    smem_allocator: cutlass.utils.SmemAllocator | None = None,
    **kwargs,
) -> cute.Tensor | cute.TensorSSA:
    if cutlass.const_expr(isinstance(tensor, cute.Tensor)):
        static_assert(smem_allocator is not None)
        if cutlass.const_expr(memspace is None):
            memspace = tensor.memspace.name
        new_tensor = allocate_tensor_like(
            tensor=tensor,
            memspace=memspace,
            smem_allocator=smem_allocator,
            dtype=dtype,
            **kwargs,
        )

    elif cutlass.const_expr(isinstance(tensor, cute.TensorSSA)):
        static_assert(memspace is None)
        static_assert(smem_allocator is None)
        static_assert(len(kwargs) == 0)
        new_tensor = cute.empty_like(
            tensor,
            dtype=dtype,
        )

    else:
        raise TypeError

    return new_tensor


def zeros_like(
    tensor: cute.Tensor | cute.TensorSSA,
    dtype: type[cute.Numeric] | None = None,
    memspace: str | None = None,
    smem_allocator: cutlass.utils.SmemAllocator | None = None,
    **kwargs,
) -> cute.Tensor | cute.TensorSSA:
    if cutlass.const_expr(isinstance(tensor, cute.Tensor)):
        static_assert(smem_allocator is not None)
        if cutlass.const_expr(memspace is None):
            memspace = tensor.memspace.name
        new_tensor = allocate_tensor_like(
            tensor=tensor,
            memspace=memspace,
            smem_allocator=smem_allocator,
            dtype=dtype,
            **kwargs,
        )
        new_tensor.fill(value=0)

    elif cutlass.const_expr(isinstance(tensor, cute.TensorSSA)):
        static_assert(memspace is None)
        static_assert(smem_allocator is None)
        static_assert(len(kwargs) == 0)
        new_tensor = cute.zeros_like(
            tensor,
            dtype=dtype,
        )

    else:
        raise TypeError

    return new_tensor


def ones_like(
    tensor: cute.Tensor | cute.TensorSSA,
    dtype: type[cute.Numeric] | None = None,
    memspace: str | None = None,
    smem_allocator: cutlass.utils.SmemAllocator | None = None,
    **kwargs,
) -> cute.Tensor | cute.TensorSSA:
    if cutlass.const_expr(isinstance(tensor, cute.Tensor)):
        static_assert(smem_allocator is not None)
        if cutlass.const_expr(memspace is None):
            memspace = tensor.memspace.name
        new_tensor = allocate_tensor_like(
            tensor=tensor,
            memspace=memspace,
            smem_allocator=smem_allocator,
            dtype=dtype,
            **kwargs,
        )
        new_tensor.fill(value=1)

    elif cutlass.const_expr(isinstance(tensor, cute.TensorSSA)):
        static_assert(memspace is None)
        static_assert(smem_allocator is None)
        static_assert(len(kwargs) == 0)
        new_tensor = cute.ones_like(
            tensor,
            dtype=dtype,
        )

    else:
        raise TypeError

    return new_tensor


def full_like(
    tensor: cute.Tensor | cute.TensorSSA,
    fill_value: int | float,
    dtype: type[cute.Numeric] | None = None,
    memspace: str | None = None,
    smem_allocator: cutlass.utils.SmemAllocator | None = None,
    **kwargs,
) -> cute.Tensor | cute.TensorSSA:
    if cutlass.const_expr(isinstance(tensor, cute.Tensor)):
        static_assert(smem_allocator is not None)
        if cutlass.const_expr(memspace is None):
            memspace = tensor.memspace.name
        new_tensor = allocate_tensor_like(
            tensor=tensor,
            memspace=memspace,
            smem_allocator=smem_allocator,
            dtype=dtype,
            **kwargs,
        )
        new_tensor.fill(value=fill_value)

    elif cutlass.const_expr(isinstance(tensor, cute.TensorSSA)):
        static_assert(memspace is None)
        static_assert(smem_allocator is None)
        static_assert(len(kwargs) == 0)
        new_tensor = cute.full_like(
            tensor,
            fill_value=fill_value,
            dtype=dtype,
        )

    else:
        raise TypeError

    return new_tensor
