# Copyright (c) 2025, Tri Dao.
# Modified version of https://github.com/Dao-AILab/quack/blob/main/quack/cute_dsl_utils.py

import cutlass
import cutlass.cute as cute
from torch.utils import _pytree as pytree
from dataclasses import dataclass, fields
from cutlass.cutlass_dsl import NumericMeta
from cutlass.base_dsl.typing import (
    JitArgument,
    DynamicExpression,
)
from quack.cute_dsl_utils import mlir_namedtuple


StaticTypes = (
    cutlass.Constexpr,
    NumericMeta,
    int,
    bool,
    str,
    float,
    type(None),
)


@dataclass
class DataClassJitArgument(JitArgument):

    def __c_pointers__(self) -> list:
        fields_all = [
            getattr(self, field.name)
            for field in fields(self)
        ]
        fields_dynamic = [
            field
            for field in fields_all
            if not isinstance(field, StaticTypes)
        ]
        c_ptrs = []
        for obj in fields_dynamic:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self) -> list:
        fields_all = [
            getattr(self, field.name)
            for field in fields(self)
        ]
        fields_dynamic = [
            field
            for field in fields_all
            if not isinstance(field, StaticTypes)
        ]
        mlir_types = []
        mlir_nitems = []
        for obj in fields_dynamic:
            if hasattr(obj, "__get_mlir_types__"):
                mlir_type = obj.__get_mlir_types__()
                mlir_types.extend(mlir_type)
                mlir_nitems.append(len(mlir_type))
            else:
                mlir_nitems.append(0)

        self._mlir_nitems = mlir_nitems
        return mlir_types

    def __new_from_mlir_values__(self, values: list) -> object:
        fields_all = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
        fields_static = {
            name: field
            for name, field in fields_all.items()
            if isinstance(field, StaticTypes)
        }
        fields_dynamic = {
            name: field
            for name, field in fields_all.items()
            if not isinstance(field, StaticTypes)
        }
        for (name, field), n_items in zip(
            fields_dynamic.items(),
            self._mlir_nitems,
        ):
            fields_dynamic[name] = cutlass.new_from_mlir_values(
                obj=field,
                values=values[:n_items],
            )
            values = values[n_items:]

        return type(self)(
            **fields_dynamic,
            **fields_static,
        )


@dataclass
class DataClassDynamicExpression(DynamicExpression):

    def __extract_mlir_values__(self) -> list:
        fields_all = [
            getattr(self, field.name)
            for field in fields(self)
        ]
        fields_dynamic = [
            field
            for field in fields_all
            if not isinstance(field, StaticTypes)
        ]
        mlir_values = []
        mlir_nitems = []
        for obj in fields_dynamic:
            mlir_value = cutlass.extract_mlir_values(obj)
            mlir_values.extend(mlir_value)
            mlir_nitems.append(len(mlir_value))
        self._mlir_nitems = mlir_nitems
        return mlir_values

    def __new_from_mlir_values__(self, values: list) -> object:
        fields_all = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
        fields_static = {
            name: field
            for name, field in fields_all.items()
            if isinstance(field, StaticTypes)
        }
        fields_dynamic = {
            name: field
            for name, field in fields_all.items()
            if not isinstance(field, StaticTypes)
        }
        for (name, field), n_items in zip(
            fields_dynamic.items(),
            self._mlir_nitems,
        ):
            fields_dynamic[name] = cutlass.new_from_mlir_values(
                obj=field,
                values=values[:n_items],
            )
            values = values[n_items:]

        return type(self)(
            **fields_dynamic,
            **fields_static,
        )


def register_pytree_dataclass(cls: type[object]) -> type[object]:
    pytree.register_dataclass(cls)
    return cls


def not_static(obj: object) -> bool:
    return not isinstance(obj, StaticTypes)


@register_pytree_dataclass
@dataclass
class PyTreeJitArgument(JitArgument):

    def __c_pointers__(self) -> list:
        leaves, treespec = pytree.tree_flatten(self)
        c_ptrs = []
        for leaf in leaves:
            if not_static(leaf) and hasattr(leaf, "__c_pointers__"):
                c_ptrs.extend(leaf.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self) -> list:
        leaves, treespec = pytree.tree_flatten(self)
        mlir_types = []
        mlir_nitems = []
        for leaf in leaves:
            if not_static(leaf) and hasattr(leaf, "__get_mlir_types__"):
                mlir_type = leaf.__get_mlir_types__()
                mlir_types.extend(mlir_type)
                mlir_nitems.append(len(mlir_type))
            else:
                mlir_nitems.append(0)

        self._treespec = treespec
        self._mlir_nitems = mlir_nitems
        return mlir_types

    def __new_from_mlir_values__(self, values: list) -> object:
        leaves, treespec = pytree.tree_flatten(self)
        assert treespec == self._treespec
        assert len(leaves) == len(self._mlir_nitems)
        assert len(values) == sum(self._mlir_nitems)

        for i, n_items in enumerate(self._mlir_nitems):
            if not_static(leaves[i]):
                leaves[i] = cutlass.new_from_mlir_values(
                    obj=leaves[i],
                    values=values[:n_items],
                )
                values = values[n_items:]
            else:
                assert n_items == 0

        return pytree.tree_unflatten(
            leaves=leaves,
            treespec=treespec,
        )


@register_pytree_dataclass
@dataclass
class PyTreeDynamicExpression(DynamicExpression):

    def __extract_mlir_values__(self) -> list:
        leaves, treespec = pytree.tree_flatten(self)
        mlir_values = []
        mlir_nitems = []
        for leaf in leaves:
            if not_static(leaf):
                mlir_value = cutlass.extract_mlir_values(leaf)
                mlir_values.extend(mlir_value)
                mlir_nitems.append(len(mlir_value))
            else:
                mlir_nitems.append(0)

        self._treespec = treespec
        self._mlir_nitems = mlir_nitems
        return mlir_values

    def __new_from_mlir_values__(self, values: list) -> object:
        leaves, treespec = pytree.tree_flatten(self)
        assert treespec == self._treespec
        assert len(leaves) == len(self._mlir_nitems)
        assert len(values) == sum(self._mlir_nitems)

        for i, n_items in enumerate(self._mlir_nitems):
            if not_static(leaves[i]):
                leaves[i] = cutlass.new_from_mlir_values(
                    obj=leaves[i],
                    values=values[:n_items],
                )
                values = values[n_items:]
            else:
                assert n_items == 0

        return pytree.tree_unflatten(
            leaves=leaves,
            treespec=treespec,
        )
