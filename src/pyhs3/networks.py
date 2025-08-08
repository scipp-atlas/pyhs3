"""
Dependency graph utilities for PyHS3.

Provides classes and utilities for building and managing dependency graphs
between parameters, functions, and distributions in HS3 models.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

import rustworkx as rx

from pyhs3.typing.aliases import TensorVar

T = TypeVar("T")


class NamedDiGraph:
    """
    A directed graph with named nodes for easier dependency management.

    Wraps rustworkx.PyDiGraph to provide convenient access to nodes by name
    rather than numeric indices, simplifying dependency graph construction
    and traversal.
    """

    def __init__(self) -> None:
        """Initialize a new NamedDiGraph with an empty graph."""
        self.graph = rx.PyDiGraph()
        self.name_to_index: dict[str, int] = {}

    def add_named_node(self, name: str, node_data: dict[str, Any]) -> int:
        """
        Add a node with a name identifier.

        Args:
            name: Unique name for the node
            node_data: Dictionary of node attributes

        Returns:
            The node index in the underlying graph

        Raises:
            ValueError: If a node with this name already exists
        """
        if name in self.name_to_index:
            msg = f"Node with name '{name}' already exists"
            raise ValueError(msg)

        idx = self.graph.add_node(node_data)
        self.name_to_index[name] = idx
        return idx

    def add_named_edge(
        self, from_name: str, to_name: str, edge_data: Any = None
    ) -> None:
        """
        Add an edge between two named nodes.

        Args:
            from_name: Name of the source node
            to_name: Name of the target node
            edge_data: Optional edge attributes

        Raises:
            KeyError: If either node name doesn't exist
        """
        try:
            from_idx = self.name_to_index[from_name]
            to_idx = self.name_to_index[to_name]
        except KeyError as e:
            msg = f"Node '{e.args[0]}' not found in graph"
            raise KeyError(msg) from e

        self.graph.add_edge(from_idx, to_idx, edge_data)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        """
        Get node data by name or index using dictionary-style access.

        Args:
            key: Name of the node (str) or index (int)

        Returns:
            Node data dictionary

        Raises:
            KeyError: If node name/index doesn't exist
            TypeError: If key is not str or int
        """
        if isinstance(key, str):
            # Access by name
            try:
                idx = self.name_to_index[key]
                return cast(dict[str, Any], self.graph[idx])
            except KeyError:
                msg = f"Node '{key}' not found in graph"
                raise KeyError(msg) from None
        if isinstance(key, int):
            # Access by index
            return cast(dict[str, Any], self.graph[key])

        # This should never happen due to type annotations, but kept for runtime safety
        msg = f"Key must be str (name) or int (index), got {type(key)}"  # type: ignore[unreachable]
        raise TypeError(msg)

    def get(
        self, key: str | int, default: T | None = None
    ) -> dict[str, Any] | T | None:
        """
        Get node data by name or index with optional default.

        Args:
            key: Name of the node (str) or index (int)
            default: Value to return if node doesn't exist

        Returns:
            Node data dictionary or default value

        Raises:
            TypeError: If key is not str or int
        """
        try:
            return self[key]
        except KeyError:
            return default

    def get_node_index(self, name: str) -> int:
        """
        Get the underlying graph index for a named node.

        Args:
            name: Name of the node

        Returns:
            The node index

        Raises:
            KeyError: If node name doesn't exist
        """
        try:
            return self.name_to_index[name]
        except KeyError:
            msg = f"Node '{name}' not found in graph"
            raise KeyError(msg) from None

    def topological_sort(self) -> list[int]:
        """
        Get nodes in topological order.

        Returns:
            List of node indices in topological order

        Raises:
            ValueError: If the graph contains cycles
        """
        try:
            result = rx.topological_sort(self.graph)
            return list(result)
        except rx.DAGHasCycle as e:
            msg = "Circular dependency detected in graph"
            raise ValueError(msg) from e

    def __contains__(self, name: str) -> bool:
        """Check if a named node exists."""
        return name in self.name_to_index

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.name_to_index)

    @property
    def node_names(self) -> list[str]:
        """List of all node names."""
        return list(self.name_to_index.keys())


def build_entity_mappings(
    parameterset: Any, functions: Any, distributions: Any
) -> tuple[dict[str, str], dict[str, TensorVar]]:
    """
    Build mappings of entity names to types and constants.

    Analyzes parameters, functions, and distributions to create lookup
    tables for entity types and any constants generated by distributions.

    Args:
        parameterset: Collection of parameter points
        functions: Collection of function definitions
        distributions: Collection of distribution definitions

    Returns:
        Tuple of (entity_types mapping, constants mapping)
    """
    entity_types: dict[str, str] = {}
    constants_map: dict[str, TensorVar] = {}

    # Map all parameter names
    for param in parameterset:
        entity_types[param.name] = "parameter"

    # Map all function names and their dependencies
    for func in functions:
        for param_name in func.parameters:
            entity_types.setdefault(param_name, "parameter")
        entity_types[func.name] = "function"

    # Map all distribution names, dependencies, and constants
    for dist in distributions:
        for param_name in dist.parameters:
            entity_types.setdefault(param_name, "parameter")
        entity_types[dist.name] = "distribution"

        # Also map any constants generated by this distribution
        for constant_name, constant_tensor in dist.constants.items():
            entity_types[constant_name] = "constant"
            constants_map[constant_name] = constant_tensor

    return entity_types, constants_map


def build_dependency_graph(
    parameterset: Any, functions: Any, distributions: Any
) -> tuple[NamedDiGraph, dict[str, TensorVar]]:
    """
    Build a complete dependency graph for model entities.

    Creates a directed graph representing dependencies between parameters,
    functions, and distributions. Also returns constants mapping for later use.
    Unknown parameter references are automatically created as parameter nodes.

    Args:
        parameterset: Collection of parameter points
        functions: Collection of function definitions
        distributions: Collection of distribution definitions

    Returns:
        Tuple of (dependency graph, constants mapping)

    Raises:
        ValueError: If circular dependencies exist during topological sort
    """
    # Build entity mappings
    entity_types, constants_map = build_entity_mappings(
        parameterset, functions, distributions
    )

    # Create graph and add nodes
    graph = NamedDiGraph()
    for entity_name, entity_type in entity_types.items():
        graph.add_named_node(entity_name, {"type": entity_type, "name": entity_name})

    # Add edges for dependencies
    for entity in [*functions, *distributions]:
        # Get parameter dependencies for this entity
        param_names = entity.parameters

        for param_name in param_names:
            # Add edge: dependency -> entity (param/func/dist feeds into entity)
            graph.add_named_edge(param_name, entity.name, None)

    return graph, constants_map
