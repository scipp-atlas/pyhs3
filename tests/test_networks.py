"""
Unit tests for the networks module.

Tests for NamedDiGraph class and dependency graph utilities.
"""

from __future__ import annotations

import pytest

from pyhs3.distributions import GaussianDist
from pyhs3.functions import GenericFunction, ProductFunction
from pyhs3.networks import NamedDiGraph, build_dependency_graph, build_entity_mappings
from pyhs3.parameter_points import ParameterPoint, ParameterSet


class TestNamedDiGraph:
    """Test NamedDiGraph implementation."""

    def test_named_digraph_creation(self):
        """Test NamedDiGraph can be created."""
        graph = NamedDiGraph()
        assert len(graph) == 0
        assert graph.node_names == []

    def test_add_named_node(self):
        """Test adding named nodes."""
        graph = NamedDiGraph()

        # Add first node
        idx1 = graph.add_named_node("node1", {"type": "parameter", "value": 1.0})
        assert idx1 == 0
        assert len(graph) == 1
        assert "node1" in graph
        assert graph.node_names == ["node1"]

        # Add second node
        idx2 = graph.add_named_node("node2", {"type": "function", "value": 2.0})
        assert idx2 == 1
        assert len(graph) == 2
        assert "node2" in graph
        assert set(graph.node_names) == {"node1", "node2"}

    def test_add_duplicate_named_node_raises_error(self):
        """Test that adding duplicate node names raises ValueError."""
        graph = NamedDiGraph()
        graph.add_named_node("node1", {"type": "parameter"})

        with pytest.raises(ValueError, match="Node with name 'node1' already exists"):
            graph.add_named_node("node1", {"type": "function"})

    def test_add_named_edge(self):
        """Test adding edges between named nodes."""
        graph = NamedDiGraph()
        graph.add_named_node("node1", {"type": "parameter"})
        graph.add_named_node("node2", {"type": "function"})

        # Add edge with no data
        graph.add_named_edge("node1", "node2")

        # Add edge with data
        graph.add_named_edge("node1", "node2", {"weight": 1.5})

    def test_add_named_edge_nonexistent_node_raises_error(self):
        """Test that adding edge to nonexistent node raises KeyError."""
        graph = NamedDiGraph()
        graph.add_named_node("node1", {"type": "parameter"})

        with pytest.raises(KeyError, match="Node 'nonexistent' not found in graph"):
            graph.add_named_edge("node1", "nonexistent")

        with pytest.raises(KeyError, match="Node 'nonexistent' not found in graph"):
            graph.add_named_edge("nonexistent", "node1")

    def test_getitem_by_name(self):
        """Test accessing node data by name."""
        graph = NamedDiGraph()
        node_data = {"type": "parameter", "value": 42}
        graph.add_named_node("test_node", node_data)

        retrieved_data = graph["test_node"]
        assert retrieved_data == node_data

    def test_getitem_by_index(self):
        """Test accessing node data by index."""
        graph = NamedDiGraph()
        node_data = {"type": "parameter", "value": 42}
        idx = graph.add_named_node("test_node", node_data)

        retrieved_data = graph[idx]
        assert retrieved_data == node_data

    def test_getitem_nonexistent_name_raises_error(self):
        """Test that accessing nonexistent node by name raises KeyError."""
        graph = NamedDiGraph()

        with pytest.raises(KeyError, match="Node 'nonexistent' not found in graph"):
            _ = graph["nonexistent"]

    def test_getitem_invalid_type_raises_error(self):
        """Test that accessing with invalid key type raises TypeError."""
        graph = NamedDiGraph()

        with pytest.raises(
            TypeError, match="Key must be str \\(name\\) or int \\(index\\)"
        ):
            _ = graph[3.14]  # float is not allowed

    def test_get_with_default(self):
        """Test get method with default values."""
        graph = NamedDiGraph()
        node_data = {"type": "parameter", "value": 42}
        idx = graph.add_named_node("test_node", node_data)

        # Test successful get by name
        assert graph.get("test_node") == node_data

        # Test successful get by index
        assert graph.get(idx) == node_data

        # Test get with default for nonexistent name
        assert graph.get("nonexistent", "default") == "default"

        # Test get with None default for nonexistent name
        assert graph.get("nonexistent") is None

    def test_get_node_index(self):
        """Test getting node index by name."""
        graph = NamedDiGraph()
        idx = graph.add_named_node("test_node", {"type": "parameter"})

        assert graph.get_node_index("test_node") == idx

        with pytest.raises(KeyError, match="Node 'nonexistent' not found in graph"):
            graph.get_node_index("nonexistent")

    def test_topological_sort_simple(self):
        """Test topological sort on simple graph."""
        graph = NamedDiGraph()
        graph.add_named_node("A", {"type": "parameter"})
        graph.add_named_node("B", {"type": "function"})
        graph.add_named_node("C", {"type": "distribution"})

        # A -> B -> C
        graph.add_named_edge("A", "B")
        graph.add_named_edge("B", "C")

        sorted_nodes = graph.topological_sort()
        assert len(sorted_nodes) == 3

        # Verify ordering: A should come before B, B should come before C
        a_idx = graph.get_node_index("A")
        b_idx = graph.get_node_index("B")
        c_idx = graph.get_node_index("C")

        a_pos = sorted_nodes.index(a_idx)
        b_pos = sorted_nodes.index(b_idx)
        c_pos = sorted_nodes.index(c_idx)

        assert a_pos < b_pos < c_pos

    def test_topological_sort_cycle_raises_error(self):
        """Test that topological sort raises error on cyclic graph."""
        graph = NamedDiGraph()
        graph.add_named_node("A", {"type": "function"})
        graph.add_named_node("B", {"type": "function"})

        # Create cycle: A -> B -> A
        graph.add_named_edge("A", "B")
        graph.add_named_edge("B", "A")

        with pytest.raises(ValueError, match="Circular dependency detected in graph"):
            graph.topological_sort()

    def test_contains(self):
        """Test __contains__ method."""
        graph = NamedDiGraph()
        graph.add_named_node("test_node", {"type": "parameter"})

        assert "test_node" in graph
        assert "nonexistent" not in graph

    def test_len(self):
        """Test __len__ method."""
        graph = NamedDiGraph()
        assert len(graph) == 0

        graph.add_named_node("node1", {"type": "parameter"})
        assert len(graph) == 1

        graph.add_named_node("node2", {"type": "function"})
        assert len(graph) == 2

    def test_node_names_property(self):
        """Test node_names property."""
        graph = NamedDiGraph()
        assert graph.node_names == []

        graph.add_named_node("node1", {"type": "parameter"})
        graph.add_named_node("node2", {"type": "function"})

        names = graph.node_names
        assert set(names) == {"node1", "node2"}
        assert len(names) == 2


class TestBuildEntityMappings:
    """Test build_entity_mappings function."""

    def test_build_entity_mappings_basic(self):
        """Test basic entity mapping building."""
        # Create test data
        param1 = ParameterPoint(name="mu", value=0.0)
        parameterset = ParameterSet(name="test_params", parameters=[param1])

        func = ProductFunction(name="test_func", factors=["mu"])
        functions = [func]

        # Use numeric sigma to create a constant
        dist = GaussianDist(name="test_dist", mean="mu", sigma=1.0, x="mu")
        distributions = [dist]

        # Build mappings
        entity_types, constants_map = build_entity_mappings(
            parameterset, functions, distributions
        )

        # Check entity types
        assert entity_types["mu"] == "parameter"
        assert entity_types["test_func"] == "function"
        assert entity_types["test_dist"] == "distribution"

        # Should have sigma constant from distribution (numeric sigma creates constant)
        assert "constant_test_dist_sigma" in entity_types
        assert entity_types["constant_test_dist_sigma"] == "constant"

        # Also check x parameter got mapped
        assert entity_types["mu"] == "parameter"  # x parameter is also mu

        # Check constants map
        assert "constant_test_dist_sigma" in constants_map
        assert constants_map["constant_test_dist_sigma"] is not None

    def test_build_entity_mappings_with_numeric_parameters(self):
        """Test entity mapping with numeric parameters in distributions."""
        parameterset = ParameterSet(name="test_params", parameters=[])
        functions = []

        # Distribution with numeric sigma
        dist = GaussianDist(name="test_dist", mean="mu", sigma=2.0, x="obs")
        distributions = [dist]

        entity_types, constants_map = build_entity_mappings(
            parameterset, functions, distributions
        )

        # Should map parameter dependencies
        assert entity_types["mu"] == "parameter"
        assert entity_types["obs"] == "parameter"
        assert entity_types["test_dist"] == "distribution"

        # Should create constant for numeric sigma
        assert "constant_test_dist_sigma" in entity_types
        assert entity_types["constant_test_dist_sigma"] == "constant"
        assert "constant_test_dist_sigma" in constants_map

    def test_build_entity_mappings_empty_collections(self):
        """Test entity mapping with empty collections."""
        parameterset = ParameterSet(name="empty", parameters=[])
        functions = []
        distributions = []

        entity_types, constants_map = build_entity_mappings(
            parameterset, functions, distributions
        )

        assert entity_types == {}
        assert constants_map == {}


class TestBuildDependencyGraph:
    """Test build_dependency_graph function."""

    def test_build_dependency_graph_basic(self):
        """Test basic dependency graph building."""
        # Create test data
        param = ParameterPoint(name="mu", value=0.0)
        parameterset = ParameterSet(name="test_params", parameters=[param])

        func = ProductFunction(name="test_func", factors=["mu"])
        functions = [func]

        dist = GaussianDist(name="test_dist", mean="test_func", sigma=1.0, x="mu")
        distributions = [dist]

        # Build graph
        graph, constants_map = build_dependency_graph(
            parameterset, functions, distributions
        )

        # Check nodes exist
        assert "mu" in graph
        assert "test_func" in graph
        assert "test_dist" in graph
        assert "constant_test_dist_sigma" in graph

        # Check node types
        assert graph["mu"]["type"] == "parameter"
        assert graph["test_func"]["type"] == "function"
        assert graph["test_dist"]["type"] == "distribution"
        assert graph["constant_test_dist_sigma"]["type"] == "constant"

        # Check constants map
        assert "constant_test_dist_sigma" in constants_map

        # Test topological sort works (no cycles)
        sorted_nodes = graph.topological_sort()
        assert len(sorted_nodes) == 4

    def test_build_dependency_graph_with_unknown_entity_creates_parameter(self):
        """Test that unknown entity references auto-create parameter nodes."""
        parameterset = ParameterSet(name="test_params", parameters=[])

        # Function referencing unknown parameter that's not in parameterset or other functions
        func = GenericFunction(name="test_func", expression="unknown_param * 2")
        functions = [func]

        distributions = []

        # Should succeed - unknown entities are auto-created as parameters
        graph, _constants_map = build_dependency_graph(
            parameterset, functions, distributions
        )

        # Verify the unknown parameter was auto-created
        assert "unknown_param" in graph
        assert graph["unknown_param"]["type"] == "parameter"
        assert "test_func" in graph
        assert graph["test_func"]["type"] == "function"

    def test_build_dependency_graph_circular_dependency(self):
        """Test that circular dependencies are detected during topological sort."""
        parameterset = ParameterSet(name="test_params", parameters=[])

        # Create circular dependency: func1 -> func2 -> func1
        func1 = GenericFunction(name="func1", expression="func2 + 1")
        func2 = GenericFunction(name="func2", expression="func1 * 2")
        functions = [func1, func2]

        distributions = []

        # build_dependency_graph should succeed (just builds the graph)
        graph, _constants_map = build_dependency_graph(
            parameterset, functions, distributions
        )

        # But topological sort should detect the cycle
        with pytest.raises(ValueError, match="Circular dependency detected in graph"):
            graph.topological_sort()

    def test_build_dependency_graph_complex_dependencies(self):
        """Test complex dependency graph with multiple levels."""
        # Create multi-level dependencies: param -> func1 -> func2 -> dist
        param = ParameterPoint(name="base_param", value=1.0)
        parameterset = ParameterSet(name="test_params", parameters=[param])

        func1 = GenericFunction(name="func1", expression="base_param * 2")
        func2 = GenericFunction(name="func2", expression="func1 + 1")
        functions = [func1, func2]

        dist = GaussianDist(name="test_dist", mean="func2", sigma=1.0, x="base_param")
        distributions = [dist]

        graph, _constants_map = build_dependency_graph(
            parameterset, functions, distributions
        )

        # Verify all nodes exist
        expected_nodes = {
            "base_param",
            "func1",
            "func2",
            "test_dist",
            "constant_test_dist_sigma",
        }
        actual_nodes = set(graph.node_names)
        assert expected_nodes == actual_nodes

        # Verify topological ordering respects dependencies
        sorted_nodes = graph.topological_sort()

        # base_param should come before func1
        base_pos = next(
            i
            for i, idx in enumerate(sorted_nodes)
            if graph[idx]["name"] == "base_param"
        )
        func1_pos = next(
            i for i, idx in enumerate(sorted_nodes) if graph[idx]["name"] == "func1"
        )
        func2_pos = next(
            i for i, idx in enumerate(sorted_nodes) if graph[idx]["name"] == "func2"
        )
        dist_pos = next(
            i for i, idx in enumerate(sorted_nodes) if graph[idx]["name"] == "test_dist"
        )

        assert base_pos < func1_pos < func2_pos < dist_pos
