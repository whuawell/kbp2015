from pattern.graph import Graph, Node, Edge

import json


class NoPathException(Exception):
    pass


class DependencyParse(object):

    def __init__(self, parse, enhanced=True):
        self.parse = parse
        self.g, self.edge, self.node, self.root = DependencyParse.make_graph(self.parse, enhanced)

    @classmethod
    def make_graph(cls, parse, enhanced=True):
        edge_map, node_map = {}, {}
        g = Graph()
        root = None
        for child, parent, arc in parse:
            if arc == 'root':
                root = child
            if not enhanced:
                arc = arc.split(':')[0]
            if child not in node_map:
                node_map[child] = Node(child)
            child = node_map[child]
            if parent not in node_map:
                node_map[parent] = Node(parent)
            parent = node_map[parent]
            if parent.id != child.id:
                g.add_edge(parent, child, type=arc)
        return g, edge_map, node_map, root

    @classmethod
    def parent_of(cls, node):
        parents = [e.node1 for e in node.edges if e.node2 == node]
        return parents[0] if len(parents) else None

    @classmethod
    def get_head(cls, ent_tail, ent_start, ent_end):
        seen = set()
        while True:
            parent = cls.parent_of(ent_tail)
            if parent in seen:
                raise Exception("found cycle!")
            if parent is None or parent.id >= ent_end or parent.id < ent_start:
                break
            seen.add(parent)
            ent_tail = parent
        return ent_tail

    @classmethod
    def get_edge(cls, node1, node2):
        edges = []
        for edge in node1.edges:
            if edge.node1 == node2:
                edges.append(edge.type + '_from')
            elif edge.node2 == node2:
                edges.append(edge.type + '_to')
        return edges

    def get_path(self, node1, node2, g):
        path = g.shortest_path(node1, node2, directed=False)
        if path is None:
            raise NoPathException("cannot find path between entities!")
        curr = node1
        edges = []
        for node in path[1:]:
            if curr.id == self.root:
                edges.append([curr.id, None, 'root'])
            edge = self.get_edge(curr, node)[0]
            edges.append([curr.id, node.id, edge])
            curr = node
        return edges

    def get_path_from_parse(self, subject_start, subject_end, object_start, object_end):
        subject = self.node[subject_end-1], subject_start, subject_end
        object = self.node[object_end-1], object_start, object_end
        return self.get_path(
            DependencyParse.get_head(*object),
            DependencyParse.get_head(*subject),
            self.g
        )
