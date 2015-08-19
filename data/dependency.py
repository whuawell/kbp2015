import json


class NoPathException(Exception):
    pass


class TreeNode(object):

    def __init__(self, word):
        self.word = int(word)
        self.parent = None
        self.edge_to_parent = None

    def __repr__(self):
        return str(self.word)

    def add_child(self, child, edge):
        child.parent = self
        child.edge_to_parent = edge

    def shortest_path(self, another):
        seen = set()
        my_path = []
        curr = self
        while curr:
            if curr in seen:
                raise NoPathException('found cycle!')
            my_path += [curr]
            seen.add(curr)
            curr = curr.parent
        another_path = []
        curr = another
        common_ancestor = None
        while curr:
            if curr in seen:
                common_ancestor = curr
                break
            another_path += [curr]
            curr = curr.parent
        if common_ancestor is None:
            raise NoPathException('no path between nodes')
        path = []
        for node in my_path:
            if node == common_ancestor:
                break
            path += [(node.word, node.parent.word, node.edge_to_parent + '_from')]
        for node in reversed(another_path):
            path += [(node.parent.word, node.word, node.edge_to_parent + '_to')]
        return path


class DependencyParse(object):

    def __init__(self, parse, enhanced=True):
        self.parse = parse
        self.node_map = DependencyParse.make_graph(self.parse, enhanced)

    @classmethod
    def make_graph(cls, parse, enhanced=True):
        node_map = {}
        for child, parent, arc in parse:
            if not enhanced:
                arc = arc.split(':')[0]
            if child not in node_map:
                node_map[child] = TreeNode(child)
            if parent not in node_map:
                node_map[parent] = TreeNode(parent)
            node_map[parent].add_child(node_map[child], arc)
        return node_map

    def get_head(self, ent_tail, ent_start, ent_end):
        seen = set()
        curr = self.node_map[ent_tail]
        while True:
            parent = curr.parent
            if parent in seen:
                raise NoPathException("found cycle!")
            if parent is None or parent.word >= ent_end or parent.word < ent_start:
                break
            seen.add(parent)
            ent_tail = parent
        return self.node_map[ent_tail]

    def get_path_from_parse(self, subject_start, subject_end, object_start, object_end):
        subject = subject_end-1, subject_start, subject_end
        object = object_end-1, object_start, object_end
        return self.get_head(*object).shortest_path(self.get_head(*subject))
