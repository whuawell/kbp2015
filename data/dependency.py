from pattern.graph import Graph, Node, Edge

import json

def parse_words(words):
    parsed = words[2:-2].split('","')
    isdigit = unicode.isdigit if isinstance(parsed[0], unicode) else str.isdigit
    parsed = ['0'*len(word) if isdigit(word) else word for word in parsed]
    return parsed

def make_graph(parse):
    edge_map, node_map = {}, {}
    g = Graph()
    for arc in parse.split("\n"):
        child, parent, arc_type = arc.split("\t")
        arc_type = arc_type.split(':')[0]
        child, parent = [int(k)-1 for k in [child, parent]]
        # print [words[k] for k in [parent, child]], arc_type
        if child not in node_map:
            node_map[child] = Node(child)
        child = node_map[child]
        if parent not in node_map:
            node_map[parent] = Node(parent)
        parent = node_map[parent]
        if parent.id != child.id:
            g.add_edge(parent, child, type=arc_type)
    return g, edge_map, node_map

def get_parent(node):
    parents = [e.node1 for e in node.edges if e.node2 == node]
    return parents[0] if len(parents) else None

def get_head(ent_tail, ent_start, ent_end):
    seen = set()
    while True:
        parent = get_parent(ent_tail)
        if parent in seen:
            raise Exception("found cycle!")
        if parent is None or parent.id >= ent_end or parent.id < ent_start:
            break
        seen.add(parent)
        ent_tail = parent
    return ent_tail

def get_edge(node1, node2):
    edges = []
    for edge in node1.edges:
        if edge.node1 == node2:
            edges.append(edge.type + '1')
        elif edge.node2 == node2:
            edges.append(edge.type + '2')
    return edges

class NoPathException(Exception):
    pass

def get_path(node1, node2, g):
    path = g.shortest_path(node1, node2)
    if path is None:
        raise NoPathException("cannot find path between entities!")
    curr = node1
    edges = []
    for node in path[1:]:
        edge = get_edge(curr, node)[0]
        edges.append([curr.id, node.id, edge])
        curr = node
    return edges

def get_path_from_parse(parse, subject_start, subject_end, object_start, object_end):
    g, edge_map, node_map = make_graph(parse)
    subject = node_map[subject_end-1], subject_start, subject_end
    object = node_map[object_end-1], object_start, object_end
    return get_path(get_head(*object), get_head(*subject), g)

if __name__ == '__main__':
    d = {u'object_begin': u'22', u'object_head': u'22', u'subject_begin': u'34', u'pos': u'{"DT","JJ","NN","VBD","JJ","IN","JJ","NN","NNS","IN","CD",",","CC","NN","IN","DT","NNS","CC","IN","JJ","NNPS","IN","NNP","VBD","JJ","TO","NNP","NNP","NNP","JJ","NN","IN","JJ","JJ","NNP","NNP","IN","DT","JJ","NNP","NNP","NN","RB","."}', u'object_ner': u'STATE_OR_PROVINCE', u'lemma': u'{"the","indian","vote","be","important","in","several","state","race","in","2006",",","and","turnout","on","the","reservation","and","among","urban","Indians","in","Montana","be","crucial","to","Democrat","Jon","Tester\'s","close","victory","over","incumbent","republican","Conrad","Burns","in","the","recent","U.S.","Senate","election","here","."}', u'relation': u'per:stateorprovinces_of_residence', u'subject_ner': u'PERSON', u'words': u'{"The","Indian","vote","was","important","in","several","state","races","in","2006",",","and","turnout","on","the","reservations","and","among","urban","Indians","in","Montana","was","crucial","to","Democrat","Jon","Tester\'s","close","victory","over","incumbent","Republican","Conrad","Burns","in","the","recent","U.S.","Senate","election","here","."}', u'object_end': u'23', u'dependency': u'1\t3\tdet\n2\t3\tamod\n3\t5\tnsubj\n4\t5\tcop\n5\t0\troot\n6\t9\tcase\n7\t9\tamod\n8\t9\tcompound\n9\t5\tnmod:in\n10\t11\tcase\n11\t9\tnmod:in\n12\t5\tpunct\n13\t5\tcc\n14\t14\tconj:and\n15\t17\tcase\n16\t17\tdet\n17\t14\tnmod:on\n18\t14\tcc\n19\t21\tcase\n20\t21\tamod\n21\t14\tnmod:among\n22\t23\tcase\n23\t21\tnmod:in\n24\t25\tcop\n25\t5\tconj:and\n26\t29\tcase\n27\t29\tcompound\n28\t29\tcompound\n29\t25\tnmod:to\n30\t31\tamod\n31\t29\tdep\n32\t36\tcase\n33\t36\tamod\n34\t36\tamod\n35\t36\tcompound\n36\t31\tnmod:over\n37\t42\tcase\n38\t42\tdet\n39\t42\tamod\n40\t42\tcompound\n41\t42\tcompound\n42\t31\tnmod:in\n43\t42\tadvmod\n44\t5\tpunct', u'ner': u'{"O","NATIONALITY","O","O","O","O","O","O","O","O","DATE","O","O","O","O","O","O","O","O","O","MISC","O","STATE_OR_PROVINCE","O","O","O","ORGANIZATION","ORGANIZATION","O","O","O","O","O","IDEOLOGY","PERSON","PERSON","O","O","O","COUNTRY","ORGANIZATION","O","O","O"}', u'subject_head': u'35', u'subject_end': u'36'}
    from text.dataset import Example
    ex = Example(d)
    words = parse_words(ex.words)
    print ' '.join(words[int(d[u'subject_begin']):int(d[u'subject_end'])]), ' '.join(words[int(d[u'object_begin']):int(d[u'object_end'])])
    print ' '.join(words)

    for line in ex.dependency.split("\n"):
        child, parent, arc = line.split("\t")
        print words[int(child)-1], words[int(parent)-1], arc

    print
    print 'shortest path'
    for child, parent, arc in get_path_from_parse(ex.dependency, int(ex.subject_begin), int(ex.subject_end), int(ex.object_begin), int(ex.object_end)):
        print words[child], words[parent], arc

