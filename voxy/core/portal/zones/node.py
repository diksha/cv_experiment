import json
from collections import defaultdict, deque

# trunk-ignore(flake8/E731)
Node = lambda: {"is_parent": True, "data": set()}


class CameraLocationTree:
    def __init__(self):
        # root[orgnization] = dict (key: parent , val: child )
        self.root = defaultdict(lambda: defaultdict(Node))

    # insert a (child, parent) dep tuple into the graph
    def insert(self, relationship):

        # need to type check the relationship to have all the field
        # trunk-ignore(pylint/W0621)
        child, parent, organization, cam_list = relationship

        # if parent in self.idx[organization]:
        if child:
            self.root[organization][parent]["data"].add(child)
        else:
            self.root[organization][parent]["data"] = cam_list
            self.root[organization][parent]["is_parent"] = False

        # check if this is duplicated. If duplicated (child, parent), we will ignore it.
        # if self.trie[organization][parent] = child
        return True

    def get_tree(self, organization, root_key):

        result = defaultdict(dict)
        queue = deque([(root_key, result)])
        while queue:
            node_key, sub_tree = queue.popleft()
            if self.root[organization][node_key]["is_parent"]:
                for child_key in self.root[organization][node_key]["data"]:
                    new_sub_tree = defaultdict(dict)
                    sub_tree[child_key] = new_sub_tree
                    queue += ((child_key, new_sub_tree),)
            else:
                sub_tree[
                    node_key
                ] = f'{self.root[organization][node_key]["data"]}'
        return result


if __name__ == "__main__":

    # (child, parent) record
    db_record_with_leaf = [
        (2, 1),
        (3, 1),
        (4, 3),
        (5, 3),
        (6, 3),
        (8, 6),
        (9, 6),
        (8, None),
        (9, None),
        (4, None),
        (5, None),
        (6, None),
    ]
    db_record = [(2, 1), (3, 1), (4, 3), (5, 3), (6, 3), (8, 6), (9, 6)]

    cam_location = CameraLocationTree()

    # populate the full record (child, parent, organization, camera list)
    for child, parent in db_record_with_leaf:
        if parent:
            cam_location.insert((child, parent, "USCold", []))
        else:
            cam_location.insert(
                (
                    child,
                    parent,
                    "USCold",
                    ["cam" + str(child), "cam_alt_" + str(child)],
                )
            )

    print(
        json.dumps(
            cam_location.get_tree("USCold", 1), indent=4, sort_keys=True
        )
    )
