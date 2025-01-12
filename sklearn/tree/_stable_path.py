from attrs import define


@define(slots=True, frozen=True)
class DecodedStablePath:
    """
    A class representing the part of a path (from root to `node_id`) within a decision tree that will not change upon label flip.
    If a path is vulnerable, flipping the label of the corresponding instance will cause the path to change below the
    `node_id` and `node_id` itself only. If instead a path is not vulnerable, then the path will not change

    This class object is `frozen`, which means that it is immutable and cannot be changed after creation.

    Parameters
    ----------
    vulnerable: `bool`
        Whether the path is vulnerable or not.
    node_id: `int`
        The node id where the path changes.
    samples: `int`
        The number of samples that reach the last step of the path.
    target: `int`
        The target instance.
    depth: `int`
        The depth of the node of the tree.
    label: `int`
        The label of the target instance.
    path: `str`
        The encoded path of the tree, composed of `L` for left and `R` for right. If the path terminates with `*`, then
        it means that the path is vulnerable after the last path step.
    """

    vulnerable: bool
    node_id: int
    samples: int
    target: int
    depth: int
    label: int
    path: str

    def compute_subtree_params(self, params: dict) -> dict:
        """
        Compute the parameters for a subtree retrain based on the current parameters and the depth of the node.
        This method will return an updated set of parameters, where the `max_depth` parameter will be updated to
        reflect the depth of the node.

        Parameters
        ----------
        params: `dict`
            The parameters of the original tree.

        Returns
        -------
        `dict`
            The new parameters for the subtree.
        """

        max_depth = params.get("max_depth", None)
        new_param = params.copy()

        if max_depth is not None:
            new_param["max_depth"] = max_depth - self.depth

        return new_param
