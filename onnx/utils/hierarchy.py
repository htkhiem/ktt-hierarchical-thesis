"""
Definition of the hierarchy metadata class.

This class is used to store the necessary hierarchical information
for all DistilBERT-based models.
"""
import json
from functools import reduce
import torch
import numpy as np


class PerLevelHierarchy:
    """The hierarchy structure in use by all DistilBERT-based models."""

    def __init__(
            self,
            codes=None,
            cls2idx=None,
            classes=None,
            levels=None,
            level_offsets=None,
            build_parent=True,
            parent_of=None,
            build_R=True,
            build_M=True
    ):
        """
        Construct a hierarchy structure from the dataset.

        - level_sizes is a list of (distinct) class counts per hierarchical
        level. Its length dictates the maximum hierarchy construction depth.
        - classes is the list of distinct classes, in the order we have
        assembled.
        """
        self.parent_of = None
        self.R = None
        self.M = None
        self.levels = []
        self.classes = []
        self.level_offsets = []

        # Non-basic constructor requested
        if codes is not None and cls2idx is not None:
            if levels is None:
                self.levels = [
                    len(d.keys()) for d in cls2idx
                ]  # TODO: Rename to level_sizes
            else:
                self.levels = levels
            if classes is None:
                self.classes = reduce(lambda acc, elem: acc + elem, [
                    list(d.keys()) for d in cls2idx
                ], [])
            else:
                self.classes = classes
            # Where each level starts in a global n-hot category vector
            # Its last element is coincidentally the length, which also allows
            # us to simplify the slicing code by blindly doing
            # [offset[i] : offset[i+1]].
            if level_offsets is None:
                self.level_offsets = reduce(
                    lambda acc, elem: acc + [acc[len(acc) - 1] + elem], self.levels, [0]
                )
            else:
                self.level_offsets = level_offsets

            if parent_of is not None:
                self.parent_of = [
                    torch.LongTensor(lvl)
                    for lvl in parent_of
                ]
            elif build_parent:
                # Use -1 to indicate 'undiscovered'
                self.parent_of = [
                    torch.LongTensor([-1] * level_size)
                    for level_size in self.levels
                ]
            if build_R:
                self.R = torch.zeros(
                    (self.levels[-1], len(self.classes)),
                    dtype=bool
                )
                # Every leaf has an edge to itself
                self.R[
                    np.arange(self.levels[-1]),
                    np.arange(
                        self.level_offsets[-2],
                        self.level_offsets[-2] + self.levels[-1]
                    )
                ] = 1
            if build_M:
                n = len(self.classes)
                self.M = torch.zeros((n, n), dtype=torch.bool)
                # Every class is an ancestor of itself
                self.M.fill_diagonal_(1)

            for lst in codes:
                if build_parent and parent_of is None:
                    # Build parent() (Algorithm 1)
                    # First-level classes' parent is root, but here we set them
                    # to themselves.
                    # This effectively zeroes out the hierarchical loss for
                    # this level.
                    self.parent_of[0][lst[0]] = lst[0]
                    for i in range(1, len(self.levels)):
                        child_idx = lst[i]
                        parent_idx = lst[i-1]
                        if self.parent_of[i][child_idx] == -1:
                            self.parent_of[i][child_idx] = parent_idx

                if build_R:
                    # Build R (Algorithm 2)
                    # Get level-local leaf index. Assume codes column has
                    # already been trimmed to the used depth.
                    leaf_idx = lst[-1]
                    for level, code in enumerate(lst[:-1]):
                        self.R[leaf_idx, code + self.level_offsets[level]] = 1

                if build_M:
                    # Build M (for C-HMCNN). Basically R, but for all classes
                    # instead of just leaf classes.
                    for i, ancestor in enumerate(lst):
                        ancestor_idx = ancestor + self.level_offsets[i] - 1
                        for j, offspring in enumerate(lst[i+1:]):
                            offspring_idx = offspring + self.level_offsets[j+i+1] - 1
                            self.M[ancestor_idx, offspring_idx] = 1

    @classmethod
    def from_dict(cls, h_dict):
        """
        Create a new PerLevelHierarchy instance from a Python dict.

        Said dict must be in the schema exported by to_dict().
        """
        instance = cls()
        instance.classes = h_dict['classes']
        instance.level_offsets = h_dict['level_offsets']
        instance.levels = h_dict['level_sizes']
        instance.parent_of = [
            torch.LongTensor(level)
            for level in h_dict['parent_of']
        ]
        if h_dict['M'] is not None:
            instance.M = torch.BoolTensor(h_dict['M'])
        if h_dict['R'] is not None:
            instance.R = torch.BoolTensor(h_dict['R'])

        return instance

    @classmethod
    def from_json(cls, json, config=None):
        """
        Create a new PerLevelHierarchy instance from a JSON string (or path).

        If a path string is specified, the corresponding JSON file on disk will
        be used.
        Said dict must be in the schema exported by to_dict().
        If config is specified, the 'device' key will be read and the instance
        will be moved to the specified device.
        """
        if isinstance(json, str):
            with open(json, 'r') as f:
                serial = f.read()
        else:
            serial = json
        instance = cls(
            config,
            [],
            [],
            False,
            False,
            False
        )
        instance.classes = serial['classes']
        instance.level_offsets = serial['level_offsets']
        instance.levels = serial['level_offsets']
        instance.parent_of = [
            torch.LongTensor(level).to(config['device'])
            for level in serial['parent_of']
        ]
        if 'M' in serial.keys():
            instance.M = torch.BoolTensor(serial['M'])
        if 'R' in serial.keys():
            instance.R = torch.BoolTensor(serial['R'])

        if config is not None:
            instance.to(config['device'])

        return instance

    def to_dict(self):
        """Export data from this hierarchy object as a Python dict."""
        parent_of = [p.tolist() for p in self.parent_of]
        d = {
            'classes': self.classes,
            'level_offsets': self.level_offsets,
            'level_sizes': self.levels,
            'parent_of': parent_of
        }
        if hasattr(self, 'M'):
            d['M'] = self.M
        if hasattr(self, 'R'):
            d['R'] = self.R
        return d

    def to_json(self, path=None):
        """
        Serialise this hierarchy metadata object into JSON.

        If a path is specified, it will also write the JSON into a file.
        """
        parent_of = [p.tolist() for p in self.parent_of]
        hierarchy_json = {
            'classes': self.classes,
            'level_offsets': self.level_offsets,
            'level_sizes': self.levels,
            'parent_of': parent_of
        }
        # Special metadata for some models
        if self.M is not None:
            hierarchy_json['M'] = self.M.int().tolist()
        if self.R is not None:
            hierarchy_json['R'] = self.R.int().tolist()
        if path is not None:
            with open(path, "w") as outfile:
                # Convert parent_of back to lists
                json.dump(hierarchy_json, outfile)
        return hierarchy_json

    def to(self, device):
        """
        Move PyTorch tensors to specified device.

        This is implemented to expose a PyTorch-like interface to this class.
        """
        if self.parent_of is not None:
            for i, level in enumerate(self.parent_of):
                self.parent_of[i] = level.to(device)
        if self.M is not None:
            self.M.to(device)
        if self.R is not None:
            self.R.to(device)

        return self
