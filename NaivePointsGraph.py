import numpy as np


class NaivePointsGraph:
    def __init__(self, dist_acc=20, colinear_acc=np.pi / 10):
        self.dist_acc = dist_acc
        self.colinear_acc = colinear_acc
        self._points = {}  # (long, lat)
        self._edges = {}  # (point_idx, point_idx)
        self._last_v = -1

    def insert_vertex(self, long, lat, debug=False):
        closest = self._find_closest(long, lat)
        if closest is None or self.dist(long, lat, *self.get_vertex(closest)) > self.dist_acc:
            self._last_v += 1
            self._points[self._last_v] = (long, lat)
            self._edges[self._last_v] = set()
        elif debug:
            print(f'Merging point with {self.get_vertex(closest)}')

    def insert_edge(self, long1, lat1, long2, lat2, debug=False):
        # find key-points
        ui1 = self._find_closest(long1, lat1)
        ui2 = self._find_closest(long2, lat2)
        if (ui1 != ui2 and
                self.dist(long1, lat1, *self.get_vertex(ui1)) <= self.dist_acc and
                self.dist(long2, lat2, *self.get_vertex(ui2)) <= self.dist_acc):
            # snap to key-points
            self._edges[ui1].add(ui2)
            self._edges[ui2].add(ui1)
            # check if neighbors can be removed due to co-linearity
            self._remove_colinear(ui1)
            self._remove_colinear(ui2)
        elif debug:
            print('Edge cannot be added, keypoint for one of the points was not found')

    def get_vertex(self, vi):
        return self._points[vi]

    def get_edges(self, vi):
        return self._edges[vi]

    def list_vertices(self):
        return self._points.values()

    def list_edges(self):
        return [(*self.get_vertex(k), *self.get_vertex(v)) for k, n in self._edges.items() for v in n if v > k]

    def _find_closest(self, long, lat):
        if len(self._points) == 0:
            return None
        return min(self._points.keys(), key=lambda i: self.dist(long, lat, *self.get_vertex(i)))

    def _remove_colinear(self, vi):
        if len(self.get_edges(vi)) != 2:
            return
        v = self.get_vertex(vi)
        ui1, ui2 = self.get_edges(vi)
        u1, u2 = self.get_vertex(ui1), self.get_vertex(ui2)
        a1, a2 = self.azimuth(*u1, *v), self.azimuth(*v, *u2)
        if a1 - self.colinear_acc <= a2 <= a1 + self.colinear_acc:
            del self._edges[vi]
            del self._points[vi]
            self._edges[ui1].remove(vi)
            self._edges[ui2].remove(vi)
            self._edges[ui1].add(ui2)
            self._edges[ui2].add(ui1)

    @staticmethod
    def dist(long1, lat1, long2, lat2):
        """
            a = sin²(Δϕ/2) + cos(ϕ1)cos(ϕ2)sin²(Δλ/2)
            d = 2R*arctan2(a,1−a)
        """
        long1, lat1 = np.deg2rad(long1), np.deg2rad(lat1),
        long2, lat2 = np.deg2rad(long2), np.deg2rad(lat2),
        R = 6_371_000
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((long2 - long1) / 2) ** 2
        d = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return d

    @staticmethod
    def azimuth(long1, lat1, long2, lat2):
        """
            θ = arctan2(sin(Δλ)cos(ϕ2), cos(ϕ1)sin(ϕ2) − sin(ϕ1)cos(ϕ2)cos(Δλ))
        """
        long1, lat1 = np.deg2rad(long1), np.deg2rad(lat1),
        long2, lat2 = np.deg2rad(long2), np.deg2rad(lat2),
        t = np.arctan2(np.sin(long2 - long1) * np.cos(lat2),
                       np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(long2 - long1))
        return t
