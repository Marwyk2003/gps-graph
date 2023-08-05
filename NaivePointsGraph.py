import numpy as np


class NaivePointsGraph:
    def __init__(self, dist_acc=0.001):
        self._points = {}  # (long, lat)
        self._edges = {}  # (point_idx, point_idx)
        self.dist_acc = dist_acc
        self._last_v = -1

    def insert_vertex(self, long, lat, debug=False):
        closest = self.find_closest(long, lat)
        if closest is None or self.dist(long, lat, *self.get_vertex(closest)) > self.dist_acc:
            self._last_v += 1
            self._points[self._last_v] = (long, lat)
            self._edges[self._last_v] = set()
        elif debug:
            print(f'Merging point with {self.get_vertex(closest)}')

    def insert_edge(self, long1, lat1, long2, lat2, debug=False):
        closest1 = self.find_closest(long1, lat1)
        closest2 = self.find_closest(long2, lat2)
        if (closest1 != closest2 and
                self.dist(long1, lat1, *self.get_vertex(closest1)) <= self.dist_acc and
                self.dist(long2, lat2, *self.get_vertex(closest2)) <= self.dist_acc):
            self._edges[closest1].add(closest2)
            self._edges[closest2].add(closest1)
        elif debug:
            print('Edge cannot be added, keypoint for one of the points was not found')

    def insert_edge_i(self, vi, ui, debug=False):
        self._edges[vi].add(ui)
        self._edges[ui].add(vi)

    def get_vertex(self, vi):
        return self._points[vi]

    def get_edges(self, vi):
        return self._edges[vi]

    def remove_vertex(self, vi):
        del self._edges[vi]
        del self._points[vi]

    def remove_edge(self, vi, ui):
        self._edges[ui].remove(vi)
        self._edges[vi].remove(ui)

    def list_vertices(self):
        return list(zip(*self._points.values()))

    def list_edges(self):
        return [(*self.get_vertex(k), *self.get_vertex(v)) for k, n in self._edges.items() for v in n if v > k]

    def find_closest(self, long, lat):
        if len(self._points) == 0:
            return None
        return min(range(len(self._points)), key=lambda i: self.dist(long, lat, *self.get_vertex(i)))

    def size(self):
        return len(self._points)

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
