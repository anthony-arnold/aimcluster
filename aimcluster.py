import numpy


class DimensionError(Exception):
    pass


def normalise(v):
    """
    Normalise a 3D vector.

    :param v: A vector to normalise.
    :return: The normalised vector.
    """
    norm = numpy.sqrt(numpy.dot(v, v))
    return v / norm

class Line:
    """
    A line reprented by a point, c, and a vector v, such that the line is
    described as the set of points, p such that p = c + v t where t is scalar.
    """
    def __init__(self, c, v):
        """
        Initialise a line.

        :param c: A point on the line.
        :param v: A vector.
        """
        # Check c and v are in R_3
        if numpy.size(c) != 3:
            raise DimensionError("Expected c to be three-dimensional")
        if numpy.size(v) != 3:
            raise DimensionError("Expected v to be three-dimensional")

        self.c = numpy.asarray(c, float).flatten()
        self.v = numpy.asarray(v, float).flatten()


class Plane:
    """
    A representation of a plane normal to the vector N and
    containing the point P.
    """

    def __init__(self, n, p):
        """
        Initialise a plane.

        :param n: A vector normal to the plane.
        :param p: A point on the plane, not colinear with n.
        """
        # Check n and p are in R_3
        if numpy.size(n) != 3:
            raise DimensionError("Expected n to be three-dimensional")
        if numpy.size(p) != 3:
            raise DimensionError("Expected p to be three-dimensional")

        _n = normalise(n)

        self.n = numpy.asarray(_n, float).flatten()
        self.p = numpy.asarray(p, float).flatten()

        self._proj = None

    def _get_projection(self):
        """
        Calculate the projection from 3D onto this plane.
        """
        d = numpy.dot(self.n, self.p)
        u = self.p - self.n * d
        x = normalise(u)
        y = numpy.cross(self.n, x)

        self._proj = (x, y)

    def project(self, π):
        if self._proj is None:
            self._get_projection()

        return numpy.asarray([
            numpy.dot(π, x) for x in self._proj
        ])

def plane_line_intersect(plane, line):
    """
    Determine the point where the plane and line intersect.

    :param plane: An instance of Plane
    :param line: An instance of Line

    :return: An ndarray of size 3, denoting the intersection point, or None if
    the line is parallel to the plane.
    """
    up = numpy.dot(plane.p - line.c, plane.n)
    down = numpy.dot(line.v, plane.n)
    t = up / down

    if t == 0:
        return None

    return line.c + line.v * t


def reposition(bs, cs, vs, p, v0):
    """
    Given three starting positions, their respective final positions and
    velocities, and the desired end point, determine a new starting position.

    It's assumed that two of the starting positions are offset from the third
    position (not order-specific) in directions normal to the starting velocity
    and normal to each other.

    The starting velocity is considered constant.

    :param bs: The starting positions, a 3-tuple of 3-arrays.
    :param cs: The stopping positions, a 3-tuple of 3-arrays.
    :param vs: The stopping velocities, a 3-tuple of 3-arrays.
    :param p: The desired end point, a 3-array.
    :param v0: The starting velocity.
    :return: A point estimating the new start point.
    """
    # Determine the starting plane
    plane0 = Plane(v0, bs[0])

    # Get the projected offsets of the other starting points.
    rb = [b - bs[0] for b in bs[1:]]
    r0 = [plane0.project(r) for r in rb]

    # Determine the final plane
    plane1 = Plane(-vs[0], p)

    # Get the plane-intersection points for each cluster
    lines = [Line(c, v) for c, v in zip(cs, vs)]
    intersects = [plane_line_intersect(plane1, line) for line in lines]

    # Get the projected offsets of the end points.
    r1 = [plane1.project(p - c) for c in intersects]

    # Get the over-/under- correction factor.
    fs = [r1[0] / r for r in r1[1:]]

    # Adjust the initial offsets by the correction factor.
    # sum to get the final offset in projected space
    a = [r * f for r, f in zip(r0, fs)]
    r = numpy.sum(a, axis=0)
    r = numpy.append(r, 0)

    # Move the initial point by the offset in 3D space to get
    # the final repositioned starting point.
    basis = [normalise(b - bs[0]) for b in bs[1:]]
    basis.append(normalise(v0))

    final = numpy.sum([rc * b for rc, b in zip(r, basis)], axis=0)

    return bs[0] + final
