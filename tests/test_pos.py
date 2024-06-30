import itertools
import math

import pytest

from amaze.simu.pos import Vec, Pos

COORDS = [v / 2 for v in range(-4, 5)]


def test_vec_null():
    print(Vec.null())
    assert Vec.null() == Vec.null()
    assert not Vec.null()
    assert not (Vec.null() == [])
    assert Vec.null() != []
    assert (Vec.null() + Vec.null()) == Vec.null()

    null = Vec.null()
    null += Vec.null()
    assert Vec.null() == null
    null -= Vec.null()
    assert Vec.null() == null
    assert Vec.null() / 2 == Vec.null()

    assert Vec.null().is_null()
    assert Vec.null().length() == 0

    assert Vec.from_polar(0, 0) == Vec.null()
    assert Vec.from_polar(0, 1) == Vec(1, 0)
    assert Vec.from_polar(0.5 * math.pi, 1) == Vec(0, 1)
    assert Vec.from_polar(math.pi, 1) == Vec(-1, 0)
    assert Vec.from_polar(1.5 * math.pi, 1) == Vec(0, -1)

    assert Pos.null() == Vec.null()
    assert Pos.null().aligned() == Vec.null()


@pytest.mark.parametrize("coords", [(x, y) for x, y in itertools.product(COORDS, COORDS)])
def test_vec(coords):
    p0 = Vec(*coords)
    print(p0)
    print(str(p0))
    print(repr(p0))
    print(hash(p0))
    assert not (p0.x == 0 and p0.y == 0) or p0 == Vec.null()
    assert not (p0.x == 0 and p0.y == 0) or p0.is_null()

    p0 += Vec.null()
    assert p0 == Vec(*coords)

    p0 += p0
    assert p0 == 2 * Vec(*coords)

    p0 -= Vec(*coords)
    assert p0 == Vec(*coords)

    assert (2 * p0) / 2 == p0
    assert (p0 / 2) * 2 == p0
    assert (0.5 * p0) / 0.5 == p0
    assert (p0 / 0.5) * 0.5 == p0

    assert all(lhs == rhs for lhs, rhs in zip(p0, iter(p0)))
    pytest.raises(IndexError, p0.__getitem__, -1)
    pytest.raises(IndexError, p0.__getitem__, 2)

    assert (2 * p0).length() == p0.length() * 2

    print(p0)
    p1 = p0.copy()
    assert p0 == p1
    p1 += p0 + (0.1, 0.1)
    assert p0 != p1

    assert Vec(*coords) == Vec(*p0)
    assert p0 == Vec.from_polar(math.atan2(p0.y, p0.x), p0.length())

    assert Pos(*p0) == p0
    assert Pos(*p0).aligned() == tuple(int(v) for v in p0)
