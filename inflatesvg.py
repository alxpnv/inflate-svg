from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from itertools import accumulate
from math import atan2, cos, radians, sqrt
from operator import itemgetter
from pathlib import Path

from svgpathtools import Arc, hex2rgb, path_encloses_pt, svg2paths2

def circle(p, d):
    return d * sqrt(1 - (p - 1)**2)

def interpolate(x, a, b):
    return (x - a) / (b - a)

def triangulate(points, edges):
    def calc_angle(i1, i2, i3, orientationonly=False):
        v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
        crossproduct = (v2.real - v1.real) * (v3.imag - v1.imag) - (v2.imag - v1.imag) * (v3.real - v1.real)

        if orientationonly:
            return crossproduct

        dotproduct = (v2.real - v1.real) * (v3.real - v1.real) + (v2.imag - v1.imag) * (v3.imag - v1.imag)
        angle = atan2(crossproduct, dotproduct)

        return angle

    def remove_point(p1):
        for i, (cw, ccw) in edgedata.items():
            ncw  = [(p, o) for (p, o) in cw  if p != p1]
            nccw = [(p, o) for (p, o) in ccw if p != p1]
            edgedata[i] = (ncw, nccw)

    def triangulate_poly(poly):
        startpt = poly[0]
        for i in range(len(poly) - 2):
            tris.append([startpt, poly[i + 1], poly[i + 2]])

    if (len(points)) < 3:
        return []

    i = 0
    edgedata = {}
    for (p1, p2) in edges:
        cw, ccw  = [], []

        for p in points:
            if p == p1 or p == p2:
                continue

            a = calc_angle(p1, p2, p)
            if a > 0:
                cw.append((p, a))
            else:
                ccw.append((p, a))

        cw.sort(key=itemgetter(1), reverse=True)
        ccw.sort(key=itemgetter(1), reverse=True)
        edgedata[i] = (cw, ccw)
        i += 1

    done = True
    for (cw, ccw) in edgedata.values():
        if cw and ccw:
            done = False
            break

    edgedata = dict(sorted(edgedata.items(), key=lambda item: min(len(item[1][0]), len(item[1][1]))))

    lastpoly = []
    polygons = []
    tris     = []
    if done:
        if not edges:
            if calc_angle(points[0], points[1], points[3], orientationonly=True) > 0:
                tris.append([points[0], points[3], points[1]])
            else:
                tris.append([points[0], points[1], points[2]])
            if calc_angle(points[0], points[2], points[3], orientationonly=True) > 0:
                tris.append([points[0], points[3], points[2]])
            else:
                tris.append([points[0], points[2], points[3]])
        else:
            p1, p2  = edges[0]
            cw, ccw = edgedata[0]
            if cw:
                polygons.append([p1, *[j for (j, _) in cw], p2])
            if ccw:
                polygons.append([p1, p2, *[j for (j, _) in ccw]])
    else:
        for i, (cw, ccw) in edgedata.items():
            if not cw or not ccw:
                continue
            p1, p2 = edges[i]

            # corner triangles
            if len(cw) == 1:
                p = cw[0][0]
                tris.append([p1, p, p2])
                remove_point(p)
                lastpoly = [p1, p2, *[j for (j, _) in ccw]]
            if len(ccw) == 1:
                p = ccw[0][0]
                tris.append([p1, p2, p])
                remove_point(p)
                lastpoly = [p1, *[j for (j, _) in cw], p2]

            if len(cw) != 1 and len(ccw) != 1:
                if len(cw) < len(ccw):
                    polygons.append([p1, *[j for (j, _) in cw], p2])
                    lastpoly = [p1, p2, *[j for (j, _) in ccw]]
                    for (p, _) in cw:
                        remove_point(p)
                else:
                    polygons.append([p1, p2, *[j for (j, _) in ccw]])
                    lastpoly = [p1, *[j for (j, _) in cw], p2]
                    for (p, _) in ccw:
                        remove_point(p)

        if lastpoly:
            polygons.append(lastpoly)

    for poly in polygons:
        triangulate_poly(poly)

    return tris

def depth_range(s):
    d = int(s)
    if 0 <= d <= 100:
        return d
    else:
        raise ArgumentTypeError('Depth not in range [0-100]')

parser = ArgumentParser(prog='inflatesvg.py')
parser.add_argument('svgfile')
parser.add_argument('-B', '--both-sides', action=BooleanOptionalAction, help='inflate both sides')
parser.add_argument('--depth', type=depth_range, default=50, metavar="[0-100]", help='depth')
parser.add_argument('--resolution', type=int, choices=[32, 64, 128], default=32, help='resolution on X-axis')
args = parser.parse_args()
infile = Path(args.svgfile)
name = infile.stem

objects, attributes, svg_attributes = svg2paths2(args.svgfile)

style = attributes[0]['style']
fill = hex2rgb(style.split(';')[0].split(':')[1])
r, g, b = map(lambda x: x / 255.0, fill)

# WORKAROUND: radialrange() is not implemented for arcs
for i in range(len(objects[0])):
    if isinstance(objects[0][i], Arc):
        objects[0][i] = next(objects[0][i].as_cubic_curves(1))

paths = objects[0].continuous_subpaths()
if any(not p.isclosed() for p in paths):
    exit("Path is not closed")

xmin, xmax, ymin, ymax = paths[0].bbox()
width, height  = xmax - xmin, ymax - ymin
xmid, ymid = width  / 2, height / 2
opt = complex(xmax + 1, ymax + 1)

resx = args.resolution
dx = width  / resx
resy = int(height / dx)
dy = height / resy

# compute distance map
vnum     = 0
vertices = []
distmap  = [[0 for j in range(resx + 2)] for i in range(resy + 2)]
grid = [[{} for j in range(resx + 2)] for i in range(resy + 2)]
for i in range(resy + 2):
    for j in range(resx + 2):
        x = xmin + j * dx
        y = ymin + i * dy
        pt = complex(x, y)

        distmap[i][j] = min(p.radialrange(pt)[0][0] for p in paths)

        insidefigure = path_encloses_pt(pt, opt, paths[0])
        insidehole = any(path_encloses_pt(pt, opt, p) for p in paths[1:])

        grid[i][j] = {'tl': [], 'bl': [], 'br': [], 'tr': []}
        if not insidefigure or insidehole:
            distmap[i][j] = -distmap[i][j]
        else:
            grid[i][j]['tl'] = [vnum]
            if i > 0:
                grid[i - 1][j]['bl'] = [vnum]
            if j > 0:
                grid[i][j - 1]['tr'] = [vnum]
            if i > 0 and j > 0:
                grid[i - 1][j - 1]['br'] = [vnum]
            vertices.append(pt)
            vnum += 1
ngrid = vnum

alpha = radians(5)
maxdist = max(map(max, distmap))

vdists = []
for i in range(resy + 2):
    for j in range(resx + 2):
        if distmap[i][j] < 0:
            if -distmap[i][j] > maxdist:
               distmap[i][j] = -maxdist
            distmap[i][j] /= maxdist
        else:
            distmap[i][j] /= maxdist
            vdists.append(distmap[i][j])

# compute isolines using the marching squares algorithm
isovalues = (1 - cos(i * alpha) for i in range(4))
cases = (
    [],
    [['b', 'l']],
    [['b', 'r']],
    [['r', 'l']],
    [['r', 't']],
    [['b', 'r'], ['t', 'l']],
    [['b', 't']],
    [['t', 'l']],
    [['t', 'l']],
    [['b', 't']],
    [['b', 'l'], ['r', 't']],
    [['r', 't']],
    [['r', 'l']],
    [['b', 'r']],
    [['b', 'l']],
    []
)

vdict  = {}
vcount = ngrid
edges  = [[[] for j in range(resx + 1)] for i in range(resy + 1)]
for d in isovalues:
    cache = {}
    threshold = [[1 if distmap[i][j] > d else 0 for j in range(resx + 2)] for i in range(resy + 2)]
    for i in range(1, resy + 1):
        for j in range(0, resx):
            x = xmin + j * dx
            y = ymin + i * dy

            index = threshold[i][j] + 2 * threshold[i][j + 1] + 4 * threshold[i - 1][j + 1] + 8 * threshold[i - 1][j]
            case = cases[index]

            for c in case:
                points = []
                for k in c:
                    adding = True
                    n = vnum
                    if k == 'b':
                        coeff = interpolate(d, distmap[i][j], distmap[i][j + 1])
                        pt = complex(x + dx * coeff, y)
                    elif k == 'r':
                        coeff = interpolate(d, distmap[i][j + 1], distmap[i - 1][j + 1])
                        pt = complex(x + dx, y - dy * coeff)
                    elif k == 't':
                        if i > 0:
                            n = cache[i - 1, j]['b']
                            adding = False
                        else:
                            coeff = interpolate(d, distmap[i - 1][j], distmap[i - 1][j + 1])
                            pt = complex(x + dx * coeff, y - dy)
                    elif k == 'l':
                        if j > 0:
                            n = cache[i, j - 1]['r']
                            adding = False
                        else:
                            coeff = interpolate(d, distmap[i][j], distmap[i - 1][j])
                            pt = complex(x, y - dy * coeff)

                    if (i, j) not in cache:
                        cache[i, j] = {}

                    if adding:
                        vertices.append(pt)
                        cache[i, j][k] = vnum
                        vnum += 1
                    else:
                        cache[i, j][k] = n
                    points.append(n)

                edges[i][j].append(points)
    vdict[d] = len(vertices) - vcount
    vcount = len(vertices)

# triangulation
triangles = []
for i in range(1, resy + 1):
    for j in range(0, resx):
        tl = grid[i - 1][j]['tl']
        tr = grid[i - 1][j]['tr']
        br = grid[i - 1][j]['br']
        bl = grid[i - 1][j]['bl']

        points = tl + tr + br + bl
        if not points and not edges:
            continue

        if not edges[i][j]:
            if len(points) == 3:
                triangles.append(points)
            elif len(points) == 4:
                imid, jmid = resy // 2, resx // 2
                if (j < jmid and i <= imid) or (j >= jmid and i > imid):
                    triangles.append([points[0], points[3], points[1]])
                    triangles.append([points[1], points[3], points[2]])
                else:
                    triangles.append([points[0], points[2], points[1]])
                    triangles.append([points[0], points[3], points[2]])
            continue

        for (p1, p2) in edges[i][j]:
            if p1 not in points:
                points.append(p1)
            if p2 not in points:
                points.append(p2)

        triangles += triangulate(points, edges[i][j])

keys   = list(vdict.keys())
sums   = list(accumulate([ngrid] + [*vdict.values()]))
ranges = list(zip(sums[0:-1], sums[1:]))
border = ranges[0]
nvert  = len(vertices)

depth = args.depth * (maxdist / 100)
with open(infile.with_suffix('.obj'), 'w') as obj:
    obj.write("o {:s}\n".format(name))
    obj.write("mtllib heart.mtl\n")
    obj.write("usemtl mat_{:s}\n".format(name))
    obj.write("s 1\n")

    obj.write("\n# vertices\n")
    for i in range(ngrid):
        vx, vy = vertices[i].real, vertices[i].imag
        d = circle(vdists[i], depth)
        obj.write("v {:.4f} {:.4f} {:.4f}\n".format(vx - xmin - xmid, -vy + ymin + ymid, d))
    obj.write("\n")
    for i in range(len(keys)):
        d = circle(keys[i], depth)
        for j in range(*ranges[i]):
            vx, vy = vertices[j].real, vertices[j].imag
            obj.write("v {:.4f} {:.4f} {:.4f}\n".format(vx - xmin - xmid, -vy + ymin + ymid, d))

    obj.write("\n# triangles\n")
    for (v1, v2, v3) in triangles:
        obj.write("f {:d} {:d} {:d}\n".format(v1 + 1, v2 + 1, v3 + 1))

    if depth != 0 and args.both_sides:
        obj.write("\n# mirrored vertices\n")
        for i in range(ngrid):
            vx, vy = vertices[i].real, vertices[i].imag
            d = circle(vdists[i], depth)
            obj.write("v {:.4f} {:.4f} {:.4f}\n".format(vx - xmin - xmid, -vy + ymin + ymid, -d))
        obj.write("\n")
        for i in range(1, len(keys)):
            d = circle(keys[i], depth)
            for j in range(*ranges[i]):
                vx, vy = vertices[j].real, vertices[j].imag
                obj.write("v {:.4f} {:.4f} {:.4f}\n".format(vx - xmin - xmid, -vy + ymin + ymid, -d))

        obj.write("\n# mirrored triangles\n")
        for t in triangles:
            for i in range(3):
                if t[i] < border[0]:
                    t[i] += nvert
                elif t[i] >= border[1]:
                    t[i] += nvert - (border[1] - border[0])
            obj.write("f {:d} {:d} {:d}\n".format(t[0] + 1, t[1] + 1, t[2] + 1))

with open(infile.with_suffix('.mtl'), 'w') as mtl:
    mtl.write("newmtl mat_{:s}\n".format(name))
    mtl.write("\tKa {:.4f} {:.4f} {:.4f}\n".format(r, g, b))
    mtl.write("\tKd {:.4f} {:.4f} {:.4f}\n".format(r, g, b))
    mtl.write("\tKs 1.0000 1.0000 1.0000\n")
    mtl.write("\tNs 1000.0000\n")
    mtl.write("\tillum 1\n")
