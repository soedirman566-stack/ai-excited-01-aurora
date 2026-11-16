#!/usr/bin/env python3
"""
Aurora Surfer — a generative digital painting.

A joyful penguin surfs a towering ocean wave woven from aurora-borealis
light: ribbons of electric green, teal and violet curl like a breaking
wave under a starry sky, and the spray at the crest dissolves into stars.

Rendered with Python + PIL/numpy at 2x supersampling, then downscaled
to 1024x1024 for a smooth, painterly finish.
"""

import math
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

SS = 2                      # supersampling factor
S = 1024                    # final size
W = H = S * SS              # working size

rng = random.Random(20261116)
nprng = np.random.default_rng(20261116)


# ---------------------------------------------------------------- helpers

def C(hexstr):
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def vgradient(size, stops):
    """Vertical gradient image from [(pos0..1, color), ...]."""
    w, h = size
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    arr = np.zeros((h, w, 3), dtype=np.float32)
    stops = sorted(stops)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        seg = (ys >= p0) & (ys <= p1)
        if not seg.any():
            continue
        t = (ys[seg] - p0) / max(p1 - p0, 1e-6)
        for ch in range(3):
            arr[seg, :, ch] = lerp(c0[ch], c1[ch], t)[:, None]
    arr[ys < stops[0][0]] = stops[0][1]
    arr[ys > stops[-1][0]] = stops[-1][1]
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def catmull_rom(points, samples_per_seg=24, closed=False):
    """Smooth a polyline through points with Catmull-Rom splines."""
    pts = list(points)
    if closed:
        pts = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        pts = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append(pts[-2])
    return out


def sc(points):
    """Scale 1024-space points to working size."""
    return [(x * SS, y * SS) for (x, y) in points]


def glow_paste(base, layer, blur, strength=1.0):
    """Screen-blend a blurred copy of layer over base for glow."""
    g = layer.filter(ImageFilter.GaussianBlur(blur))
    if strength != 1.0:
        g = g.point(lambda v: int(v * strength))
    return ImageChops.screen(base, g)


def fractal_noise(w, h, octaves=4, seed=7):
    r = np.random.default_rng(seed)
    total = np.zeros((h, w), dtype=np.float32)
    amp, size = 1.0, 8
    for _ in range(octaves):
        small = r.random((size, size)).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize(
            (w, h), Image.BICUBIC)
        total += amp * (np.asarray(img, dtype=np.float32) / 255.0)
        amp *= 0.55
        size *= 2
    total /= total.max()
    return total


AURORA = [C("#2bff9e"), C("#14e6c8"), C("#0fb9d8"),
          C("#7a5cff"), C("#c65bff"), C("#19f08a")]


def aurora_color(t):
    """Cycle through the aurora palette."""
    t = t % 1.0
    n = len(AURORA)
    i = int(t * n)
    return lerp_color(AURORA[i], AURORA[(i + 1) % n], (t * n) - i)


# ------------------------------------------------------------------- sky

def make_sky():
    sky = vgradient((W, H), [
        (0.00, C("#02040e")),
        (0.35, C("#061733")),
        (0.62, C("#0a2c47")),
        (0.78, C("#0b3d4e")),
        (0.80, C("#062238")),   # horizon -> sea
        (1.00, C("#020610")),
    ])

    # soft nebula glows
    neb = Image.new("RGB", (W, H), (0, 0, 0))
    nd = ImageDraw.Draw(neb)
    for (x, y, r, col) in [
            (240, 180, 420, C("#0d3b2e")), (700, 120, 520, C("#14204d")),
            (880, 300, 380, C("#0a2f45")), (150, 420, 300, C("#123a5c"))]:
        nd.ellipse([x * SS - r * SS, y * SS - r * SS,
                    x * SS + r * SS, y * SS + r * SS], fill=col)
    neb = neb.filter(ImageFilter.GaussianBlur(120 * SS))
    sky = ImageChops.screen(sky, neb)

    # stars
    star = Image.new("RGB", (W, H), (0, 0, 0))
    sd = ImageDraw.Draw(star)
    for _ in range(850):
        x = rng.uniform(0, S)
        y = rng.uniform(0, S * 0.82) ** 1.12
        b = rng.uniform(60, 255) * (1.0 - y / (S * 1.05))
        if b < 25:
            continue
        r = rng.choice([0.7, 0.7, 1.1, 1.6, 2.3])
        tint = rng.choice([(255, 255, 255), (200, 235, 255),
                           (210, 255, 230), (255, 244, 214)])
        col = tuple(int(tint[i] * b / 255) for i in range(3))
        sd.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
                   fill=col)
    sky = ImageChops.add(sky, star.filter(ImageFilter.GaussianBlur(0.7 * SS)),
                         1, 0)
    sky = glow_paste(sky, star, 5 * SS, 0.8)

    # moon + halo (upper right)
    moon = Image.new("RGB", (W, H), (0, 0, 0))
    md = ImageDraw.Draw(moon)
    mx, my, mr = 812, 168, 52
    md.ellipse([(mx - mr) * SS, (my - mr) * SS, (mx + mr) * SS, (my + mr) * SS],
               fill=C("#fdf6d8"))
    sky = glow_paste(sky, moon, 26 * SS, 0.9)
    sky = glow_paste(sky, moon, 7 * SS, 1.0)
    md2 = ImageDraw.Draw(sky)
    md2.ellipse([(mx - mr) * SS, (my - mr) * SS, (mx + mr) * SS, (my + mr) * SS],
                fill=C("#fffbe9"))
    for (cx, cy, cr, shade) in [(790, 150, 10, "#e8e0bd"), (832, 188, 7, "#eee5c2"),
                                (822, 138, 5, "#eae1c0")]:
        md2.ellipse([(cx - cr) * SS, (cy - cr) * SS, (cx + cr) * SS, (cy + cr) * SS],
                    fill=C(shade))
    return sky


# ------------------------------------------------------- ambient aurora

def ambient_aurora(base):
    """Faint aurora curtains high in the sky, echoing the wave."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for band in range(9):
        x0 = -80 + band * 95 + rng.uniform(-25, 25)
        phase = rng.uniform(0, math.tau)
        col = aurora_color(band / 9.0 + 0.08)
        width = rng.uniform(16, 34)
        for y in range(0, 470, 6):
            x = x0 + 66 * math.sin(y * 0.006 + phase) + y * 0.22
            a = int(66 * (1 - y / 470) ** 1.4 * rng.uniform(0.75, 1.0))
            ld.line([(x * SS, y * SS), ((x + width) * SS, y * SS)],
                    fill=col + (a,), width=int(7 * SS))
    layer = layer.filter(ImageFilter.GaussianBlur(9 * SS))
    base.alpha_composite(layer)
    return base


# ------------------------------------------------------------------- sea

def paint_sea(base):
    """Reflection shimmer on the dark sea (visible in the barrel hollow
    and along the bottom edge)."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for _ in range(210):
        y = rng.uniform(800, 1010)
        x = rng.uniform(0, S)
        ln = rng.uniform(14, 90) * (1.15 - (y - 800) / 260)
        col = aurora_color(rng.random())
        a = int(rng.uniform(14, 60) * (1.05 - (y - 800) / 240))
        ld.line([(x * SS, y * SS), ((x + ln) * SS, y * SS)],
                fill=col + (max(a, 0),), width=int(rng.uniform(1.2, 2.6) * SS))
    layer = layer.filter(ImageFilter.GaussianBlur(1.6 * SS))
    base.alpha_composite(layer)
    return base


# ------------------------------------------------------------------ wave

# Silhouette of the breaking wave (1024-space control points).
WAVE_CTRL = [
    (-30, 1054), (-30, 640), (30, 452), (150, 338), (300, 268),
    (428, 246), (512, 282), (540, 352), (512, 428), (440, 452),
    (452, 512), (540, 578), (672, 668), (830, 762), (1054, 838),
    (1054, 1054),
]
CREST_CTRL = WAVE_CTRL[2:10]        # the bright top edge / lip


def wave_mask():
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.polygon(sc(catmull_rom(WAVE_CTRL, 20)), fill=255)
    return mask


def wave_body():
    """Deep luminous base fill of the wave."""
    body = vgradient((W, H), [
        (0.20, C("#177550")),
        (0.42, C("#0e6470")),
        (0.66, C("#0a4066")),
        (1.00, C("#052644")),
    ])
    return body


def aurora_curtains():
    """Flowing aurora ribbons that will live inside the wave silhouette."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    bands = 52
    for band in range(bands):
        t = band / (bands - 1)
        x0 = lerp(-140, 1060, t) + rng.uniform(-14, 14)
        col = lerp_color(aurora_color(t * 0.95 + 0.02), (255, 255, 255), 0.16)
        phase = rng.uniform(0, math.tau)
        sway = rng.uniform(20, 44)
        freq = rng.uniform(0.004, 0.0075)
        drift = rng.uniform(0.30, 0.55)
        width = rng.uniform(13, 30)
        for y in range(215, 1030, 5):
            x = x0 + sway * math.sin(y * freq + phase) + (y - 215) * drift
            fade_top = min(1.0, (y - 215) / 90)
            fade_bot = min(1.0, (1030 - y) / 200)
            a = int(150 * fade_top * fade_bot * rng.uniform(0.8, 1.0))
            ld.line([(x * SS, y * SS), ((x + width) * SS, (y + 5) * SS)],
                    fill=col + (a,), width=int(6 * SS))
    layer = layer.filter(ImageFilter.GaussianBlur(5 * SS))

    # bright energy threads inside the curtains
    threads = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(threads)
    for band in range(26):
        t = band / 25
        x0 = lerp(-100, 1020, t) + rng.uniform(-10, 10)
        col = aurora_color(t * 0.95 + 0.04)
        phase = rng.uniform(0, math.tau)
        sway = rng.uniform(24, 46)
        freq = rng.uniform(0.004, 0.007)
        drift = rng.uniform(0.32, 0.52)
        pts = []
        for y in range(230, 1000, 8):
            x = x0 + sway * math.sin(y * freq + phase) + (y - 230) * drift
            pts.append((x * SS, y * SS))
        td.line(pts, fill=col + (150,), width=int(2.2 * SS))
    threads = threads.filter(ImageFilter.GaussianBlur(1.4 * SS))
    layer.alpha_composite(threads)
    return layer


def paint_wave(base):
    mask = wave_mask()
    body = wave_body()

    # luminous curtains clipped to the silhouette
    curtains = aurora_curtains()
    body_rgba = body.convert("RGBA")
    body_rgba.alpha_composite(curtains)
    # extra luminosity: a blurred screen-blend of the curtains
    glowc = curtains.convert("RGB").filter(ImageFilter.GaussianBlur(10 * SS))
    body_rgba = ImageChops.screen(body_rgba.convert("RGB"),
                                  glowc.point(lambda v: int(v * 0.5))).convert("RGBA")

    # bright flow streaks sweeping down the wave face (clipped to the mask)
    flow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fld = ImageDraw.Draw(flow)
    for i in range(15):
        y0 = rng.uniform(430, 800)
        x0 = rng.uniform(60, 420) + (y0 - 430) * 0.55
        ln = rng.uniform(150, 330)
        col = aurora_color(rng.random())
        a = int(rng.uniform(45, 105))
        pts = [(x0 + k * ln / 12 + 14 * math.sin(k * 0.9 + i),
                y0 + k * ln * 0.62 / 12) for k in range(13)]
        fld.line([(px * SS, py * SS) for (px, py) in pts],
                 fill=col + (a,), width=int(2.4 * SS))
    flow.putalpha(ImageChops.multiply(flow.getchannel("A"), mask))
    body_rgba.alpha_composite(flow.filter(ImageFilter.GaussianBlur(1.3 * SS)))

    # painterly shading: darken the lower body, lighten under the crest
    shade = Image.new("L", (W, H), 0)
    sh = np.asarray(shade, dtype=np.float32)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    sh[:] = (np.clip((ys - 0.55) / 0.45, 0, 1) * 120).astype(np.float32)
    shade = Image.fromarray(sh.astype(np.uint8))
    dark = Image.new("RGBA", (W, H), (2, 8, 24, 255))
    body_rgba = Image.composite(dark, body_rgba, shade.point(lambda v: v))

    # barrel shadow under the lip
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse(sc([(300, 430), (620, 640)]), fill=(1, 6, 18, 150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(26 * SS))
    body_rgba.alpha_composite(shadow)

    base.paste(body_rgba, (0, 0), mask)

    # glow spilling from the wave into the night
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow.paste(body_rgba, (0, 0), mask)
    base.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18 * SS))
                         .point(lambda v: int(v * 0.55)))

    # bright rim along the crest
    rim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rim)
    crest = sc(catmull_rom(CREST_CTRL, 26))
    rd.line(crest, fill=(190, 255, 225, 190), width=int(9 * SS))
    rd.line(crest, fill=(245, 255, 250, 235), width=int(3.4 * SS))
    # softer rim light tracing the inner face, separating wave from sky
    face = sc(catmull_rom(WAVE_CTRL[8:15], 26))
    rd.line(face, fill=(140, 255, 225, 105), width=int(6 * SS))
    rd.line(face, fill=(215, 255, 240, 140), width=int(2.2 * SS))
    base.alpha_composite(rim.filter(ImageFilter.GaussianBlur(1.1 * SS)))
    return base


def paint_foam_and_spray(base):
    # foam clusters hugging the crest and lip: soft underlay + sharp caps
    foam = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(foam)
    crest = catmull_rom(CREST_CTRL, 26)
    for (x, y) in crest:
        for _ in range(5):
            r = rng.uniform(3, 13)
            ox, oy = rng.uniform(-14, 14), rng.uniform(-10, 10)
            a = int(rng.uniform(80, 200))
            fd.ellipse([(x + ox - r) * SS, (y + oy - r) * SS,
                        (x + ox + r) * SS, (y + oy + r) * SS],
                       fill=(235, 255, 250, a))
    base.alpha_composite(foam.filter(ImageFilter.GaussianBlur(2.2 * SS)))
    fd2 = ImageDraw.Draw(base)
    for (x, y) in crest[::2]:
        for _ in range(2):
            r = rng.uniform(1.6, 5)
            ox, oy = rng.uniform(-11, 11), rng.uniform(-8, 8)
            a = int(rng.uniform(140, 235))
            fd2.ellipse([(x + ox - r) * SS, (y + oy - r) * SS,
                         (x + ox + r) * SS, (y + oy + r) * SS],
                        fill=(245, 255, 252, a))
    # sparse foam wash lapping at the bottom, clustered under the wave
    for _ in range(150):
        x = min(max(rng.gauss(330, 260), 4), S - 4)
        y = rng.uniform(966, 1018)
        r = rng.uniform(1.5, 5)
        a = int(rng.uniform(26, 88) * (y - 950) / 80)
        fd2.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
                    fill=(220, 250, 245, max(a, 0)))

    # spray bursting off the lip, dissolving into stars
    spray = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spd = ImageDraw.Draw(spray)
    origin = (452, 300)
    for _ in range(300):
        ang = rng.uniform(math.pi * 0.95, math.pi * 1.75)
        dist = rng.uniform(8, 250) ** 1.12
        x = origin[0] + math.cos(ang) * dist * 0.95
        y = origin[1] + math.sin(ang) * dist * 0.62
        r = rng.uniform(0.8, 4.6) * max(0.35, 1 - dist / 330)
        a = int(rng.uniform(70, 235) * max(0.15, 1 - dist / 320))
        col = rng.choice([(255, 255, 255), (205, 255, 235), (190, 240, 255)])
        spd.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
                    fill=col + (a,))
    # four-point sparkles among the spray
    for _ in range(46):
        ang = rng.uniform(math.pi * 0.95, math.pi * 1.8)
        dist = rng.uniform(30, 265)
        x = origin[0] + math.cos(ang) * dist * 0.95
        y = origin[1] + math.sin(ang) * dist * 0.62
        r = rng.uniform(4, 11) * (1 - dist / 380)
        if r <= 1:
            continue
        a = int(rng.uniform(120, 255) * (1 - dist / 360))
        col = rng.choice([(255, 255, 255), (170, 255, 215), (200, 225, 255)])
        spd.polygon([(x * SS, (y - r) * SS), ((x + r * 0.22) * SS, (y - r * 0.22) * SS),
                     ((x + r) * SS, y * SS), ((x + r * 0.22) * SS, (y + r * 0.22) * SS),
                     (x * SS, (y + r) * SS), ((x - r * 0.22) * SS, (y + r * 0.22) * SS),
                     ((x - r) * SS, y * SS), ((x - r * 0.22) * SS, (y - r * 0.22) * SS)],
                    fill=col + (max(a, 0),))
    base.alpha_composite(spray.filter(ImageFilter.GaussianBlur(0.5 * SS)))
    return base


# ---------------------------------------------------------------- penguin

def paint_penguin(base):
    """A joyful penguin riding the wave face on a coral surfboard.

    Drawn upright on its own tile (so ellipses stay true), then the tile
    is rotated to lean with the wave face and pasted onto the scene.
    """
    T = 620                                # tile size in 1024-space units
    tile = Image.new("RGBA", (T * SS, T * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    OX, OY = T / 2, T / 2 + 10             # body centre inside the tile

    def P(x, y):
        return ((OX + x) * SS, (OY + y) * SS)

    def poly(pts, **kw):
        d.polygon([P(x, y) for (x, y) in pts], **kw)

    def ell(cx0, cy0, rx, ry, **kw):
        d.ellipse([(OX + cx0 - rx) * SS, (OY + cy0 - ry) * SS,
                   (OX + cx0 + rx) * SS, (OY + cy0 + ry) * SS], **kw)

    ink = C("#12141f") + (255,)
    belly_c = C("#f5f9ff") + (255,)
    orange = C("#ff9f1c") + (255,)
    teal = C("#1de9c3") + (255,)
    red = C("#ff4655") + (255,)

    # surfboard (coral red, sunny stripe), drawn first so feet overlap it
    poly([(-128, 88), (102, 66), (138, 76), (106, 90), (-110, 106), (-136, 96)],
         fill=red)
    poly([(-118, 91), (104, 71), (120, 76), (102, 83), (-104, 99), (-124, 94)],
         fill=C("#ffd23e") + (255,))

    # splash bursting around the board
    for _ in range(60):
        x = rng.uniform(-138, 84)
        y = rng.uniform(84, 122)
        r = rng.uniform(1.6, 6)
        a = int(rng.uniform(90, 225))
        d.ellipse([(OX + x - r) * SS, (OY + y - r) * SS,
                   (OX + x + r) * SS, (OY + y + r) * SS],
                  fill=(235, 255, 250, a))

    # scarf tails streaming behind (to the left), behind the body
    poly([(-24, -26), (-128, -58), (-142, -40), (-100, -30), (-134, -12),
          (-28, -5)], fill=red)
    poly([(-24, -13), (-104, -16), (-112, -3), (-26, 3)],
         fill=C("#e03347") + (255,))

    # flippers raised in joy, teal rim on the outer edge
    left_flip = [(-40, -16), (-84, -50), (-110, -68), (-119, -52),
                 (-98, -14), (-58, 18), (-43, 9)]
    right_flip = [(-x, y) for (x, y) in left_flip]
    poly([(x - 3.5, y - 3.5) for (x, y) in left_flip], fill=teal)
    poly(left_flip, fill=ink)
    poly([(x + 3.5, y - 3.5) for (x, y) in right_flip], fill=teal)
    poly(right_flip, fill=ink)

    # feet planted on the board
    ell(-36, 76, 16, 7.5, fill=orange)
    ell(32, 70, 16, 7.5, fill=orange)

    # body with a teal rim light, then the white belly
    ell(-3, -3, 63, 83, fill=teal)
    ell(0, 0, 61, 81, fill=ink)
    ell(2, 22, 41, 55, fill=belly_c)
    # white face patch, continuous with the belly (cartoon-penguin style),
    # so the dark facial features stay readable
    ell(0, -34, 33, 32, fill=belly_c)

    # happy face: two closed eye arcs drawn as explicit polylines (∩ ∩),
    # a bigger beak, soft cheeks — all on the white face patch
    for ex in (-16, 16):
        arc_pts = []
        for k in range(17):
            th = math.pi * k / 16
            arc_pts.append(P(ex + 10 * math.cos(th), -46 - 8 * math.sin(th)))
        d.line(arc_pts, fill=(16, 18, 30, 255), width=int(5.4 * SS), joint="curve")
    poly([(-13, -34), (13, -34), (0, -17)], fill=orange)
    poly([(-7, -29), (7, -29), (0, -19)], fill=C("#ffb84d") + (255,))
    ell(-27, -22, 7.5, 4.6, fill=(255, 120, 140, 130))
    ell(27, -22, 7.5, 4.6, fill=(255, 120, 140, 130))

    # slim scarf band low across the neck, with a knot
    ell(0, -6, 38, 9, fill=red)
    ell(-36, -4, 7.5, 7.5, fill=C("#e03347") + (255,))

    # soft contact glow on the wave under the board, then paste the tile
    cx, cy = 640, 565
    seat = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sdt = ImageDraw.Draw(seat)
    sdt.ellipse([(cx - 145) * SS, (cy + 82) * SS, (cx + 145) * SS, (cy + 118) * SS],
                fill=(3, 22, 28, 110))
    base.alpha_composite(seat.filter(ImageFilter.GaussianBlur(8 * SS)))

    tile = tile.resize((int(tile.width * 1.18), int(tile.height * 1.18)),
                       Image.BICUBIC)
    tile = tile.rotate(-24, resample=Image.BICUBIC, expand=True)
    base.alpha_composite(tile, dest=(int(cx * SS - tile.width / 2),
                                     int(cy * SS - tile.height / 2)))
    return base


# ----------------------------------------------------------------- finish

def finish(img):
    img = img.convert("RGB")
    # painterly grain
    noise = fractal_noise(W, H, octaves=4, seed=11)
    arr = np.asarray(img).astype(np.float32)
    arr += ((noise - 0.5) * 14.0)[:, :, None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # gentle vignette
    vig = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse([-W * 0.28, -H * 0.28, W * 1.28, H * 1.28], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(160 * SS))
    black = Image.new("RGB", (W, H), (1, 2, 8))
    img = Image.composite(img, ImageChops.multiply(img, black), vig)

    img = ImageEnhance.Color(img).enhance(1.07)
    img = ImageEnhance.Contrast(img).enhance(1.03)
    return img.resize((S, S), Image.LANCZOS)


def main():
    img = make_sky().convert("RGBA")
    img = ambient_aurora(img)
    img = paint_sea(img)
    img = paint_wave(img)
    img = paint_foam_and_spray(img)
    img = paint_penguin(img)
    out = finish(img)
    out.save("entry.png", optimize=True)
    print("saved entry.png", out.size)


if __name__ == "__main__":
    main()
