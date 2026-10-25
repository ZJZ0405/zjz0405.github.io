"""3D Hadamard transform: the data, stage by stage.

Manim Community scene. The five FWHT stages are real partial transforms:
each stage pairs channels at distance h = 1, 2, 4, 8, 16 and applies
(u+v, u-v)/sqrt(2). The same stages in reverse order give H^{-1} = H.

Render with the official Manim Docker image:
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/manim -w /manim --entrypoint /opt/venv/bin/manim \
    manimcommunity/manim:latest -qh --fps 30 --format=mp4 \
    --media_dir .manim-render3d scripts/manim/hadamard_3d.py Hadamard3D
"""

import numpy as np
from manim import *
from manim import Text as MText

BG = "#0f172a"
FG = "#e5e7eb"
MUTED = "#94a3b8"
TEAL = "#2dd4bf"
ORANGE = "#fb923c"
GREEN = "#34d399"
NEG = "#3b82f6"
MID = "#0e7490"
POS = "#fb923c"


def txt(s, **kwargs):
    kwargs.setdefault("font", "DejaVu Sans")
    return MText(s, **kwargs)


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]) / np.sqrt(2.0)
    return H


def fwht_stages(X):
    """[X, S1 X, S2 S1 X, ..., H X] for the FWHT stages."""
    A = X.copy()
    stages = [A.copy()]
    d = A.shape[1]
    h = 1
    while h < d:
        B = A.copy()
        for start in range(0, d, 2 * h):
            for k in range(h):
                u = A[:, start + k].copy()
                v = A[:, start + k + h].copy()
                B[:, start + k] = (u + v) / np.sqrt(2.0)
                B[:, start + k + h] = (u - v) / np.sqrt(2.0)
        A = B
        stages.append(A.copy())
        h *= 2
    return stages


def make_data(n=32, seed=7):
    rng = np.random.default_rng(seed)
    X = 0.11 * rng.standard_normal((n, n))
    for j, sign in zip((5, 20), (1.0, -1.0)):
        X[:, j] += sign * rng.uniform(6.0, 8.0, n)
    for _ in range(8):
        i = int(rng.integers(n))
        j = int(rng.integers(n))
        X[i, j] += rng.choice([-1.0, 1.0]) * rng.uniform(4.0, 6.0)
    return X


class Hadamard3D(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        n = 32
        X = make_data(n)
        H = hadamard(n)
        stages = fwht_stages(X)
        assert np.allclose(stages[-1], X @ H)

        # A fixed 3/4 view: depth comes from perspective, not a spinning camera.
        self.set_camera_orientation(phi=55 * DEGREES, theta=-50 * DEGREES, zoom=1.05)

        span = 4.4
        zscale = 0.26
        zmax = max(np.abs(A).max() for A in stages) * zscale
        axes = ThreeDAxes(
            x_range=[-span, span, 1],
            y_range=[-span, span, 1],
            z_range=[-zmax, zmax, 1],
            x_length=2 * span,
            y_length=2 * span,
            z_length=2 * zmax,
            axis_config={"include_ticks": False, "color": MUTED, "stroke_width": 0},
        )

        def sample(A, u, v):
            """Bilinear sample of the matrix; keeps the surface smooth."""
            i = v * (n - 1)
            j = u * (n - 1)
            i0 = min(int(np.floor(i)), n - 1)
            j0 = min(int(np.floor(j)), n - 1)
            i1 = min(i0 + 1, n - 1)
            j1 = min(j0 + 1, n - 1)
            di = i - i0
            dj = j - j0
            return (
                A[i0, j0] * (1 - di) * (1 - dj)
                + A[i0, j1] * (1 - di) * dj
                + A[i1, j0] * di * (1 - dj)
                + A[i1, j1] * di * dj
            )

        def make_surface(A):
            def f(u, v):
                z = sample(A, u, v)
                return axes.c2p(
                    (u - 0.5) * span,
                    (0.5 - v) * span,
                    z * zscale,
                )

            surf = Surface(
                f,
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(40, 40),
                checkerboard_colors=False,
                fill_opacity=0.90,
                stroke_color="#0e7490",
                stroke_width=0.08,
            )
            surf.set_fill_by_value(
                axes=axes,
                colorscale=[(NEG, -zmax), (MID, 0.0), (POS, zmax)],
                axis=2,
            )
            return surf

        surfaces = [make_surface(A) for A in stages]
        surface = surfaces[0]

        floor = Square(
            side_length=span * 1.02,
            stroke_width=0,
            fill_color="#0b1220",
            fill_opacity=0.45,
        )
        floor.move_to([0, 0, -zmax * 1.2])

        # thin floor grid: enough to anchor the surface in 3D
        grid = VGroup()
        for t in np.linspace(-span / 2, span / 2, 5):
            grid.add(
                Line3D(
                    [t, -span / 2, -zmax * 1.18],
                    [t, span / 2, -zmax * 1.18],
                    color=MUTED,
                    stroke_width=0.8,
                    stroke_opacity=0.10,
                )
            )
            grid.add(
                Line3D(
                    [-span / 2, t, -zmax * 1.18],
                    [span / 2, t, -zmax * 1.18],
                    color=MUTED,
                    stroke_width=0.8,
                    stroke_opacity=0.10,
                )
            )

        self.add(floor, grid, surface)

        # ---------------- fixed overlay
        title = txt(
            "Hadamard transform: the data, stage by stage",
            font_size=31,
            color=FG,
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.24)
        subtitle = txt(
            "five butterfly stages forward, then the same H in reverse",
            font_size=17,
            color=MUTED,
        )
        subtitle.next_to(title, DOWN, buff=0.07)

        formula = txt("X  →  XH  →  XH² = X", font_size=19, color=FG)
        formula.to_edge(DOWN, buff=0.92)

        hint = txt("height = value  ·  fixed scale", font_size=13, color=MUTED)
        hint.to_corner(DR, buff=0.28)

        stage_labels = []
        for k in range(5):
            stage_labels.append(
                txt(f"stage {k+1}/5   ·   h = {2**k}", font_size=18, color=TEAL)
            )
        for k in range(5):
            stage_labels.append(
                txt(f"same H  ·  reverse   ·   h = {2**k}", font_size=18, color=ORANGE)
            )
        for lab in stage_labels:
            lab.to_corner(DL, buff=0.28)
            lab.shift(UP * 0.36)
            lab.set_opacity(0)
        stage_labels[0].set_opacity(1)
        current_label = [0]

        segs = VGroup()
        for _ in range(5):
            seg = Rectangle(
                width=0.48,
                height=0.10,
                stroke_width=0.8,
                stroke_color=MUTED,
                fill_color=BG,
                fill_opacity=0.25,
            )
            segs.add(seg)
        segs.arrange(RIGHT, buff=0.06)
        segs.next_to(formula, DOWN, buff=0.12)

        note = txt("H² = I: the same H twice returns X", font_size=17, color=GREEN)
        note.next_to(formula, UP, buff=0.12)
        check = txt("✓", font_size=36, color=GREEN)
        check.next_to(note, RIGHT, buff=0.10)
        note.set_opacity(0)
        check.set_opacity(0)

        self.add_fixed_in_frame_mobjects(
            title, subtitle, formula, hint, *stage_labels, segs, note, check
        )
        self.wait(1.0)

        def switch_label(idx):
            if idx == current_label[0]:
                return
            self.play(
                stage_labels[current_label[0]].animate.set_opacity(0),
                stage_labels[idx].animate.set_opacity(1),
                run_time=0.18,
            )
            current_label[0] = idx

        def paint_progress(k, backward=False):
            for i, seg in enumerate(segs):
                if backward:
                    active = i >= k
                    col = ORANGE if active else BG
                else:
                    active = i <= k
                    col = TEAL if active else BG
                seg.set_fill(col, opacity=0.8 if active else 0.2)

        # forward: 5 real FWHT stages
        for k in range(1, 6):
            switch_label(k - 1)
            paint_progress(k - 1, backward=False)
            self.play(Transform(surface, surfaces[k]), run_time=0.72, rate_func=smooth)
            self.wait(0.06)

        self.wait(0.7)
        self.play(
            note.animate.set_opacity(1),
            check.animate.set_opacity(1),
            run_time=0.5,
        )
        self.wait(1.3)
        self.play(
            note.animate.set_opacity(0),
            check.animate.set_opacity(0),
            run_time=0.35,
        )

        # backward: same H, stages reversed
        for k in range(4, -1, -1):
            switch_label(5 + k)
            paint_progress(k, backward=True)
            self.play(Transform(surface, surfaces[k]), run_time=0.62, rate_func=smooth)
            self.wait(0.05)

        # reset for a seamless loop
        for i, lab in enumerate(stage_labels):
            lab.set_opacity(1 if i == 0 else 0)
        current_label[0] = 0
        for seg in segs:
            seg.set_fill(BG, opacity=0.2)
        self.wait(0.9)
