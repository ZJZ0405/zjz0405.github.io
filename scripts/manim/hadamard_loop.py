"""Hadamard loop: sparse outliers -> dense -> sparse.

Manim Community scene (works with the 3b1b/manim visual language).
Render (Manim Community in Docker):
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/manim -w /manim --entrypoint /opt/venv/bin/manim \
    manimcommunity/manim:latest -qh --fps 30 --format=mp4 \
    --media_dir .manim-render scripts/manim/hadamard_loop.py HadamardLoop

For 3b1b/manim (ManimGL), change the imports to manimlib
(`from manimlib import *`, `from manimlib import Text as MText`);
the scene structure and animations used here are compatible.
"""

import numpy as np
from manim import *
from manim import Text as MText

# ---------------------------------------------------------------- palette
BG = "#0f172a"
FG = "#e5e7eb"
MUTED = "#94a3b8"
ORANGE = "#fb923c"
TEAL = "#2dd4bf"
GREEN = "#34d399"
NEG = "#2563eb"
MID = "#1e293b"
POS = "#f97316"
CMAP = color_gradient([NEG, MID, POS], 256)
SANS = "DejaVu Sans"


def txt(s, **kwargs):
    kwargs.setdefault("font", SANS)
    return MText(s, **kwargs)


# ---------------------------------------------------------------- data
def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]) / np.sqrt(2.0)
    return H


def make_data(n_tokens=32, n_channels=32, seed=7):
    """A matrix with a few outlier channels; H is the Sylvester matrix."""
    rng = np.random.default_rng(seed)
    X = 0.11 * rng.standard_normal((n_tokens, n_channels))

    # a few channels carry the outliers (mixed signs across tokens)
    for j, sign in zip((3, 9, 16, 22, 28), (1.0, -1.0, 1.0, -1.0, 1.0)):
        X[:, j] += sign * rng.uniform(2.6, 4.8, n_tokens)

    # isolated spikes make the transformed matrix look genuinely dense
    for _ in range(18):
        i = int(rng.integers(n_tokens))
        j = int(rng.integers(n_channels))
        X[i, j] += rng.choice([-1.0, 1.0]) * rng.uniform(3.5, 7.0)

    H = hadamard(n_channels)
    Y = X @ H
    Z = Y @ H
    assert np.allclose(Z, X), "H is an involution: H @ H must be I"
    return X, Y, Z


# ---------------------------------------------------------------- mobjects
class Heatmap(VGroup):
    """A crisp matrix heatmap made of squares."""

    def __init__(self, data, cell=0.14, **kwargs):
        super().__init__(**kwargs)
        self.nrows, self.ncols = data.shape
        self.cell = cell
        self._cells = []
        for i in range(self.nrows):
            for j in range(self.ncols):
                sq = Square(
                    side_length=cell,
                    stroke_width=0.35,
                    stroke_color=BG,
                    fill_opacity=1,
                )
                sq.move_to(
                    [
                        (j - (self.ncols - 1) / 2) * cell,
                        ((self.nrows - 1) / 2 - i) * cell,
                        0,
                    ]
                )
                self.add(sq)
                self._cells.append((sq, i, j))
        self.set_data(data)

    def set_data(self, data, vmax=1.0):
        vmax = max(float(vmax), 1e-9)
        for sq, i, j in self._cells:
            v = float(np.clip(data[i, j] / vmax, -1.0, 1.0))
            sq.set_fill(CMAP[int((v + 1.0) * 0.5 * 255)], opacity=1)


def make_h_icon(cell=0.24):
    H4 = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
    )
    cells = VGroup()
    for i in range(4):
        for j in range(4):
            v = H4[i, j]
            sq = Square(
                side_length=cell,
                stroke_width=0.5,
                stroke_color="#334155",
                fill_color="#e2e8f0" if v == 1 else "#0b1220",
                fill_opacity=1,
            )
            lab = txt(
                "+" if v == 1 else "−",
                font_size=16,
                color="#0f172a" if v == 1 else MUTED,
            )
            lab.move_to(sq.get_center())
            cells.add(VGroup(sq, lab))
    cells.arrange_in_grid(rows=4, cols=4, buff=0)
    border = SurroundingRectangle(cells, color=MUTED, buff=0.06, stroke_width=1.0)
    label = txt("H", font_size=30, color=FG)
    label.next_to(border, DOWN, buff=0.10)
    sub = txt("d × d", font_size=13, color=MUTED)
    sub.next_to(label, DOWN, buff=0.00)
    return VGroup(cells, border, label, sub)


def make_legend():
    def swatch(color):
        return Square(
            side_length=0.13,
            stroke_width=0.5,
            stroke_color="#334155",
            fill_color=color,
            fill_opacity=1,
        )

    items = VGroup(
        swatch(NEG), txt("−", font_size=18, color=MUTED),
        swatch(MID), txt("0", font_size=18, color=MUTED),
        swatch(POS), txt("+", font_size=18, color=MUTED),
    )
    items.arrange(RIGHT, buff=0.10)
    note = txt("normalized per frame", font_size=15, color=MUTED)
    return VGroup(items, note).arrange(RIGHT, buff=0.25)


class HadamardLoop(Scene):
    def construct(self):
        self.camera.background_color = BG

        X, Y, Z = make_data()
        Xn = X / np.abs(X).max()
        Yn = Y / np.abs(Y).max()
        Zn = Z / np.abs(Z).max()
        assert np.allclose(Xn, Zn)

        # ---- static layout
        title = txt(
            "Hadamard transform: sparse → dense → sparse",
            font_size=38,
            color=FG,
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.38)
        subtitle = txt(
            "the same matrix H, applied twice:  H² = I",
            font_size=23,
            color=MUTED,
        )
        subtitle.next_to(title, DOWN, buff=0.12)

        grid = Heatmap(Xn)
        grid.move_to(np.array([-2.9, 0.0, 0.0]))
        border = SurroundingRectangle(
            grid, color="#334155", buff=0.06, stroke_width=1.2
        )

        xlab = txt("channels  →", font_size=17, color=MUTED)
        xlab.next_to(grid, DOWN, buff=0.14)
        ylab = txt("tokens", font_size=17, color=MUTED).rotate(PI / 2)
        ylab.next_to(grid, LEFT, buff=0.16)
        legend = make_legend()
        legend.next_to(xlab, DOWN, buff=0.12)

        h_icon = make_h_icon()
        h_icon.move_to(np.array([0.85, 0.85, 0.0]))
        arrow = Arrow(
            grid.get_right() + RIGHT * 0.10,
            h_icon.get_left() + LEFT * 0.15,
            buff=0.0,
            color=TEAL,
            stroke_width=3,
        )
        xH = txt("×H", font_size=23, color=TEAL).next_to(arrow, UP, buff=0.06)

        # ---- formula with a moving highlight
        t1 = txt("X", font_size=31, color=ORANGE)
        a1 = txt("→", font_size=31, color=MUTED)
        t2 = txt("XH", font_size=31, color=TEAL)
        a2 = txt("→", font_size=31, color=MUTED)
        t3 = txt("XH² = X", font_size=31, color=GREEN)
        formula = VGroup(t1, a1, t2, a2, t3).arrange(RIGHT, buff=0.18)
        formula.move_to(np.array([3.75, 0.55, 0.0]))
        hl = Underline(t1, color=ORANGE, buff=0.08, stroke_width=3)

        # ---- step labels (cross-faded)
        step1 = txt("1.  X: sparse outliers", font_size=23, color=ORANGE)
        step2 = txt("2.  XH: all channels mixed", font_size=23, color=TEAL)
        step3 = txt("3.  XH² = X: exact recovery", font_size=23, color=GREEN)
        for s in (step1, step2, step3):
            s.move_to(np.array([1.75, 1.75, 0.0]), aligned_edge=LEFT)
        step2.set_opacity(0)
        step3.set_opacity(0)

        note1 = txt(
            "H² = I: the same H twice returns X", font_size=21, color=FG
        )
        note2 = txt(
            "a random H' = DH does not: H'² ≠ I", font_size=18, color=ORANGE
        )
        note1.move_to(np.array([1.75, -0.55, 0.0]), aligned_edge=LEFT)
        note2.next_to(note1, DOWN, buff=0.12, aligned_edge=LEFT)

        check = txt("✓", font_size=54, color=GREEN)
        check.next_to(t3, RIGHT, buff=0.25)

        same = txt("same H", font_size=18, color=GREEN)
        same.next_to(h_icon, UP, buff=0.12)

        # ---- initial frame
        self.add(
            title, subtitle, grid, border, xlab, ylab, legend,
            h_icon, arrow, xH, formula, hl, step1, step2, step3,
        )
        self.wait(1.0)

        # ---- first transform: X -> XH
        tr1 = ValueTracker(0.0)

        def update1(_):
            a = tr1.get_value()
            grid.set_data((1 - a) * Xn + a * Yn)

        self.play(h_icon.animate.scale(1.08), rate_func=there_and_back, run_time=0.5)
        grid.add_updater(update1)
        self.play(tr1.animate.set_value(0.5), run_time=1.2, rate_func=linear)
        self.play(
            step1.animate.set_opacity(0),
            step2.animate.set_opacity(1),
            Transform(
                hl,
                Underline(t2, color=TEAL, buff=0.08, stroke_width=3),
            ),
            run_time=0.4,
        )
        self.play(tr1.animate.set_value(1.0), run_time=1.2, rate_func=linear)
        grid.remove_updater(update1)
        self.wait(0.8)

        # ---- second transform: XH -> XH² = X
        tr2 = ValueTracker(0.0)

        def update2(_):
            a = tr2.get_value()
            grid.set_data((1 - a) * Yn + a * Zn)

        self.play(h_icon.animate.scale(1.08), rate_func=there_and_back, run_time=0.5)
        grid.add_updater(update2)
        self.play(tr2.animate.set_value(0.5), run_time=1.2, rate_func=linear)
        self.play(
            step2.animate.set_opacity(0),
            step3.animate.set_opacity(1),
            Transform(
                hl,
                Underline(t3, color=GREEN, buff=0.08, stroke_width=3),
            ),
            FadeIn(same, shift=LEFT * 0.08),
            run_time=0.4,
        )
        self.play(tr2.animate.set_value(1.0), run_time=1.2, rate_func=linear)
        grid.remove_updater(update2)

        # ---- success frame
        self.play(FadeIn(check, scale=1.3), run_time=0.45)
        self.play(
            FadeIn(note1, shift=UP * 0.08),
            FadeIn(note2, shift=UP * 0.08),
            run_time=0.7,
        )
        self.wait(2.2)

        # ---- restore the initial state so the video can loop
        self.play(
            FadeOut(check),
            FadeOut(note1),
            FadeOut(note2),
            FadeOut(same),
            step3.animate.set_opacity(0),
            step1.animate.set_opacity(1),
            Transform(
                hl,
                Underline(t1, color=ORANGE, buff=0.08, stroke_width=3),
            ),
            run_time=0.8,
        )
        self.wait(1.0)
