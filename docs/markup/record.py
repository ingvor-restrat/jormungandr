#!/usr/bin/env python3
"""Render the real OU spread trainer frames into the committed GIF."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("ou-spread.gif")
COMMAND = [
    sys.executable,
    "examples/train_ou_spread.py",
    "--record",
    "--actors",
    "2",
    "--train-episodes",
    "28",
    "--validation-episodes",
    "7",
    "--horizon",
    "48",
    "--minimum-updates",
    "60",
    "--interval-ms",
    "0",
    "--actor-pause-ms",
    "25",
    "--seed",
    "337911",
]
ANSI = re.compile(r"\x1b\[([0-9;]*)m")
REGULAR_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
BACKGROUND = "#07111d"
PANEL = "#0b1725"
CHROME = "#111f30"
DEFAULT = "#d8e2ec"


def xterm(index: int) -> tuple[int, int, int]:
    basic = [
        (0, 0, 0),
        (205, 49, 49),
        (13, 188, 121),
        (229, 229, 16),
        (36, 114, 200),
        (188, 63, 188),
        (17, 168, 205),
        (229, 229, 229),
        (102, 102, 102),
        (241, 76, 76),
        (35, 209, 139),
        (245, 245, 67),
        (59, 142, 234),
        (214, 112, 214),
        (41, 184, 219),
        (255, 255, 255),
    ]
    if index < 16:
        return basic[index]
    if index < 232:
        index -= 16
        red, green, blue = index // 36, index // 6 % 6, index % 6

        def channel(value: int) -> int:
            return 0 if value == 0 else 55 + value * 40

        return channel(red), channel(green), channel(blue)
    shade = 8 + (index - 232) * 10
    return shade, shade, shade


def spans(line: str):
    position = 0
    color: tuple[int, int, int] | str = DEFAULT
    bold = False
    dim = False
    for match in ANSI.finditer(line):
        if match.start() > position:
            yield line[position : match.start()], color, bold, dim
        codes = [int(code) for code in match.group(1).split(";") if code] or [0]
        cursor = 0
        while cursor < len(codes):
            code = codes[cursor]
            if code == 0:
                color, bold, dim = DEFAULT, False, False
            elif code == 1:
                bold = True
            elif code == 2:
                dim = True
            elif code == 22:
                bold, dim = False, False
            elif 30 <= code <= 37:
                color = xterm(code - 30)
            elif 90 <= code <= 97:
                color = xterm(code - 90 + 8)
            elif code == 38 and cursor + 2 < len(codes) and codes[cursor + 1] == 5:
                color = xterm(codes[cursor + 2])
                cursor += 2
            elif code == 39:
                color = DEFAULT
            cursor += 1
        position = match.end()
    if position < len(line):
        yield line[position:], color, bold, dim


def render(
    frame: str,
    regular,
    bold_font,
    *,
    columns: int,
    rows: int,
) -> Image.Image:
    clean_lines = frame.strip("\n").splitlines()
    probe = Image.new("RGB", (1, 1))
    char_width = ImageDraw.Draw(probe).textlength("M", font=regular)
    line_height = 23
    left, top, chrome = 30, 47, 30
    width = int(left * 2 + columns * char_width)
    height = top + rows * line_height + 27
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, width - 8, height - 8), 14, fill=PANEL)
    draw.rounded_rectangle((8, 8, width - 8, 8 + chrome), 14, fill=CHROME)
    draw.rectangle((8, 8 + chrome - 12, width - 8, 8 + chrome), fill=CHROME)
    for offset, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        x = 27 + offset * 20
        draw.ellipse((x, 18, x + 10, 28), fill=color)
    draw.text(
        (width - 205, 15),
        "jormungandr · synthetic",
        font=regular,
        fill="#71849a",
    )
    for row, line in enumerate(clean_lines):
        x = left
        y = top + row * line_height
        for text, color, is_bold, is_dim in spans(line):
            chosen = bold_font if is_bold else regular
            if is_dim and isinstance(color, tuple):
                color = tuple(int(channel * 0.68) for channel in color)
            draw.text((x, y), text, font=chosen, fill=color)
            x += char_width * len(text)
    return image


def main() -> None:
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not current_pythonpath
        else source_path + os.pathsep + current_pythonpath
    )
    process = subprocess.run(
        COMMAND,
        cwd=ROOT,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frames = [
        frame.strip("\n")
        for frame in process.stdout.decode("utf-8").split("\x0c")
        if frame.strip()
    ]
    if not frames:
        raise RuntimeError("OU spread trainer produced no recording frames")
    regular = ImageFont.truetype(REGULAR_FONT, 15)
    bold_font = ImageFont.truetype(BOLD_FONT, 15)
    columns = max(
        len(ANSI.sub("", line))
        for frame in frames
        for line in frame.strip("\n").splitlines()
    )
    rows = max(len(frame.strip("\n").splitlines()) for frame in frames)
    images = [
        render(
            frame,
            regular,
            bold_font,
            columns=columns,
            rows=rows,
        )
        for frame in frames
    ]
    images.extend([images[-1]] * 8)
    images[0].save(
        OUTPUT,
        save_all=True,
        append_images=images[1:],
        duration=140,
        loop=0,
        disposal=1,
        optimize=True,
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(frames)} source frames)")


if __name__ == "__main__":
    main()
