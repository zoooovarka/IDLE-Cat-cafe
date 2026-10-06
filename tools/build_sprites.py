#!/usr/bin/env python3
"""
Сборка спрайтов для игры: исходники (sprites/*.png, тяжёлые) -> img/*.webp (лёгкие).

Что делает:
  * обрезает прозрачные поля, чтобы спрайты одинаково заполняли место в игре;
  * уменьшает до нужного размера (с запасом под экраны с высокой плотностью пикселей);
  * сохраняет в WebP с прозрачностью;
  * из иконки «Улучшение кофемашина» делает чистую кофемашину без стрелки и искр
    (главный объект для тапа).

Запуск:  pip install pillow numpy opencv-python-headless
         python3 tools/build_sprites.py
Чтобы добавить новый спрайт: положите PNG в sprites/ и допишите строку в SPRITES.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'sprites')
DST = os.path.join(ROOT, 'img')

# имя в игре: (исходный файл, максимальная сторона в px, режим)
#   'cat'  — обрезка, квадрат, персонаж стоит на нижней кромке
#   'icon' — обрезка, квадрат, по центру
#   'bg'   — фон без прозрачности
SPRITES = {
    'cat-barista':   ('Милый кот-бариста с чашкой кофе.png', 256, 'cat'),
    'cat-baker':     ('Милая кошка-пекарь с подносом булочек.png', 256, 'cat'),
    'cat-waiter':    ('Весёлый кот-официант на роликах.png', 256, 'cat'),
    'cat-icecream':  ('Чиби-котик — продавец мороженого.png', 256, 'cat'),
    'cat-pastry':    ('Котёнок-кондитер с клубничным тортом.png', 256, 'cat'),
    'cat-sushi':     ('Кот-повар готовит суши с ножом.png', 256, 'cat'),
    'cat-dj':        ('Кот-диджей с неоновым микшером.png', 256, 'cat'),
    'cat-chef':      ('Весёлый кот-повар с супом.png', 256, 'cat'),
    'guest-fashion': ('Кокетливая кошечка в розовом наряде.png', 256, 'cat'),
    'guest-panda':   ('Милый панда-путешественник с камерой.png', 256, 'cat'),
    'guest-cool':    ('Крутой кот в золотых аксессуарах.png', 256, 'cat'),
    'up-machine':    ('Улучшение кофемашина.png', 128, 'icon'),
    'up-interior':   ('Улучшение интерьер.png', 128, 'icon'),
    'up-cake':       ('Улучшение тортик.png', 128, 'icon'),
    'up-showcase':   ('улучшение винтрина.png', 128, 'icon'),
    'up-plant':      ('улучшение кустик.png', 128, 'icon'),
    'coin':          ('Монетка.png', 96, 'icon'),
    'star':          ('Звездочка.png', 96, 'icon'),
    'bg-cafe-1':     ('Уютное кафе в солнечном свете.png', 1080, 'bg'),
    'bg-cafe-2':     ('Уютное солнечное кафе с растениями.png', 1080, 'bg'),
}
MACHINE_SRC = 'Улучшение кофемашина.png'


def trim(im):
    box = im.getchannel('A').point(lambda a: 255 if a > 8 else 0).getbbox()
    return im.crop(box) if box else im


def square(im, size, bottom):
    im = trim(im)
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    x = (size - im.width) // 2
    y = size - im.height if bottom else (size - im.height) // 2
    canvas.alpha_composite(im, (x, y))
    return canvas


def clean_machine(im):
    """Убирает стрелку и искры: продолжает панель машины, отражает левый край, сглаживает швы."""
    import cv2
    a = np.array(im.convert('RGBA')).astype(int)
    H, W = a.shape[:2]
    R, G, B, A = [a[..., k] for k in range(4)]
    vis = A > 20
    X = np.arange(W)[None, :].repeat(H, 0)
    Y = np.arange(H)[:, None].repeat(W, 1)

    def dil(m, s):
        return np.array(Image.fromarray((m * 255).astype('uint8')).filter(ImageFilter.MaxFilter(s))) > 0

    def ero(m, s):
        return np.array(Image.fromarray((m * 255).astype('uint8')).filter(ImageFilter.MinFilter(s))) > 0

    arrow = dil((G > R + 25) & (G > B + 25) & vis, 17) & (Y < 118)
    spark = dil((R > 200) & (G > 190) & (B < 170) & vis & (X > 250), 13)
    out = a.copy()
    LB, RE = 44, 267  # левый край корпуса и правый край боковой панели (в пикселях исходника)
    for y in range(0, 118):
        xs = np.where(arrow[y, 150:RE + 1])[0]
        if not len(xs):
            continue
        x0 = 150 + xs[0]
        ref = a[y, x0 - 1]
        band = 36 if y < 52 else 17
        for x in range(x0, RE + 1):
            if y < 108:
                out[y, x] = ref if x <= RE - band else a[y, LB + (RE - x)]
            elif arrow[y, x]:
                out[y, x] = a[118, x] if x >= 248 else ref
    for x in range(240, RE + 1):
        for y in range(110, H):
            if spark[y, x]:
                out[y, x] = out[y - 1, x]
    for y in range(0, 196):
        for x in range(RE + 1, W):
            mx = int(round(2 * 161.5 - x))
            out[y, x] = 0 if y < 108 else (a[y, mx] if mx >= 0 else 0)
    seam = dil(arrow, 5) & ~ero(arrow, 5) & (Y > 36) & (Y < 106) & (X < RE - 17)
    rgb = out[..., :3].astype('uint8')[:, :, ::-1].copy()
    rgb = cv2.inpaint(rgb, (seam * 255).astype('uint8'), 3, cv2.INPAINT_TELEA)
    out[..., :3] = rgb[:, :, ::-1]
    low = (spark & (X > RE) & (Y >= 196)).astype('uint8') * 255
    rgb = out[..., :3].astype('uint8')[:, :, ::-1].copy()
    rgb = cv2.inpaint(rgb, low, 4, cv2.INPAINT_TELEA)
    out[..., :3] = rgb[:, :, ::-1]
    out[..., 3] = cv2.inpaint(out[..., 3].astype('uint8'), low, 4, cv2.INPAINT_TELEA)
    return Image.fromarray(out.astype('uint8'), 'RGBA')


def save(im, name, quality):
    path = os.path.join(DST, name + '.webp')
    im.save(path, 'WEBP', quality=quality, method=6)
    return os.path.getsize(path)


def main():
    os.makedirs(DST, exist_ok=True)
    total_src = total_dst = 0
    for name, (fname, size, mode) in SPRITES.items():
        path = os.path.join(SRC, fname)
        total_src += os.path.getsize(path)
        im = Image.open(path)
        if mode == 'bg':
            im = im.convert('RGB')
            im.thumbnail((size, size), Image.LANCZOS)
            n = save(im, name, 72)
        else:
            im = square(im.convert('RGBA'), size, bottom=(mode == 'cat'))
            n = save(im, name, 82)
        total_dst += n
        print(f'{name:14s} {im.width}x{im.height}  {n / 1024:6.1f} KB')

    machine = square(clean_machine(Image.open(os.path.join(SRC, MACHINE_SRC))), 300, bottom=True)
    n = save(machine, 'machine', 85)
    total_dst += n
    print(f'{"machine":14s} {machine.width}x{machine.height}  {n / 1024:6.1f} KB')
    print(f'\nИсходники: {total_src / 1048576:.1f} MB -> игра: {total_dst / 1024:.0f} KB')


if __name__ == '__main__':
    sys.exit(main())
