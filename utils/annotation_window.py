"""Fixed-image annotation window; scrolling is delivered only to the editor."""
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import os
from textwrap import wrap

import cv2
import numpy as np

FONT_SIZE_PX = 22
LINE_HEIGHT_PX = 30
TEXT_PADDING_PX = 10
STATUS_HEIGHT_PX = 4*LINE_HEIGHT_PX+2*TEXT_PADDING_PX


@dataclass(frozen=True)
class AnnotationPanel:
    """Direct-quantized linear RGB as display-only BGR8, plus separate captions."""

    image: np.ndarray
    captions: tuple[str, ...] = ()
    header: bool = False


@dataclass(frozen=True)
class AnnotationLabel:
    text: str
    position: tuple[int, int]
    color: str = 'white'


@dataclass(frozen=True)
class AnnotationDisplay:
    image: np.ndarray
    scales: tuple[float, ...]
    origins: tuple[tuple[int, int], ...]
    labels: tuple[AnnotationLabel, ...]


@lru_cache(maxsize=1)
def _font_metrics():
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    return RendererAgg(1, 1, 72), FontProperties(family='DejaVu Sans', size=FONT_SIZE_PX)


@lru_cache(maxsize=512)
def _wrap_caption(text: str, width: int) -> tuple[str, ...]:
    renderer, font = _font_metrics()
    # Measure real glyph widths rather than assuming that every character is equally wide.
    columns = max(1, len(text))
    while True:
        lines = wrap(text, width=columns) or ['']
        if columns == 1 or all(renderer.get_text_width_height_descent(line, font, False)[0] <= width
                               for line in lines):
            return tuple(lines)
        columns -= 1


def compose_annotation_panels(
    panels: tuple[AnnotationPanel, ...], widths: tuple[int, ...], height: int,
    status: str = '',
) -> AnnotationDisplay:
    """Lay out images and text separately, retaining source-pixel transforms.

    Captions use final canvas coordinates; they are never resized with source images.
    Header panels center the image beneath their captions. Other images align top left.
    """
    if len(panels) != len(widths) or height <= 0 or any(width <= 0 for width in widths):
        raise ValueError('Panel widths must match the images and all dimensions must be positive')
    canvas = np.zeros((height + STATUS_HEIGHT_PX, sum(widths), 3), np.uint8)
    left, scales, origins, labels = 0, [], [], []
    for panel, width in zip(panels, widths, strict=True):
        image = panel.image
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Annotation panels must contain display-only BGR8 images')
        h, w = image.shape[:2]
        lines = [(line, 'orangered' if caption.startswith('REJECTED') else 'white')
                 for caption in panel.captions
                 for line in _wrap_caption(caption, width-2*TEXT_PADDING_PX)]
        for i, (line, color) in enumerate(lines):
            labels.append(AnnotationLabel(line, (left+TEXT_PADDING_PX, TEXT_PADDING_PX+i*LINE_HEIGHT_PX), color))
        top = 2*TEXT_PADDING_PX+len(lines)*LINE_HEIGHT_PX if panel.header else 0
        if top >= height:
            raise ValueError('Panel captions leave no space for the image')
        scale = min(width/w, (height-top)/h)
        resized = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
        x = left+(width-resized.shape[1])//2 if panel.header else left
        canvas[top:top+resized.shape[0], x:x+resized.shape[1]] = resized
        scales.append(scale)
        origins.append((x, top))
        left += width
    lines = [line for paragraph in status.splitlines()
             for line in _wrap_caption(paragraph, sum(widths)-2*TEXT_PADDING_PX)]
    for i, line in enumerate(lines[:4]):
        labels.append(AnnotationLabel(line, (TEXT_PADDING_PX, height+TEXT_PADDING_PX+i*LINE_HEIGHT_PX)))
    return AnnotationDisplay(canvas, tuple(scales), tuple(origins), tuple(labels))


def _draw_labels(axes, labels):
    import matplotlib.patheffects as path_effects

    axes.apply_aspect()
    scale = axes.transData.get_matrix()[0, 0]
    return [axes.text(*label.position, label.text, color=label.color,
                     fontsize=FONT_SIZE_PX*scale*72/axes.figure.dpi,
                     family='DejaVu Sans', va='top', clip_on=True, parse_math=False,
                     path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
            for label in labels]


def _create_figure_manager():
    from PyQt5.QtCore import QLibraryInfo
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, FigureManagerQT

    # Importing cv2 points Qt at OpenCV's private plugins/fonts. This window uses
    # PyQt's Qt runtime, so its application must load PyQt's platform plugins.
    if 'cv2' in os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', ''):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    if 'cv2' in os.environ.get('QT_QPA_FONTDIR', ''):
        os.environ.pop('QT_QPA_FONTDIR')
    with rc_context({'toolbar': 'None'}):
        figure = Figure(figsize=(16, 5), facecolor='black')
        return FigureManagerQT(FigureCanvasQTAgg(figure), num=1)


class AnnotationWindow:
    """Display BGR8 canvases and return mouse coordinates in canvas pixels.

    The axes retain fixed image limits during scrolling and window resizing.
    Mouse events use OpenCV event codes so the geometry editor stays independent
    of the GUI backend. Closing the window is reported through ``is_closed``.
    """

    def __init__(self, title: str):
        self._manager = _create_figure_manager()
        self._manager.set_window_title(title)
        self._canvas = self._manager.canvas
        self._canvas.mpl_disconnect(self._manager.key_press_handler_id)
        self._canvas.mpl_disconnect(self._manager.button_press_handler_id)
        self._axes = self._canvas.figure.add_axes((0, 0, 1, 1))
        self._axes.set_axis_off()
        self._axes.set_navigate(False)
        self._image = None
        self._texts = []
        self._mouse_callback = None
        self._keys = deque()
        self.is_closed = False
        for name in ('button_press_event', 'button_release_event', 'motion_notify_event', 'scroll_event'):
            self._canvas.mpl_connect(name, self._mouse_event)
        self._canvas.mpl_connect('key_press_event', self._key_event)
        self._canvas.mpl_connect('close_event', self._close_event)
        self._canvas.mpl_connect('resize_event', self._resize_event)

    def set_mouse_callback(self, callback: Callable[[int, int, int, int, object], None]) -> None:
        self._mouse_callback = callback

    def _mouse_event(self, event):
        if self._mouse_callback is None:
            return
        flags = 0
        if event.name == 'button_release_event' and event.button == 1:
            code = cv2.EVENT_LBUTTONUP
        elif event.inaxes is not self._axes or event.xdata is None or event.ydata is None:
            return
        elif event.name == 'scroll_event':
            if not event.step:
                return
            code, flags = cv2.EVENT_MOUSEWHEEL, 1 if event.step > 0 else -1
        elif event.name == 'button_press_event' and event.button == 1:
            code = cv2.EVENT_LBUTTONDOWN
        elif event.name == 'motion_notify_event':
            code = cv2.EVENT_MOUSEMOVE
        else:
            return
        x = -1 if event.xdata is None else round(event.xdata)
        y = -1 if event.ydata is None else round(event.ydata)
        self._mouse_callback(code, x, y, flags, None)

    def _key_event(self, event):
        key = {'escape': 27, 'enter': 13, 'return': 13}.get(event.key)
        if key is None and event.key is not None and len(event.key) == 1:
            key = ord(event.key)
        if key is not None:
            self._keys.append(key)

    def _close_event(self, event):
        self.is_closed = True

    def _resize_event(self, event):
        self._axes.apply_aspect()
        scale = self._axes.transData.get_matrix()[0, 0]
        for text in self._texts:
            text.set_fontsize(FONT_SIZE_PX*scale*72/self._canvas.figure.dpi)

    def show_image(self, image: np.ndarray, labels: tuple[AnnotationLabel, ...] = ()) -> None:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Annotation window expects a display-only BGR8 canvas')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._image is None:
            self._image = self._axes.imshow(rgb, origin='upper', interpolation='nearest')
            self._axes.set_autoscale_on(False)
            screen = self._manager.window.screen().availableGeometry()
            h, w = image.shape[:2]
            scale = min(1., .92*screen.width()/w, .88*screen.height()/h)
            self._manager.window.resize(round(w*scale), round(h*scale))
            self._manager.show()
        else:
            self._image.set_data(rgb)
        for text in self._texts:
            text.remove()
        self._texts = _draw_labels(self._axes, labels)
        self._canvas.draw_idle()

    def wait_key(self, delay_ms: int) -> int:
        if not self._keys and not self.is_closed:
            self._canvas.start_event_loop(delay_ms / 1000)
        return self._keys.popleft() if self._keys else -1

    def close(self) -> None:
        if not self.is_closed:
            self._manager.destroy()
        self.is_closed = True
