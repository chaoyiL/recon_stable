# -*- coding: utf-8 -*-
"""WS2812 灯带上位机：配置四边数量、空出前缀及每颗灯珠颜色。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, messagebox, ttk

import serial
from serial.tools import list_ports

from hardware.led_protocol import (
    BAUD, LAYOUT_FIELDS, MAX_LED_COUNT, LedLayout, LedSettings, default_colors, encode_frame,
)
from utils.config import ConfigError, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class LedController(tk.Tk):
    def __init__(
        self,
        layout: LedLayout | None = None,
        brightness: int = 100,
        rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        settings = LedSettings(
            layout=layout if layout is not None else LedLayout(),
            brightness=brightness,
            rgb_scale=rgb_scale,
        )
        self.title("WS2812 灯带控制器")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        # X11 的 DPI 可能异常，使用像素宽度避免弹窗默认的 3i 挤成窄列。
        self.option_add("*Dialog.msg.wrapLength", 520)
        self.option_add("*Dialog.dtl.wrapLength", 520)

        self.layout = settings.layout
        self.rgb_scale = settings.rgb_scale
        self.leds = default_colors(self.layout)
        self.ser = None
        self.ready = False
        self.ready_after_id = None
        self.send_after_id = None
        self.selected = 0
        self.buttons = []
        self.global_brightness = tk.IntVar(value=settings.brightness)

        self._build_connection_bar()
        self._build_layout_controls()
        self._build_led_grid()
        self._build_sliders()
        self._build_actions()
        self._select_led(0)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_connection_bar(self):
        bar = ttk.Frame(self, padding=6)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(bar, text="串口:").pack(side="left")
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(
            bar, textvariable=self.port_var, width=16, state="readonly",
        )
        self.port_box.pack(side="left", padx=4)
        self._refresh_ports()
        ttk.Button(bar, text="刷新", command=self._refresh_ports).pack(side="left", padx=4)
        self.conn_btn = ttk.Button(bar, text="连接", command=self._toggle_connection)
        self.conn_btn.pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(bar, textvariable=self.status_var, wraplength=260).pack(side="left", padx=10)

    def _build_layout_controls(self):
        frame = ttk.LabelFrame(self, text="灯带布局（每边灯珠数）", padding=8)
        frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self.layout_vars = {}
        for column, (name, label) in enumerate(zip(LAYOUT_FIELDS, ("上", "右", "下", "左", "空出"))):
            ttk.Label(frame, text=label).grid(row=0, column=column)
            value = tk.StringVar(value=str(getattr(self.layout, name)))
            self.layout_vars[name] = value
            ttk.Spinbox(
                frame, from_=0, to=MAX_LED_COUNT, width=6, textvariable=value,
            ).grid(row=1, column=column, padx=4)
        ttk.Button(frame, text="应用布局", command=self._apply_layout).grid(
            row=0, column=5, rowspan=2, padx=8,
        )
        self.layout_summary = tk.StringVar()
        ttk.Label(self, textvariable=self.layout_summary).grid(
            row=2, column=0, sticky="w", padx=12, pady=4,
        )

    def _build_led_grid(self):
        frame = ttk.Frame(self)
        frame.grid(row=3, column=0, sticky="nsew", padx=8)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.grid_canvas = tk.Canvas(
            frame, width=min(680, max(360, self.winfo_screenwidth() - 120)),
            height=min(280, max(140, self.winfo_screenheight() // 3)),
            highlightthickness=0,
        )
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=self.grid_canvas.xview)
        vertical = ttk.Scrollbar(frame, orient="vertical", command=self.grid_canvas.yview)
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical.grid(row=0, column=1, sticky="ns")
        self.grid_canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.led_frame = ttk.Frame(self.grid_canvas, padding=10)
        self.grid_canvas.create_window(0, 0, window=self.led_frame, anchor="nw")
        self.led_frame.bind("<Configure>", lambda _event: self.grid_canvas.configure(
            scrollregion=self.grid_canvas.bbox("all"),
        ))
        self._rebuild_led_grid()

    def _rebuild_led_grid(self):
        for child in self.led_frame.winfo_children():
            child.destroy()
        self.buttons = []
        for idx, (row, column) in enumerate(self.layout.grid_positions()):
            button = tk.Button(
                self.led_frame, text=str(self.layout.skip + idx), width=4, height=2,
                fg="white", activeforeground="white", relief="raised",
                command=lambda i=idx: self._select_led(i),
            )
            button.grid(row=row, column=column, padx=2, pady=2)
            self.buttons.append(button)
            self._update_button_color(idx)
        self.layout_summary.set(
            f"可控 {self.layout.active_count} 颗 + 开头熄灭 {self.layout.skip} 颗"
            f" = 总数 {self.layout.total_count} 颗（上限 {MAX_LED_COUNT}）；编号为实际灯带索引"
        )
        self.after_idle(self._fit_grid_viewport)

    def _fit_grid_viewport(self):
        self.update_idletasks()
        # 小布局完整显示；超过屏幕可用空间的大布局通过滚动查看。
        fixed_height = self.winfo_reqheight() - self.grid_canvas.winfo_reqheight()
        max_height = max(140, min(600, self.winfo_screenheight() - fixed_height - 100))
        max_width = max(360, min(960, self.winfo_screenwidth() - 100))
        self.grid_canvas.configure(
            width=min(max_width, max(480, self.led_frame.winfo_reqwidth())),
            height=min(max_height, self.led_frame.winfo_reqheight()),
        )

    def _apply_layout(self):
        try:
            layout = LedLayout(**{name: int(value.get()) for name, value in self.layout_vars.items()})
        except ValueError as error:
            messagebox.showerror("布局无效", str(error), parent=self)
            return
        # 按边保留已有颜色，避免上边变长后把右边颜色错配到其他灯珠。
        colors = default_colors(layout)
        old_offset = new_offset = 0
        for old_count, new_count in zip(self.layout.edge_counts, layout.edge_counts):
            count = min(old_count, new_count)
            colors[new_offset:new_offset + count] = self.leds[old_offset:old_offset + count]
            old_offset += old_count
            new_offset += new_count
        self.layout, self.leds = layout, colors
        self.selected = 0
        self._rebuild_led_grid()
        self._select_led(0)
        self._send_frame()

    def _build_sliders(self):
        frame = ttk.LabelFrame(self, text="选中灯珠", padding=8)
        frame.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        frame.columnconfigure(1, weight=1)
        self.sel_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.sel_var).grid(row=0, column=0, columnspan=2, sticky="w")
        self.vars = {}
        for row, (name, maximum) in enumerate((("R", 255), ("G", 255), ("B", 255), ("亮度%", 100)), start=1):
            ttk.Label(frame, text=name, width=6).grid(row=row, column=0, sticky="w")
            value = tk.IntVar(value=100 if name == "亮度%" else 0)
            self.vars[name] = value
            ttk.Scale(
                frame, from_=0, to=maximum, variable=value, length=360,
                command=lambda _value: self._on_slider(),
            ).grid(row=row, column=1, sticky="ew")
        ttk.Button(frame, text="调色板...", command=self._pick_color).grid(row=5, column=1, sticky="e", pady=4)
        global_frame = ttk.LabelFrame(self, text="整体亮度", padding=8)
        global_frame.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        self.global_label = ttk.Label(
            global_frame, text=f"{self.global_brightness.get()}%", width=6,
        )
        self.global_label.pack(side="right")
        ttk.Scale(
            global_frame, from_=0, to=100, variable=self.global_brightness, length=360,
            command=lambda _value: self._on_global_brightness(),
        ).pack(side="left", fill="x", expand=True)
        scale_text = " ".join(
            f"{channel}×{value:.2f}"
            for channel, value in zip("RGB", self.rgb_scale)
        )
        ttk.Label(
            global_frame,
            text=f"固定输出校正 {scale_text}；按钮颜色表示逻辑色",
        ).pack(side="bottom", anchor="w")

    def _build_actions(self):
        frame = ttk.Frame(self, padding=8)
        frame.grid(row=6, column=0, sticky="ew")
        for text, command in (
            ("当前颜色应用到全部", self._apply_to_all), ("全部熄灭", self._all_off),
            ("恢复默认", self._reset_default), ("重新发送", self._send_frame),
        ):
            ttk.Button(frame, text=text, command=command).pack(side="left", padx=4)

    def _reset_default(self):
        self.leds = default_colors(self.layout)
        self._refresh_colors()
        self._send_frame()

    def _refresh_ports(self):
        ports = sorted(list_ports.comports(), key=lambda port: (port.vid is None, port.device))
        devices = [port.device for port in ports]
        self.port_box["values"] = devices
        if devices and self.port_var.get() not in devices:
            self.port_var.set(devices[0])

    def _toggle_connection(self):
        if self.ser and self.ser.is_open:
            self._disconnect()
            self.status_var.set("未连接")
            return
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("提示", "请先选择串口", parent=self)
            return
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.5, write_timeout=1)
        except serial.SerialException as error:
            messagebox.showerror(
                "连接失败", f"无法打开串口 {port}。", detail=str(error), parent=self,
            )
            return
        self.conn_btn.config(text="断开")
        self.status_var.set("等待 Arduino 启动…")
        # Uno 打开串口时会复位；等待 bootloader 退出后再发当前完整状态。
        self.ready_after_id = self.after(2000, self._connection_ready)

    def _connection_ready(self):
        self.ready_after_id = None
        if self.ser and self.ser.is_open:
            self.ready = True
            self._send_frame()

    def _disconnect(self):
        for name in ("ready_after_id", "send_after_id"):
            task = getattr(self, name)
            if task is not None:
                self.after_cancel(task)
                setattr(self, name, None)
        self.ready = False
        if self.ser is not None:
            self.ser.close()
            self.ser = None
        self.conn_btn.config(text="连接")

    def _on_close(self):
        self._disconnect()
        self.destroy()

    def _select_led(self, idx):
        self.selected = idx
        for name, value in zip(("R", "G", "B", "亮度%"), self.leds[idx]):
            self.vars[name].set(value)
        self.sel_var.set(f"灯珠 #{self.layout.skip + idx}")
        for i, button in enumerate(self.buttons):
            button.config(relief="sunken" if i == idx else "raised")

    def _on_slider(self):
        self.leds[self.selected] = tuple(self.vars[name].get() for name in ("R", "G", "B", "亮度%"))
        self._update_button_color(self.selected)
        self._send_frame()

    def _pick_color(self):
        r, g, b, _ = self.leds[self.selected]
        result = colorchooser.askcolor(color=(r, g, b), title=f"灯珠 #{self.layout.skip + self.selected}")
        if result and result[0]:
            for name, value in zip(("R", "G", "B"), result[0]):
                self.vars[name].set(int(value))
            self._on_slider()

    def _apply_to_all(self):
        self.leds = [self.leds[self.selected]] * self.layout.active_count
        self._refresh_colors()
        self._send_frame()

    def _all_off(self):
        self.leds = [(0, 0, 0, brightness) for _, _, _, brightness in self.leds]
        self._refresh_colors()
        self._send_frame()

    def _on_global_brightness(self):
        self.global_label.config(text=f"{self.global_brightness.get()}%")
        for i in range(self.layout.active_count):
            self._update_button_color(i)
        self._send_frame()

    def _refresh_colors(self):
        for i in range(self.layout.active_count):
            self._update_button_color(i)
        self._select_led(self.selected)

    def _update_button_color(self, idx):
        r, g, b, brightness = self.leds[idx]
        scale = brightness / 100 * self.global_brightness.get() / 100
        color = f"#{int(r * scale):02x}{int(g * scale):02x}{int(b * scale):02x}"
        foreground = "white" if (r + g + b) * scale < 380 else "black"
        self.buttons[idx].config(
            bg=color, activebackground=color, fg=foreground, activeforeground=foreground,
        )

    def _send_frame(self):
        if not (self.ser and self.ser.is_open and self.ready):
            return
        if self.send_after_id is not None:
            self.after_cancel(self.send_after_id)
        # 滑块拖动时合并相邻更新，仅发送最新状态，避免串口帧排队。
        self.send_after_id = self.after(40, self._write_frame)

    def _write_frame(self):
        self.send_after_id = None
        if not (self.ser and self.ser.is_open and self.ready):
            return
        frame = encode_frame(
            self.layout, self.leds, self.global_brightness.get(),
            rgb_scale=self.rgb_scale,
        )
        try:
            self.ser.reset_input_buffer()
            if self.ser.write(frame) != len(frame):
                raise serial.SerialTimeoutException("数据帧未完整发送")
            ack = self.ser.read(1)
        except serial.SerialException as error:
            self._disconnect()
            self.status_var.set(f"发送失败：{error}")
            return
        messages = {
            b"K": f"已发送 {self.layout.total_count} 颗 ✓",
            b"E": "校验失败，请重新发送",
            b"L": f"下位机拒绝数量，请检查是否超过 {MAX_LED_COUNT} 颗",
            b"M": "下位机内存不足，请减少数量",
            b"": "确认超时，请检查是否上传新版固件",
        }
        self.status_var.set(messages.get(ack, f"未知下位机响应：{ack!r}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="配置并控制四边 WS2812 灯带")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML 配置文件")
    for name in LAYOUT_FIELDS:
        parser.add_argument(f"--{name}", type=int, help=f"覆盖 led_controller.{name}")
    args = parser.parse_args()
    try:
        settings = LedSettings.from_mapping(
            load_config(args.config).get("led_controller", {}),
        )
        overrides = {name: getattr(args, name) for name in LAYOUT_FIELDS if getattr(args, name) is not None}
        layout = replace(settings.layout, **overrides)
    except (ConfigError, ValueError) as error:
        parser.error(str(error))
    LedController(layout, settings.brightness, settings.rgb_scale).mainloop()


if __name__ == "__main__":
    main()
