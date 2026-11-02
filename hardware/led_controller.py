# -*- coding: utf-8 -*-
"""WS2812 灯带上位机：手控 26 颗灯珠的颜色和亮度，经串口发给 Arduino。

协议：0xAA 0x55 + 26*3 字节 RGB + 1 字节 XOR 校验，115200 波特率。
依赖：pip install pyserial
"""

import tkinter as tk
from tkinter import ttk, colorchooser, messagebox

import serial
from serial.tools import list_ports

LED_COUNT = 26
HEADER = bytes([0xAA, 0x55])
BAUD = 115200

# 物理布局 -> 灯珠索引（与 Arduino 端分段定义一致）
# 上边 0-8 自左而右；右边 9-12 自上而下；下边 13-21 自右而左；左边 22-25 自下而上
GRID_POSITIONS = {}
for i in range(9):
    GRID_POSITIONS[i] = (0, i + 1)            # 上边：第 0 行，第 1-9 列
for i in range(4):
    GRID_POSITIONS[9 + i] = (i + 1, 10)       # 右边：第 10 列
for i in range(9):
    GRID_POSITIONS[13 + i] = (5, 9 - i)       # 下边：第 5 行，自右而左
for i in range(4):
    GRID_POSITIONS[22 + i] = (4 - i, 0)       # 左边：第 0 列，自下而上


class LedController(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WS2812 灯带控制器")
        self.resizable(False, False)

        self.ser = None
        self.selected = 0
        # 每颗灯珠的状态：[r, g, b, brightness(0-100)]
        self.leds = [[0, 0, 0, 100] for _ in range(LED_COUNT)]
        self.buttons = []
        self.global_brightness = tk.IntVar(value=100)  # 整体亮度 0-100%

        self._build_connection_bar()
        self._build_led_grid()
        self._build_sliders()
        self._build_actions()

        self._load_default_pattern()
        self._select_led(0)

    # ---------- 界面 ----------

    def _build_connection_bar(self):
        bar = ttk.Frame(self, padding=6)
        bar.grid(row=0, column=0, sticky="w")

        ttk.Label(bar, text="串口:").pack(side="left")
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(bar, textvariable=self.port_var, width=12, state="readonly")
        self.port_box.pack(side="left", padx=4)
        self._refresh_ports()

        ttk.Button(bar, text="刷新", command=self._refresh_ports).pack(side="left", padx=4)
        self.conn_btn = ttk.Button(bar, text="连接", command=self._toggle_connection)
        self.conn_btn.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(bar, textvariable=self.status_var, foreground="gray").pack(side="left", padx=10)

    def _build_led_grid(self):
        frame = ttk.Frame(self, padding=10)
        frame.grid(row=1, column=0)

        for idx in range(LED_COUNT):
            r, c = GRID_POSITIONS[idx]
            btn = tk.Button(
                frame, text=str(idx), width=4, height=2,
                bg="#000000", fg="white", activeforeground="white",
                relief="raised", command=lambda i=idx: self._select_led(i),
            )
            btn.grid(row=r, column=c, padx=2, pady=2)
            self.buttons.append(btn)

    def _build_sliders(self):
        frame = ttk.LabelFrame(self, text="选中灯珠", padding=8)
        frame.grid(row=2, column=0, sticky="ew", padx=8, pady=4)

        self.sel_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.sel_var).grid(row=0, column=0, columnspan=2, sticky="w")

        self.vars = {}
        specs = [("R", 0, 255), ("G", 0, 255), ("B", 0, 255), ("亮度%", 0, 100)]
        for row, (name, lo, hi) in enumerate(specs, start=1):
            ttk.Label(frame, text=name, width=6).grid(row=row, column=0, sticky="w")
            var = tk.IntVar(value=100 if name == "亮度%" else 0)
            scale = ttk.Scale(
                frame, from_=lo, to=hi, variable=var, length=360,
                command=lambda _v: self._on_slider(),
            )
            scale.grid(row=row, column=1, sticky="ew")
            self.vars[name] = var

        ttk.Button(frame, text="调色板...", command=self._pick_color).grid(row=5, column=1, sticky="e", pady=4)

        gframe = ttk.LabelFrame(self, text="整体亮度", padding=8)
        gframe.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        self.global_label = ttk.Label(gframe, text="100%", width=6)
        self.global_label.pack(side="right")
        ttk.Scale(
            gframe, from_=0, to=100, variable=self.global_brightness, length=360,
            command=lambda _v: self._on_global_brightness(),
        ).pack(side="left", fill="x", expand=True)

    def _build_actions(self):
        frame = ttk.Frame(self, padding=8)
        frame.grid(row=4, column=0, sticky="ew")
        ttk.Button(frame, text="当前颜色应用到全部", command=self._apply_to_all).pack(side="left", padx=4)
        ttk.Button(frame, text="全部熄灭", command=self._all_off).pack(side="left", padx=4)
        ttk.Button(frame, text="恢复默认", command=self._reset_default).pack(side="left", padx=4)
        ttk.Button(frame, text="重新发送", command=self._send_frame).pack(side="left", padx=4)

    # ---------- 默认图案 ----------

    @staticmethod
    def _boost(v, k=1.3):
        return min(255, int(round(v * k)))

    def _load_default_pattern(self):
        """按规格初始化：左纯蓝、右纯绿、上 75% 红+蓝绿过渡、下蓝绿渐变，端部 ×1.3。"""
        # 左边 22-25：纯蓝
        for i in range(22, 26):
            self.leds[i] = [0, 0, 255, 100]

        # 右边 9-12：纯绿
        for i in range(9, 13):
            self.leds[i] = [0, 255, 0, 100]

        # 上边 0-8：红 75%（191），其余 25%（64）由蓝（左）过渡到绿（右），两端加亮
        for i in range(9):
            t = i / 8.0
            r, g, b = 191, int(round(64 * t)), int(round(64 * (1.0 - t)))
            if i == 0 or i == 8:
                r, g, b = self._boost(r), self._boost(g), self._boost(b)
            self.leds[i] = [r, g, b, 100]

        # 下边 13-21：索引 13 在最右（偏绿），21 在最左（偏蓝），两端加亮
        for i in range(9):
            t = 1.0 - i / 8.0
            g, b = int(round(255 * t)), int(round(255 * (1.0 - t)))
            if i == 0 or i == 8:
                g, b = self._boost(g), self._boost(b)
            self.leds[13 + i] = [0, g, b, 100]

        for i in range(LED_COUNT):
            self._update_button_color(i)

    def _reset_default(self):
        self._load_default_pattern()
        self._select_led(self.selected)
        self._send_frame()

    # ---------- 串口 ----------

    def _refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        self.port_box["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])

    def _toggle_connection(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
            self.conn_btn.config(text="连接")
            self.status_var.set("未连接")
            return
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("提示", "请先选择串口")
            return
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.2)
        except serial.SerialException as e:
            messagebox.showerror("连接失败", str(e))
            return
        self.conn_btn.config(text="断开")
        self.status_var.set(f"已连接 {port}")
        self._send_frame()

    # ---------- 灯珠操作 ----------

    def _select_led(self, idx):
        self.selected = idx
        r, g, b, br = self.leds[idx]
        self.vars["R"].set(r)
        self.vars["G"].set(g)
        self.vars["B"].set(b)
        self.vars["亮度%"].set(br)
        self.sel_var.set(f"灯珠 #{idx}")
        for i, btn in enumerate(self.buttons):
            btn.config(relief="sunken" if i == idx else "raised")

    def _on_slider(self):
        idx = self.selected
        self.leds[idx] = [
            self.vars["R"].get(), self.vars["G"].get(),
            self.vars["B"].get(), self.vars["亮度%"].get(),
        ]
        self._update_button_color(idx)
        self._send_frame()

    def _pick_color(self):
        idx = self.selected
        r, g, b, _ = self.leds[idx]
        result = colorchooser.askcolor(color=(r, g, b), title=f"灯珠 #{idx}")
        if result and result[0]:
            r, g, b = (int(v) for v in result[0])
            self.vars["R"].set(r)
            self.vars["G"].set(g)
            self.vars["B"].set(b)
            self._on_slider()

    def _apply_to_all(self):
        r, g, b, br = self.leds[self.selected]
        for i in range(LED_COUNT):
            self.leds[i] = [r, g, b, br]
            self._update_button_color(i)
        self._send_frame()

    def _all_off(self):
        for i in range(LED_COUNT):
            self.leds[i] = [0, 0, 0, self.leds[i][3]]
            self._update_button_color(i)
        self._on_slider()  # 同步滑块并发送

    def _on_global_brightness(self):
        v = self.global_brightness.get()
        self.global_label.config(text=f"{v}%")
        for i in range(LED_COUNT):
            self._update_button_color(i)
        self._send_frame()

    def _update_button_color(self, idx):
        r, g, b, br = self.leds[idx]
        scale = br / 100.0 * self.global_brightness.get() / 100.0
        color = f"#{int(r * scale):02x}{int(g * scale):02x}{int(b * scale):02x}"
        fg = "white" if (r + g + b) * scale < 380 else "black"
        self.buttons[idx].config(bg=color, fg=color if (r, g, b, br) == (0, 0, 0, br) else fg)
        if (r, g, b) == (0, 0, 0):
            self.buttons[idx].config(fg="white")

    # ---------- 发送 ----------

    def _send_frame(self):
        if not (self.ser and self.ser.is_open):
            return
        payload = bytearray()
        gs = self.global_brightness.get() / 100.0
        for r, g, b, br in self.leds:
            s = br / 100.0 * gs
            payload += bytes([int(r * s), int(g * s), int(b * s)])
        checksum = 0
        for byte in payload:
            checksum ^= byte
        try:
            self.ser.write(HEADER + bytes(payload) + bytes([checksum]))
            ack = self.ser.read(1)
            if ack == b"K":
                self.status_var.set("已发送 ✓")
            elif ack == b"E":
                self.status_var.set("校验失败，请重发")
        except serial.SerialException as e:
            self.status_var.set(f"发送失败: {e}")


if __name__ == "__main__":
    app = LedController()
    app.mainloop()
