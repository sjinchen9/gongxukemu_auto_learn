# -*- coding: utf-8 -*-
"""
自动学习脚本 v3.1
状态机驱动 + 全局弹窗检测：每轮自检 -> VIDEO/COURSE 双状态
"""

import time, sys, os, threading, msvcrt
import pyautogui
from PIL import Image
import numpy as np

# ======================== 配置 ========================
class Config:
    SCAN_VIDEO    = 60   # 视频播放时扫描间隔
    SCAN_OTHER    = 10   # 其他操作扫描间隔
    SCAN_INITIAL  = 15   # 前两次快速扫描间隔
    INITIAL_SCANS = 3    # 快速扫描次数
    MAX_CLICKS    = 50   # 最大点击次数
    TEMPLATE_SCORE = 0.45 # 模板匹配阈值
    TEMPLATE_STEP = 8    # 搜索步长
    MISS_THRESHOLD = 3   # 连续未检测到次数阈值
    SCROLL_LIMIT   = 5   # 课程页滚动上限

# ======================== 工具函数 ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(SCRIPT_DIR, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def check_key():
    if msvcrt.kbhit():
        try:
            k = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if k == 's': return 'manual'
            if k == 'a': return 'auto'
            if k == 'q': return 'quit'
        except: pass
    return None

# ======================== 图像工具 ========================
class ImageUtils:
    @staticmethod
    def load(path):
        if not os.path.exists(path): return None
        img = Image.open(path)
        arr = np.array(img)
        gray = np.mean(arr[:,:,:3], axis=2).astype(np.float32)
        return gray, gray.shape[1], gray.shape[0], gray.mean(), gray.std()

    @staticmethod
    def screenshot():
        img = pyautogui.screenshot()
        color = np.array(img)
        gray = np.mean(color[:,:,:3], axis=2).astype(np.float32)
        return img, color, gray

    @staticmethod
    def match(screen_gray, tmpl, region=None):
        if tmpl is None: return None
        t_gray, tw, th, t_mean, t_std = tmpl
        if region:
            x1, y1, x2, y2 = region
            sub = screen_gray[y1:y2, x1:x2]
            ox, oy = x1, y1
        else:
            sub = screen_gray
            ox, oy = 0, 0

        rh, rw = sub.shape
        if rw < tw or rh < th: return None

        best_s, bx, by = -1, 0, 0
        s = Config.TEMPLATE_STEP
        for y in range(0, rh - th + 1, s):
            for x in range(0, rw - tw + 1, s):
                p = sub[y:y+th, x:x+tw]
                ps = p.std()
                if ps < 15 or t_std < 15: continue
                score = ((p - p.mean()) / ps * (t_gray - t_mean) / t_std).mean()
                if score > best_s:
                    best_s, bx, by = score, x, y

        if best_s < Config.TEMPLATE_SCORE: return None
        return ox + bx + tw//2, oy + by + th//2, best_s

    @staticmethod
    def save(img, cx, cy, score, name):
        pad = 80
        r = img.crop((max(0,cx-pad), max(0,cy-pad), min(img.width,cx+pad), min(img.height,cy+pad)))
        fn = f"{name}_pos({cx},{cy})_s{score:.2f}.png"
        r.save(os.path.join(SCREENSHOTS_DIR, fn))
        return fn

# ======================== 模板管理器 ========================
class Templates:
    def __init__(self):
        self.next_btn     = ImageUtils.load(os.path.join(SCRIPT_DIR, "button_template.png"))
        self.close_btn    = ImageUtils.load(os.path.join(SCRIPT_DIR, "close_button_template.png"))
        self.confirm_btn  = ImageUtils.load(os.path.join(SCRIPT_DIR, "confirm_button_template.png"))
        self.done_confirm = ImageUtils.load(os.path.join(SCRIPT_DIR, "queren2.png"))
        self.tab_biuxiu   = ImageUtils.load(os.path.join(SCRIPT_DIR, "bixiuke.png"))
        self.tab_xuanxiu  = ImageUtils.load(os.path.join(SCRIPT_DIR, "xuanxiuke.png"))

    def check_loaded(self):
        btns = [self.next_btn, self.close_btn, self.confirm_btn, self.done_confirm]
        ok = sum(1 for t in btns if t)
        tabs_ok = self.tab_biuxiu is not None and self.tab_xuanxiu is not None
        log(f"模板加载: {ok}/4 按钮, 选项卡={'OK' if tabs_ok else '缺'}")
        return ok >= 3

# ======================== 检测器 ========================
class Detector:
    def __init__(self, templates):
        self.tmpl = templates

    def _check_blue(self, color, cx, cy, tw, th, sh, sw, min_ratio=0.10):
        """验证匹配区域是否是蓝色按钮"""
        y1 = max(0, cy - th//2)
        x1 = max(0, cx - tw//2)
        y2 = min(sh, y1 + th)
        x2 = min(sw, x1 + tw)
        patch = color[y1:y2, x1:x2]
        blue_pixels = ((patch[:,:,2] > 150) & (patch[:,:,0] < 120)).sum()
        return blue_pixels / patch.size >= min_ratio

    def find_next(self, color, gray, sw, sh):
        t = self.tmpl.next_btn
        if not t: return []
        _, tw, th, _, _ = t
        r_ch, g_ch, b_ch = color[:,:,0], color[:,:,1], color[:,:,2]
        blue = (b_ch > 180) & (r_ch < 120) & (g_ch < 200)
        blue[:int(sh*0.12), :] = False
        if not blue.any(): return []

        mid = sw // 2
        results = []
        for xs, xe in [(0, mid), (mid, sw)]:
            m = blue[:, xs:xe]
            if not m.any(): continue
            ys, xxs = np.where(m)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xxs.min())+xs, int(xxs.max())+xs
            px, py = tw*2, th*2
            r = ImageUtils.match(gray, t, (max(0,x1-px), max(0,y1-py), min(sw,x2+px), min(sh,y2+py)))
            if r:
                cx, cy, score = r
                # 排除右上角"关闭"按钮区域
                if cx > sw * 0.88 and cy < sh * 0.2:
                    continue
                if cy < sh*0.85:
                    above = color[max(0,cy-th-10), cx]
                    below = color[min(sh-1,cy+th+10), cx]
                    if all(above < 60) and all(below < 60):
                        results.append((cx, cy, score))
        return results

    def find_close(self, color, gray, sw, sh):
        t = self.tmpl.close_btn
        if not t: return None
        _, tw, th, _, _ = t
        # 搜索右上角：宽度75%-100%，高度0%-25%
        r = ImageUtils.match(gray, t, (int(sw*0.75), 0, sw, int(sh*0.25)))
        if r:
            cx, cy, score = r
            # 颜色验证：关闭按钮周围是蓝色标题栏
            y1 = max(0, cy - th//2)
            x1 = max(0, cx - tw//2)
            y2 = min(sh, y1 + th)
            x2 = min(sw, x1 + tw)
            patch = color[y1:y2, x1:x2]
            blue_pixels = ((patch[:,:,2] > 150) & (patch[:,:,0] < 120)).sum()
            if blue_pixels / patch.size > 0.08:  # 放宽到8%
                return (cx, cy, score)
        return None

    def find_confirm(self, color, gray, sw, sh):
        for tmpl, name in [(self.tmpl.confirm_btn, 'confirm'), (self.tmpl.done_confirm, 'done_confirm')]:
            if not tmpl: continue
            _, tw, th, _, _ = tmpl
            r = ImageUtils.match(gray, tmpl, (int(sw*0.2), int(sh*0.2), int(sw*0.8), int(sh*0.8)))
            if r:
                cx, cy, score = r
                if sh*0.3 < cy < sh*0.7 and self._check_blue(color, cx, cy, tw, th, sh, sw):
                    return (cx, cy, score, name)
        return None

    def find_incomplete(self, color, sw, sh):
        r, g, b = color[:,:,0], color[:,:,1], color[:,:,2]
        gray_m = (np.abs(r.astype(int)-g.astype(int))<30) & (np.abs(g.astype(int)-b.astype(int))<30) & (r>80) & (r<180)
        blue_m = (b>160) & (r<120) & (g<180)

        # 搜索课程卡片区域：跳过页面头部
        x1, x2 = int(sw*0.10), int(sw*0.90)
        y1, y2 = int(sh*0.35), int(sh*0.85)
        gray_m[:y1,:] = gray_m[y2:,:] = gray_m[:,:x1] = gray_m[:,x2:] = False
        blue_m[:y1,:] = blue_m[y2:,:] = blue_m[:,:x1] = blue_m[:,x2:] = False

        bys, bxs = np.where(blue_m)
        if len(bxs) < 50: return None

        gys, gxs = np.where(gray_m)
        if len(gxs) < 50: return None

        # 找到所有蓝色矩形标签（聚类x坐标）
        uniq_bx = sorted(set(int(x) for x in bxs))
        blue_clusters = []
        cur = [uniq_bx[0]]
        for x in uniq_bx[1:]:
            if x - cur[-1] < 12:
                cur.append(x)
            else:
                if len(cur) > 3: blue_clusters.append((cur[0], cur[-1]))
                cur = [x]
        if len(cur) > 3: blue_clusters.append((cur[0], cur[-1]))

        # 筛选：矩形标签（宽高比>1.5，宽度适中）
        candidates = []
        for bx1, bx2 in blue_clusters:
            bw = bx2 - bx1
            if bw < 40 or bw > 160:
                continue
            # 找到这个蓝色标签的Y范围
            col_blue = bxs[(bxs >= bx1) & (bxs <= bx2)]
            if len(col_blue) < 20: continue
            col_ys = bys[(bxs >= bx1) & (bxs <= bx2)]
            by_min, by_max = col_ys.min(), col_ys.max()
            bh = by_max - by_min
            if bh < 10 or bw / max(bh, 1) < 1.3:  # 必须够扁（宽高比>1.3）
                continue

            cy = (by_min + by_max) // 2
            # 检查左边有足够灰色文字（"未完成"）
            left_gray = gxs[(gxs > bx1 - 200) & (gxs < bx1) & (gys > cy - 10) & (gys < cy + 10)]
            if len(left_gray) < 20:
                continue

            # 这个蓝色标签上方不应有太多蓝色（排除嵌套在蓝色区块中的）
            above_blue = blue_m[max(0,by_min-15):by_min, bx1:bx2].sum()
            below_blue = blue_m[by_max:min(sh,by_max+15), bx1:bx2].sum()
            if above_blue > 50 or below_blue > 50:
                continue  # 这个蓝色标签嵌在更大的蓝色区块中

            gx = (left_gray.min() + left_gray.max()) // 2
            # 质量：左边灰色越多越好，蓝色标签越独立越好
            quality = min(len(left_gray) / 100.0, 1.0)
            candidates.append((gx, cy, quality))

        if not candidates: return None

        # 按Y排序，取最靠上的
        candidates.sort(key=lambda p: p[1])
        return candidates[0][:2]

    def find_tabs(self, gray, sw, sh):
        """用精确模板匹配找必修/选修选项卡按钮"""
        results = []
        for tmpl in [self.tmpl.tab_biuxiu, self.tmpl.tab_xuanxiu]:
            if not tmpl: continue
            r = ImageUtils.match(gray, tmpl, (0, int(sh*0.10), sw, int(sh*0.45)))
            if r:
                results.append(r)
        return [(cx, cy) for cx, cy, _ in results] if results else []

    def has_page(self, color, sw, sh):
        r, g, b = color[:,:,0], color[:,:,1], color[:,:,2]
        blue = (b>150) & (r<100) & (g<150)
        has_blue = blue[:int(sh*0.15),:].sum() > 1000
        center = color[int(sh*0.2):int(sh*0.8), int(sw*0.1):int(sw*0.8)]
        dark = (center[:,:,0]<50) & (center[:,:,1]<50) & (center[:,:,2]<50)
        has_dark = dark.sum()/dark.size > 0.2
        return has_blue or has_dark

# ======================== 动作执行器 ========================
class ActionExecutor:
    @staticmethod
    def click(x, y):
        pyautogui.moveTo(x, y, duration=0.3)
        time.sleep(0.1)
        pyautogui.click(x, y)
        time.sleep(0.05)
        t = threading.Thread(target=ActionExecutor._effect, args=(x,y)); t.daemon = True; t.start()

    @staticmethod
    def _effect(x, y):
        try:
            import tkinter as tk
            root = tk.Tk(); root.overrideredirect(True)
            root.attributes('-topmost', True); root.attributes('-alpha', 0.5)
            root.config(bg='red')
            root.geometry(f'60x60+{x-30}+{y-30}')
            root.wm_attributes('-transparentcolor', 'red')
            c = tk.Canvas(root, width=60, height=60, bg='black', highlightthickness=0)
            c.pack(); c.create_oval(5,5,55,55, outline='red', width=3)
            root.after(600, root.destroy); root.mainloop()
        except: pass

    @staticmethod
    def focus_browser():
        """点击右上角蓝色标题栏区域获取焦点（安全区域，不会有可点击元素）"""
        sw, sh = pyautogui.size()
        pyautogui.click(int(sw * 0.95), int(sh * 0.04))
        time.sleep(0.3)

    @staticmethod
    def scroll_down():
        pyautogui.press('pagedown')
        time.sleep(0.5)

    @staticmethod
    def scroll_up():
        pyautogui.press('pageup')
        time.sleep(0.5)

# ======================== 状态 ========================
class State:
    VIDEO   = 0  # 视频页（有关闭按钮）
    COURSE  = 1  # 课程列表（无关闭无弹窗）

_STATE_NAMES = ['视频', '选课']

# ======================== 主程序 ========================
class AutoLearner:
    def __init__(self):
        self.state = None  # 初始未知，首次扫描时自检
        self.clicks = 0
        self.scan_count = 0
        self.miss_count = 0        # 连续未找到目标计数
        self.course_scrolls = 0
        self.tab_index = 0
        self.needs_refresh = False  # 进入选课页时需刷新页面
        self.manual = False
        self.templates = Templates()
        self.detector = Detector(self.templates)

    def run(self):
        log("=" * 50)
        log("  自动学习脚本 v3.2（双状态 + 全局弹窗 + 链式处理）")
        log("=" * 50)
        log("  s=手动扫描  a=自动模式  q=退出")
        log(f"  截图保存: {SCREENSHOTS_DIR}")
        self.templates.check_loaded()
        log("3秒后开始扫描...")
        time.sleep(3)
        self._scan()  # 立即执行首次扫描

        while self.clicks < Config.MAX_CLICKS:
            if self.manual:
                self._scan()
                for _ in range(10):
                    time.sleep(0.1)
                    cmd = check_key()
                    if cmd == 'quit': return
                    if cmd == 'auto': self.manual = False; break
            else:
                interval = self._get_interval()
                log(f"等待 {interval}s...")
                for _ in range(interval):
                    time.sleep(1)
                    cmd = check_key()
                    if cmd == 'quit': return
                    if cmd == 'manual': self.manual = True; break
                    if cmd == 'auto': self.manual = False
                if not self.manual:
                    self._scan()

        log(f"完成！共点击 {self.clicks} 次")

    def _get_interval(self):
        if self.scan_count < Config.INITIAL_SCANS:
            return Config.SCAN_INITIAL
        return Config.SCAN_VIDEO if self.state == State.VIDEO else Config.SCAN_OTHER

    def _scan(self):
        self.scan_count += 1

        # 进入选课页时刷新页面
        if self.needs_refresh and self.state == State.COURSE:
            self.needs_refresh = False
            log("刷新页面(F5)，等待10秒...")
            ActionExecutor.focus_browser()
            pyautogui.press('f5')
            time.sleep(10)
            ActionExecutor.scroll_down()

        img, color, gray = ImageUtils.screenshot()
        sw, sh = img.size

        if not self.detector.has_page(color, sw, sh):
            log(f"第{self.scan_count}轮: 未检测到培训页面，跳过")
            return

        # 状态未知时先检测（验证失败后或首次启动）
        if self.state is None:
            self.state = self._detect_state(color, gray, sw, sh)
            log(f"第{self.scan_count}轮 启动检测: [{_STATE_NAMES[self.state]}]")
            if self.state == State.COURSE:
                self.needs_refresh = True

        # 全局优先：弹窗确认按钮任何时候都立即处理
        confirm = self.detector.find_confirm(color, gray, sw, sh)
        if confirm:
            cx, cy, score, name = confirm
            log(f"第{self.scan_count}轮 [{_STATE_NAMES[self.state]}] 弹窗确认 [{name}] ({cx},{cy})")
            self._do_click(img, cx, cy, score, name)
            time.sleep(5)
            # 弹窗确认后很可能回到课程列表，标记需要刷新
            self.needs_refresh = True
            return

        log(f"第{self.scan_count}轮 [{_STATE_NAMES[self.state]}]")

        # 每轮自检
        close = self.detector.find_close(color, gray, sw, sh)
        nexts = len(self.detector.find_next(color, gray, sw, sh)) > 0
        center = color[int(sh*0.2):int(sh*0.8), int(sw*0.1):int(sw*0.8)]
        dark_ratio = ((center[:,:,0]<50)&(center[:,:,1]<50)&(center[:,:,2]<50)).sum() / center.shape[0] / center.shape[1]
        on_video = close is not None or nexts or dark_ratio > 0.3

        if on_video and self.state == State.COURSE:
            log(f"  自检修正: -> 视频")
            self.state = State.VIDEO
            self.needs_refresh = False  # 进视频页清除刷新标志
            self.miss_count = 0
        elif not on_video and self.state == State.VIDEO:
            log(f"  自检修正: -> 选课")
            self.state = State.COURSE
            self.course_scrolls = 0
            self.needs_refresh = True

        if self.state == State.VIDEO:
            self._do_video(img, color, gray, sw, sh)
        elif self.state == State.COURSE:
            self._do_course(img, color, gray, sw, sh)

    def _detect_state(self, color, gray, sw, sh):
        """多信号综合判定：视频页 vs 课程列表（弹窗由全局检测处理）"""
        has_close = self.detector.find_close(color, gray, sw, sh) is not None
        has_next = len(self.detector.find_next(color, gray, sw, sh)) > 0
        center = color[int(sh*0.2):int(sh*0.8), int(sw*0.1):int(sw*0.8)]
        dark_ratio = ((center[:,:,0]<50) & (center[:,:,1]<50) & (center[:,:,2]<50)).sum() / center.shape[0] / center.shape[1]
        has_course = self.detector.find_incomplete(color, sw, sh) is not None

        if has_close or has_next or dark_ratio > 0.3:
            return State.VIDEO
        return State.COURSE

    # ---------- 状态处理 ----------
    def _do_video(self, img, color, gray, sw, sh):
        nexts = self.detector.find_next(color, gray, sw, sh)
        if nexts:
            self.miss_count = 0
            nexts.sort(key=lambda x: x[2], reverse=True)
            cx, cy, score = nexts[0]
            self._do_click(img, cx, cy, score, 'next')
            return

        # 没有下一节：区分"正在播放"和"全部学完"
        center = color[int(sh*0.3):int(sh*0.7), int(sw*0.15):int(sw*0.85)]
        dark = (center[:,:,0]<50) & (center[:,:,1]<50) & (center[:,:,2]<50)
        video_ended = dark.sum()/dark.size > 0.5

        if video_ended:
            # 视频播完了，尝试点关闭
            self.miss_count = 0
            close = self.detector.find_close(color, gray, sw, sh)
            if close:
                cx, cy, score = close
                log(f"  视频结束，点击关闭 ({cx},{cy}) s:{score:.2f}")
                self._do_click(img, cx, cy, score, 'close')
            else:
                log("  视频结束但未检测到关闭按钮，等待...")
            return

        # 视频还在播放，不计数——等下一节按钮自然出现
        self.miss_count = 0
        log(f"  视频播放中，等待下一节...")

    def _do_course(self, img, color, gray, sw, sh):
        inc = self.detector.find_incomplete(color, sw, sh)
        if inc:
            gx, gy = inc
            self._do_click(img, gx, gy, 0.8, 'incomplete')
            time.sleep(2)
            _, c2, g2 = ImageUtils.screenshot()
            sw2, sh2 = c2.shape[1], c2.shape[0]
            close = self.detector.find_close(c2, g2, sw2, sh2)
            if close:
                self.course_scrolls = 0
                self.state = State.VIDEO
                self.needs_refresh = False
                self.miss_count = 0
                log("  -> 视频")
                return
            else:
                log(f"  误点击 ({gx},{gy})，跳过继续...")

        # 还没滚到底，一次性跳到底
        if self.course_scrolls == 0:
            log("  未找到未完成，跳转到页底...")
            ActionExecutor.focus_browser()
            pyautogui.press('end')
            time.sleep(1)
            self.course_scrolls = 1
            return

        # 在底部检查是否有未完成
        inc2 = self.detector.find_incomplete(color, sw, sh)
        if inc2:
            gx, gy = inc2
            self._do_click(img, gx, gy, 0.8, 'incomplete')
            time.sleep(2)
            _, c2, g2 = ImageUtils.screenshot()
            sw2, sh2 = c2.shape[1], c2.shape[0]
            if self.detector.find_close(c2, g2, sw2, sh2):
                self.course_scrolls = 0
                self.state = State.VIDEO
                self.needs_refresh = False
                self.miss_count = 0
                log("  -> 视频")
                return
            else:
                log("  误点击，继续...")

        # 底部也没有 → 回顶部，逐步向下找选项卡
        log("  底部无未完成，回顶部找选项卡...")
        ActionExecutor.focus_browser()
        pyautogui.press('home')
        time.sleep(1)

        tabs = None
        for offset in range(6):
            _, c3, g3 = ImageUtils.screenshot()
            sw3, sh3 = c3.shape[1], c3.shape[0]
            tabs = self.detector.find_tabs(g3, sw3, sh3)
            if tabs and len(tabs) >= 2:
                break
            if offset < 5:
                ActionExecutor.scroll_down()
                log(f"  向下找选项卡...({offset+1}/5)")

        if tabs and len(tabs) >= 1:
            self.tab_index = (self.tab_index + 1) % len(tabs)
            tx, ty = tabs[self.tab_index]
            log(f"  切换选项卡 ({tx},{ty})")
            ActionExecutor.click(tx, ty)
            self.clicks += 1
            self.course_scrolls = 0
            time.sleep(2)
            return

        log("  暂无目标，继续等待...")
        self.course_scrolls = 0

    def _do_click(self, img, cx, cy, score, name):
        fn = ImageUtils.save(img, cx, cy, score, f"{name}_r{self.scan_count}")
        log(f"  点击 [{name}] ({cx},{cy}) s:{score:.2f} -> screenshots/{fn}")
        ActionExecutor.click(cx, cy)
        self.clicks += 1

        if not self._verify_click(name, img):
            self.miss_count = 0
            if name != 'incomplete':
                self.state = None
                self.course_scrolls = 0
            log("  -> 验证失败，下轮重启检测")
            return

        # 点击关闭后必弹确认窗，立即处理，不等下轮
        if name == 'close':
            time.sleep(2)
            _, c2, g2 = ImageUtils.screenshot()
            sw2, sh2 = c2.shape[1], c2.shape[0]
            confirm = self.detector.find_confirm(c2, g2, sw2, sh2)
            if confirm:
                cx2, cy2, score2, name2 = confirm
                log(f"  链式点击弹窗 [{name2}] ({cx2},{cy2})")
                ActionExecutor.click(cx2, cy2)
                self.clicks += 1
                time.sleep(5)
                self.needs_refresh = True  # 弹窗后回到课程列表，下次刷新
            else:
                log("  关闭后未检测到弹窗，下轮重试")

    def _verify_click(self, name, click_img=None):
        time.sleep(1.5)
        img2, color, gray = ImageUtils.screenshot()
        sw, sh = img2.size

        ok = False
        if name == 'next':
            center = color[int(sh*0.2):int(sh*0.8), int(sw*0.1):int(sw*0.8)]
            dark = (center[:,:,0]<50) & (center[:,:,1]<50) & (center[:,:,2]<50)
            ok = dark.sum()/dark.size > 0.2
        elif name == 'incomplete':
            c = self.detector.find_close(color, gray, sw, sh)
            ok = c is not None
        elif name == 'close':
            c = self.detector.find_confirm(color, gray, sw, sh)
            ok = c is not None
        elif name in ('confirm', 'done_confirm'):
            c = self.detector.find_confirm(color, gray, sw, sh)
            ok = c is None
        elif name == 'tab':
            return True

        if ok:
            log("    OK")
            return True
        else:
            log("    ? 验证未通过")
            if click_img:
                fn = f"DEBUG_{name}_r{self.scan_count}_before_click.png"
                click_img.save(os.path.join(SCREENSHOTS_DIR, fn))
            fn2 = f"DEBUG_{name}_r{self.scan_count}_after_1.5s.png"
            img2.save(os.path.join(SCREENSHOTS_DIR, fn2))
            return False

# ======================== 入口 ========================
if __name__ == "__main__":
    try:
        AutoLearner().run()
    except KeyboardInterrupt:
        print("\n用户手动停止")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback; traceback.print_exc()
        input("按回车键退出...")
