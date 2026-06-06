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
        self.tab_biuxiu   = ImageUtils.load(os.path.join(SCRIPT_DIR, "bixiuke.png"))
        self.tab_xuanxiu  = ImageUtils.load(os.path.join(SCRIPT_DIR, "xuanxiuke.png"))
        self.pagination   = ImageUtils.load(os.path.join(SCRIPT_DIR, "qianwangdijiye.png"))
        self.incomplete_txt = ImageUtils.load(os.path.join(SCRIPT_DIR, "char_wei.png"))
        self.yiwancheng_txt = ImageUtils.load(os.path.join(SCRIPT_DIR, "char_yi.png"))

    def check_loaded(self):
        btns = [self.next_btn, self.close_btn, self.confirm_btn]
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
        """
        检测弹窗确定按钮（结构上下文法）
        弹窗 = 半透明遮罩 + 白色对话框容器 + 蓝色确定按钮
        必须先检测到白色对话框容器，再在容器内搜索按钮
        """
        tmpl = self.tmpl.confirm_btn
        if not tmpl: return None
        _, tw, th, _, _ = tmpl

        # === 第一步：检测弹窗对话框容器（白色/浅色矩形） ===
        r, g, b = color[:,:,0], color[:,:,1], color[:,:,2]

        # 对话框特征：中间偏下的白色矩形区域
        # 搜索范围：屏幕中心区域
        dialog_region = color[int(sh*0.25):int(sh*0.65), int(sw*0.2):int(sw*0.8)]
        white_mask = (
            (dialog_region[:,:,0] > 180) & (dialog_region[:,:,1] > 180) & (dialog_region[:,:,2] > 180)
            |
            (dialog_region[:,:,0] > 200) & (dialog_region[:,:,1] > 200) & (dialog_region[:,:,2] > 200)
        )

        if white_mask.sum() < 5000:  # 至少5000个白色像素（足够大的对话框）
            return None

        # 白色区域必须成团（连通），不能散落
        ys, xs = np.where(white_mask)
        y_min = int(sh*0.25) + int(ys.min())
        y_max = int(sh*0.25) + int(ys.max())
        x_min = int(sw*0.2) + int(xs.min())
        x_max = int(sw*0.2) + int(xs.max())

        dw, dh = x_max - x_min, y_max - y_min
        if dw < 200 or dh < 80:  # 对话框至少200x80像素
            return None
        if dw > sw * 0.7 or dh > sh * 0.5:  # 不能太大（排除全屏白色页面）
            return None

        # === 第二步：验证周围有遮罩（对话框外围区域偏暗） ===
        # 检查对话框上方30像素的像素是否偏暗
        above_y = max(0, y_min - 30)
        above_region = color[above_y:y_min, int(sw*0.3):int(sw*0.7)]
        dark_pixels = ((above_region[:,:,0] < 80) & (above_region[:,:,1] < 80) & (above_region[:,:,2] < 80))
        has_overlay = above_region.size > 0 and dark_pixels.sum() / above_region.size > 0.3

        if not has_overlay:
            return None

        # === 第三步：在对话框容器内搜索确定按钮 ===
        # 按钮通常在对话框的右下部分
        btn_region = (x_min, y_min + dh//2, x_max, y_max)
        r = ImageUtils.match(gray, tmpl, btn_region)

        if r:
            cx, cy, score = r
            if self._check_blue(color, cx, cy, tw, th, sh, sw):
                return (cx, cy, score)

        # 兜底：在整个对话框区域内搜索
        r2 = ImageUtils.match(gray, tmpl, (x_min, y_min, x_max, y_max))
        if r2:
            cx, cy, score = r2
            if self._check_blue(color, cx, cy, tw, th, sh, sw):
                return (cx, cy, score)

        return None

    def find_incomplete(self, color, gray, sw, sh, done_positions=None):
        """
        检测'未完成'课程卡片
        逐字对比法：只匹配第一个字（'未' vs '已'），大幅提高准确率
        """
        t_wei = self.tmpl.incomplete_txt   # '未' 字模板 (22x16)
        t_yi  = self.tmpl.yiwancheng_txt   # '已' 字模板 (23x18)
        if not t_wei or not t_yi: return None

        r, g, b = color[:,:,0], color[:,:,1], color[:,:,2]
        blue_m = (b>160) & (r<120) & (g<180)

        # 课程卡片区域
        x1, x2 = int(sw*0.10), int(sw*0.90)
        y1, y2 = int(sh*0.35), int(sh*0.85)
        blue_m[:y1,:] = blue_m[y2:,:] = blue_m[:,:x1] = blue_m[:,x2:] = False

        bys, bxs = np.where(blue_m)
        if len(bxs) < 50: return None

        # 聚类蓝色标签
        uniq_bx = sorted(set(int(x) for x in bxs))
        blue_clusters = []
        cur = [uniq_bx[0]]
        for x in uniq_bx[1:]:
            if x - cur[-1] < 12: cur.append(x)
            else:
                if len(cur) > 3: blue_clusters.append((cur[0], cur[-1]))
                cur = [x]
        if len(cur) > 3: blue_clusters.append((cur[0], cur[-1]))

        candidates = []
        for bx1, bx2 in blue_clusters:
            bw = bx2 - bx1
            if bw < 40 or bw > 160: continue
            col = bxs[(bxs>=bx1)&(bxs<=bx2)]
            if len(col) < 20: continue
            cys = bys[(bxs>=bx1)&(bxs<=bx2)]
            by_min, by_max = cys.min(), cys.max()
            bh = by_max - by_min
            if bh < 10 or bw/max(bh,1) < 1.3: continue

            cy = (by_min + by_max) // 2

            # 在蓝色标签左侧搜索第一个字
            left_region = (max(0,bx1-200), max(0,cy-20), bx1, min(sh,cy+20))

            # 匹配'未'字
            r_wei = ImageUtils.match(gray, t_wei, left_region)
            score_wei = r_wei[2] if r_wei else -1

            # 匹配'已'字
            r_yi = ImageUtils.match(gray, t_yi, left_region)
            score_yi = r_yi[2] if r_yi else -1

            # 判定逻辑：两者都有时需要明确差距
            if score_wei < 0.40:
                continue  # '未'字不匹配
            if score_yi > 0 and score_wei - score_yi < 0.08:
                continue  # 差异不够大，不确定
            if score_yi > 0 and score_yi > score_wei:
                continue  # '已'字得分更高 → 是"已完成"

            # 通过！点击位置
            gx = bx1 - 100
            click_y = by_min - 30

            if done_positions:
                too_close = False
                for dx, dy in done_positions:
                    if abs(gx-dx) < 80 and abs(click_y-dy) < 80:
                        too_close = True; break
                if too_close: continue

            candidates.append((gx, click_y, score_wei))

        if not candidates: return None
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

    def find_pagination(self, gray, color, sw, sh):
        """检测页面底部的翻页栏，返回下一页按钮坐标"""
        t = self.tmpl.pagination
        if not t: return None
        _, tw, th, _, _ = t
        # 搜索屏幕底部40%
        r = ImageUtils.match(gray, t, (0, int(sh*0.55), sw, sh))
        if not r: return None
        mx, my, _ = r
        # 在匹配区域内找蓝色数字按钮，取最右边那个（下一页）
        y1 = max(0, my - th//2)
        x1 = max(0, mx - tw//2)
        y2 = min(sh, y1 + th)
        x2 = min(sw, x1 + tw)
        patch = color[y1:y2, x1:x2]
        blue = (patch[:,:,2] > 150) & (patch[:,:,0] < 120)
        if not blue.any(): return None
        ys, xs = np.where(blue)
        # 找蓝色像素簇
        sorted_x = sorted(set(int(x) for x in xs))
        clusters, cur = [], [sorted_x[0]]
        for x in sorted_x[1:]:
            if x - cur[-1] < 15: cur.append(x)
            else:
                if len(cur) > 3: clusters.append(sum(cur)//len(cur))
                cur = [x]
        if len(cur) > 3: clusters.append(sum(cur)//len(cur))
        if not clusters: return None
        # 点击最右边的蓝色数字（下一页或>箭头）
        rightmost_x = x1 + clusters[-1]
        click_y = y1 + int(ys.mean())
        return (rightmost_x, click_y)

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
        """右键点击屏幕中央获取焦点，右键不会触发页面导航"""
        sw, sh = pyautogui.size()
        pyautogui.rightClick(int(sw * 0.5), int(sh * 0.5))
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
        self.done_positions = set()   # 已完成的卡片位置: {(gx, gy), ...}
        self.last_clicked = None      # 最近一次点击的卡片位置
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
            cx, cy, score = confirm
            # 避免重复点击同一个位置（10像素内）
            same_spot = self.last_clicked and abs(cx - self.last_clicked[0]) < 10 and abs(cy - self.last_clicked[1]) < 10
            if same_spot:
                log(f"  弹窗 [{cx},{cy}] 与上次位置相同，跳过")
            else:
                log(f"第{self.scan_count}轮 [{_STATE_NAMES[self.state]}] 弹窗 [{cx},{cy}]")
                self._do_click(img, cx, cy, score, 'confirm')
                self.last_clicked = (cx, cy)
                time.sleep(5)
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

        # 自检后才刷新——确保只在课程列表页刷新
        if self.needs_refresh and self.state == State.COURSE:
            self.needs_refresh = False
            log("刷新页面(F5)，等待10秒...")
            ActionExecutor.focus_browser()
            pyautogui.press('f5')
            time.sleep(10)
            ActionExecutor.scroll_down()
            # 刷新后重新截图
            img, color, gray = ImageUtils.screenshot()
            sw, sh = img.size

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
        has_course = self.detector.find_incomplete(color, gray, sw, sh, self.done_positions) is not None

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
        inc = self.detector.find_incomplete(color, gray, sw, sh, self.done_positions)
        if inc:
            gx, gy = inc
            self.last_clicked = (gx, gy)  # 记录本次点击位置
            self._do_click(img, gx, gy, 0.8, 'incomplete')
            time.sleep(2)
            _, c2, g2 = ImageUtils.screenshot()
            sw2, sh2 = c2.shape[1], c2.shape[0]
            close = self.detector.find_close(c2, g2, sw2, sh2)
            if close:
                # 成功进入课程 → 标记此位置为已完成
                self.done_positions.add((gx, gy))
                log(f"  卡片 ({gx},{gy}) 已加入完成列表")
                self.course_scrolls = 0
                self.state = State.VIDEO
                self.needs_refresh = False
                self.miss_count = 0
                log("  -> 视频")
                return
            else:
                # 点击了但没进入课程（误点击），也标记避免重试
                self.done_positions.add((gx, gy))
                log(f"  误点击 ({gx},{gy})，标记避免重试...")

        # 还没滚到底，一次性跳到底
        if self.course_scrolls == 0:
            log("  未找到未完成，跳转到页底...")
            ActionExecutor.focus_browser()
            pyautogui.press('end')
            time.sleep(1)
            self.course_scrolls = 1
            return

        # 在底部检查是否有未完成
        inc2 = self.detector.find_incomplete(color, gray, sw, sh, self.done_positions)
        if inc2:
            gx, gy = inc2
            self.last_clicked = (gx, gy)
            self._do_click(img, gx, gy, 0.8, 'incomplete')
            time.sleep(2)
            _, c2, g2 = ImageUtils.screenshot()
            sw2, sh2 = c2.shape[1], c2.shape[0]
            if self.detector.find_close(c2, g2, sw2, sh2):
                self.done_positions.add((gx, gy))
                log(f"  卡片 ({gx},{gy}) 已加入完成列表")
                self.course_scrolls = 0
                self.state = State.VIDEO
                self.needs_refresh = False
                self.miss_count = 0
                log("  -> 视频")
                return
            else:
                self.done_positions.add((gx, gy))
                log(f"  误点击 ({gx},{gy})，标记避免重试...")

        # 底部也没有 → 检查是否有翻页栏
        pg = self.detector.find_pagination(gray, color, sw, sh)
        if pg:
            px, py = pg
            log(f"  点击翻页 ({px},{py})")
            ActionExecutor.click(px, py)
            self.clicks += 1
            self.course_scrolls = 0
            time.sleep(3)
            return

        # 没有翻页栏 → 回顶部逐步向下找选项卡
        log("  无翻页，回顶部找选项卡...")
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
                cx2, cy2, score2 = confirm
                log(f"  链式点击弹窗 ({cx2},{cy2})")
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
        elif name == 'confirm':
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
