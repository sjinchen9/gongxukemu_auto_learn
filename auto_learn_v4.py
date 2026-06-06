# -*- coding: utf-8 -*-
"""
自动学习脚本 v4.0 — OCR驱动版
v3 全部架构 + PaddleOCR 文字识别替换 NCC/颜色检测
运行: py -3.11 auto_learn_v4.py
"""

import sys
if sys.version_info.major != 3 or sys.version_info.minor != 11:
    print("=" * 50)
    print("错误: 必须用 Python 3.11 运行此脚本")
    print(f"当前: Python {sys.version}")
    print("正确命令: py -3.11 auto_learn_v4.py")
    print("=" * 50)
    input("按回车退出...")
    sys.exit(1)

import time, os, threading, ctypes, msvcrt
import pyautogui
from PIL import Image
import numpy as np

# ======================== 窗口调整 ========================
def minimize_console():
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            cw, ch = 700, 180
            x = (screen_w - cw) // 2
            y = 0
            user32.SetWindowPos(hwnd, 0, x, y, cw, ch, 0)
    except: pass

# ======================== 配置 ========================
class Config:
    SCAN_VIDEO    = 60
    SCAN_OTHER    = 10
    SCAN_INITIAL  = 15
    INITIAL_SCANS = 3
    MAX_CLICKS    = 999  # 足够挂一晚上
    MISS_THRESHOLD = 3
    SCROLL_LIMIT   = 5

# ======================== 工具函数 ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(SCRIPT_DIR, "screenshots")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

class Logger:
    """双输出日志：终端 + 文件"""
    _log_file = None
    _log_path = None

    @classmethod
    def init(cls, username=""):
        if cls._log_file:
            cls._log_file.close()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{username}_" if username else ""
        cls._log_path = os.path.join(LOGS_DIR, f"{name}{timestamp}.log")
        cls._log_file = open(cls._log_path, 'w', encoding='utf-8')

    @classmethod
    def get_path(cls):
        return cls._log_path

    @classmethod
    def write(cls, msg):
        print(msg)
        if cls._log_file:
            cls._log_file.write(msg + '\n')
            cls._log_file.flush()

    @classmethod
    def close(cls):
        if cls._log_file:
            cls._log_file.close()

def log(msg):
    Logger.write(f"[{time.strftime('%H:%M:%S')}] {msg}")

def check_key():
    if msvcrt.kbhit():
        try:
            k = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if k == 's': return 'manual'
            if k == 'a': return 'auto'
            if k == 'q': return 'quit'
        except: pass
    return None

# ======================== OCR 引擎 ========================
import os as _os
import sys as _sys

# 彻底关闭 PaddleOCR/PaddlePaddle 日志
_os.environ.setdefault('GLOG_minloglevel', '3')
_os.environ.setdefault('GLOG_v', '0')
_os.environ.setdefault('FLAGS_logtostderr', '0')
_os.environ.setdefault('FLAGS_v', '0')
_os.environ.setdefault('FLAGS_minloglevel', '3')
_os.environ.setdefault('DISABLE_PADDLE_LOG', '1')

# 重定向stderr 吃掉Paddle的C++日志
import io
_stderr_backup = _sys.stderr
_sys.stderr = io.StringIO()

try:
    from paddleocr import PaddleOCR
finally:
    _sys.stderr = _stderr_backup  # 恢复stderr

import warnings
warnings.filterwarnings('ignore')

# 抑制ppocr的Python logger
import logging
for _mod_name in ['ppocr', 'paddleocr', 'paddlex', 'paddle']:
    logging.getLogger(_mod_name).setLevel(logging.CRITICAL + 1)

class OCREngine:
    """PaddleOCR 封装，全局单例"""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from paddleocr import PaddleOCR
            log("初始化 PaddleOCR...")
            cls._instance = PaddleOCR(lang='ch')
            cls._instance.ocr(np.zeros((50, 200, 3), dtype=np.uint8))
            log("OCR 就绪")
        return cls._instance

    @staticmethod
    def scan(img_array):
        """截屏OCR，返回 [{text, cx, cy, conf}, ...]"""
        ocr = OCREngine.get()
        result = ocr.ocr(img_array)
        if not result or not result[0]:
            return []
        texts = []
        for line in result[0]:
            box, (text, conf) = line
            if conf < 0.5: continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            texts.append({
                'text': text.strip(),
                'cx': int(sum(xs)/4),
                'cy': int(sum(ys)/4),
                'conf': conf,
            })
        return texts

# ======================== 截图工具 ========================
class ImageUtils:
    @staticmethod
    def screenshot():
        img = pyautogui.screenshot()
        return img, np.array(img)

    @staticmethod
    def load(path):
        """加载模板图片（NCC匹配用）"""
        if not os.path.exists(path): return None
        img = Image.open(path)
        arr = np.array(img)
        gray = np.mean(arr[:,:,:3], axis=2).astype(np.float32)
        return gray, gray.shape[1], gray.shape[0], gray.mean(), gray.std()

    @staticmethod
    def save(img, cx, cy, score, name):
        pad = 80
        r = img.crop((max(0,cx-pad), max(0,cy-pad), min(img.width,cx+pad), min(img.height,cy+pad)))
        fn = f"{name}_pos({cx},{cy})_s{score:.2f}.png"
        r.save(os.path.join(SCREENSHOTS_DIR, fn))
        return fn

    @staticmethod
    def dark_ratio(color, x1=0.15, y1=0.30, x2=0.85, y2=0.70):
        """统计区域内暗像素比例"""
        h, w = color.shape[:2]
        r = color[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
        dark = (r[:,:,0]<50)&(r[:,:,1]<50)&(r[:,:,2]<50)
        return dark.sum() / dark.size

# ======================== 检测器（OCR版） ========================
class Detector:
    def __init__(self):
        self._cache = []       # 当前轮OCR结果缓存
        self._cache_sw = 0
        self._cache_sh = 0
        # 翻页栏模板（NCC匹配，从v3继承的可靠方案）
        self.pagination_tmpl = ImageUtils.load(os.path.join(SCRIPT_DIR, "qianwangdijiye.png"))

    def scan(self, img_array):
        """执行OCR并缓存结果"""
        self._cache = OCREngine.scan(img_array)
        self._cache_sw = img_array.shape[1]
        self._cache_sh = img_array.shape[0]
        return self._cache

    def _all(self):
        return self._cache

    def _find_texts(self, keywords, region=None, exact=False):
        """在缓存中搜索关键词。exact=True时精确匹配（不再做包含判断）"""
        results = []
        for t in self._cache:
            matched = False
            for kw in keywords:
                if exact:
                    if kw == t['text']:
                        matched = True; break
                else:
                    if kw in t['text']:
                        matched = True; break
            if not matched: continue

            sw, sh = self._cache_sw, self._cache_sh
            if region:
                rx1, ry1, rx2, ry2 = region
                x = t['cx']
                y = t['cy']
                if not (rx1*sw <= x <= rx2*sw and ry1*sh <= y <= ry2*sh):
                    continue
            results.append(t)
        return results

    def _find_first(self, keywords, region=None):
        r = self._find_texts(keywords, region)
        return r[0] if r else None

    def has_page(self, color=None):
        """检测是否在培训页。OCR关键词 + 颜色兜底（保留v3逻辑）"""
        # OCR检测
        keywords = ('下一节','关闭','确定','确认','未完成','已完成','必修','选修','章节')
        ocr_ok = len(self._find_texts(keywords)) > 0 or len(self._cache) > 20

        if ocr_ok: return True

        # 颜色兜底（和v3完全一致）：蓝色标题栏或大面积暗色
        if color is not None:
            sh, sw = color.shape[:2]
            r, g, b = color[:,:,0], color[:,:,1], color[:,:,2]
            # 蓝色标题栏
            blue_header = (b > 150) & (r < 100) & (g < 150)
            has_blue = blue_header[:int(sh*0.15), :].sum() > 1000
            # 画面暗色比例
            center = color[int(sh*0.2):int(sh*0.8), int(sw*0.1):int(sw*0.8)]
            dark = (center[:,:,0]<50) & (center[:,:,1]<50) & (center[:,:,2]<50)
            has_dark = dark.sum() / dark.size > 0.2
            return has_blue or has_dark

        return False

    def find_next(self):
        """下一节按钮"""
        return self._find_texts(('下一节',), region=(0,0.12, 1,0.88))

    def find_close(self):
        """关闭按钮（右上角）"""
        return self._find_first(('关闭',), region=(0.85, 0, 1, 0.2))

    def find_confirm(self, color=None):
        """
        弹窗确定按钮（中部区域）。
        需同时验证弹窗上下文: 白色对话框 + 遮罩。
        可选：颜色验证按钮位置确实是蓝色按钮。
        """
        results = self._find_texts(('确定','确认'), region=(0.2, 0.25, 0.8, 0.75))
        if not results: return []
        # 颜色验证：匹配位置周围必须是蓝色按钮
        if color is not None:
            sh, sw = color.shape[:2]
            verified = []
            for t in results:
                cx, cy = t['cx'], t['cy']
                # 检查局部区域是否有蓝色像素（按钮特征）
                y1 = max(0, cy-15); y2 = min(sh, cy+15)
                x1 = max(0, cx-40); x2 = min(sw, cx+40)
                patch = color[y1:y2, x1:x2]
                blue = (patch[:,:,2] > 150) & (patch[:,:,0] < 120)
                if blue.sum() / patch.size > 0.08:
                    verified.append(t)
            return verified
        return results

    def find_incomplete(self, done_positions=None):
        """
        未完成课程卡片。
        严格排除已完成：只有'未完成'分数远超'已完成'时才接受
        """
        incs = self._find_texts(('未完成',), region=(0.10, 0.35, 0.90, 0.85))
        dones = self._find_texts(('已完成', '已学完', '已通过'), region=(0.10, 0.35, 0.90, 0.85), exact=True)

        # 全局校验：如果已完成数量远超未完成，说明OCR在误读，直接放弃
        if len(dones) > len(incs) + 1 and len(incs) < 2:
            return None

        done_pos = set((t['cx'], t['cy']) for t in dones)

        candidates = []
        for t in incs:
            # 检查附近200px水平+80px垂直内是否有已完成标记
            too_close = False
            for dx, dy in done_pos:
                if abs(t['cx']-dx) < 200 and abs(t['cy']-dy) < 80:
                    too_close = True; break
            if too_close: continue

            # 排除已点击过
            if done_positions:
                skip = False
                for dx, dy in done_positions:
                    if abs(t['cx']-dx) < 100 and abs(t['cy']-dy) < 100:
                        skip = True; break
                if skip: continue

            candidates.append(t)

        if not candidates: return None
        candidates.sort(key=lambda t: t['cy'])
        return (candidates[0]['cx'], candidates[0]['cy'])

    def find_tabs(self):
        """必修课/选修课 选项卡"""
        tabs = self._find_texts(('必修课','选修课','必修','选修'), region=(0.05, 0.10, 0.95, 0.45))
        return [(t['cx'], t['cy']) for t in tabs] if tabs else []

    def find_pagination(self, color, sw, sh):
        """
        翻页栏检测（v3 NCC版，已验证可靠）。
        用模板匹配定位翻页栏区域，再在其中找蓝色数字作为下一页。
        """
        t = self.pagination_tmpl
        if not t: return None
        _, tw, th, _, _ = t
        gray = np.mean(color[:,:,:3], axis=2).astype(np.float32)
        # 搜索屏幕底部
        r = self._match(gray, t, (0, int(sh*0.55), sw, sh))
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
        ys_list, xs_list = np.where(blue)
        # 找蓝色像素簇（按X坐标分组）
        sorted_x = sorted(set(int(x) for x in xs_list))
        clusters = []
        cur = [sorted_x[0]]
        for x in sorted_x[1:]:
            if x - cur[-1] < 15: cur.append(x)
            else:
                if len(cur) > 3:
                    cx = sum(cur) // len(cur)
                    # 计算这个簇的蓝色强度
                    cluster_blue_count = sum(1 for bx in xs_list if abs(bx - cx) < 10)
                    clusters.append((cx, cluster_blue_count))
                cur = [x]
        if len(cur) > 3:
            cx = sum(cur) // len(cur)
            cluster_blue_count = sum(1 for bx in xs_list if abs(bx - cx) < 10)
            clusters.append((cx, cluster_blue_count))
        if len(clusters) < 2: return None

        # 按X坐标排序
        clusters.sort(key=lambda c: c[0])
        # 找到蓝色最强的簇（当前页码高亮）
        max_blue_idx = max(range(len(clusters)), key=lambda i: clusters[i][1])
        # 点击当前页右边的下一个数字
        next_idx = max_blue_idx + 1
        if next_idx < len(clusters):
            target_x = x1 + clusters[next_idx][0]
        else:
            # 没有下一个数字，用最右边的（>符号或最后一页）
            target_x = x1 + clusters[-1][0]
        click_y = y1 + int(ys_list.mean())
        return (target_x, click_y)

    @staticmethod
    def _match(screen_gray, tmpl, region=None):
        """NCC模板匹配（从v3搬过来，用于翻页栏）"""
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
        s = 8
        for y in range(0, rh - th + 1, s):
            for x in range(0, rw - tw + 1, s):
                p = sub[y:y+th, x:x+tw]
                ps = p.std()
                if ps < 15 or t_std < 15: continue
                score = ((p - p.mean()) / ps * (t_gray - t_mean) / t_std).mean()
                if score > best_s:
                    best_s, bx, by = score, x, y
        if best_s < 0.45: return None
        return ox + bx + tw//2, oy + by + th//2, best_s

    def find_completion_status(self):
        """检测学分完成状态，返回匹配到的文字或None"""
        for t in self._cache:
            text = t['text']
            if '还需' in text and ('必修' in text or '选修' in text):
                return text
        return None

    def find_video_time(self):
        """检测视频播放器时间轴。格式: 08:11/11:03"""
        for t in self._cache:
            text = t['text']
            if '/' not in text or ':' not in text:
                continue
            clean = text.replace(' ', '').replace('|', '').replace('l', '').replace('L', '')
            if len(clean) > 25:
                continue
            # 只要有 / 和 : ，且长度合理，就是时间轴
            return clean
        return None

    def find_username(self):
        """找'退出'文字左侧最近的一个短中文词（2-4字），大概率是姓名"""
        # 先找"退出"
        tuichu = None
        for t in self._cache:
            if t['text'].strip() == '退出':
                tuichu = t
                break
        if not tuichu:
            return None

        # 在"退出"左侧找最近的2-4字纯中文
        best = None
        best_dist = 9999
        for t in self._cache:
            text = t['text'].strip()
            if not (2 <= len(text) <= 4):
                continue
            if not all('一' <= c <= '鿿' for c in text):
                continue
            # 必须在"退出"左侧且Y坐标接近
            if t['cx'] >= tuichu['cx'] or abs(t['cy'] - tuichu['cy']) > 30:
                continue
            dist = tuichu['cx'] - t['cx']
            if dist < best_dist:
                best_dist = dist
                best = text
        return best or ""

# ======================== 动作执行器（v3原样保留） ========================
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
    def press_key(key):
        """键盘快捷键（End/Home/F5），需要先点一下页面获取焦点"""
        sw, sh = pyautogui.size()
        pyautogui.click(int(sw * 0.5), int(sh * 0.5))
        time.sleep(0.3)
        pyautogui.press(key)
        time.sleep(0.5)

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
    VIDEO   = 0
    COURSE  = 1

_STATE_NAMES = ['视频', '选课']

# ======================== 主程序（v3架构完整保留） ========================
class AutoLearner:
    def __init__(self):
        self.state = None
        self.clicks = 0
        self.scan_count = 0
        self.miss_count = 0
        self.course_scrolls = 0
        self.tab_index = 0
        self.needs_refresh = False
        self.done_positions = set()
        self.last_clicked = None
        self.manual = False
        self.detector = Detector()
        self.username = ""
        self.credits_done = False
        self.courses_completed = 0

    def run(self):
        self._start_time = time.time()
        minimize_console()
        self._scan_all_texts = []
        # 先打印到终端，暂不写文件
        print(f"[{time.strftime('%H:%M:%S')}] " + "=" * 50)
        print(f"[{time.strftime('%H:%M:%S')}]   自动学习脚本 v4.0（OCR驱动，v3架构）")
        print(f"[{time.strftime('%H:%M:%S')}] " + "=" * 50)
        print("3秒后开始扫描...")
        time.sleep(3)
        self._scan()
        # 拿到用户名后初始化日志
        name = self.username or ""
        Logger.init(name)
        # 把头部信息写入日志文件
        log("=" * 50)
        log(f"  自动学习脚本 v4.0（OCR驱动，v3架构）- 用户: {name}")
        log("=" * 50)
        log("  s=手动扫描  a=自动模式  q=退出")
        log(f"  截图保存: {SCREENSHOTS_DIR}")
        log(f"  日志保存: {Logger.get_path()}")

        while self.clicks < Config.MAX_CLICKS and not self.credits_done:
            if self.manual:
                self._scan()
                if self.credits_done: break
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
        if self.credits_done:
            log("★ 全部学分已完成，脚本自动退出")
        elapsed = time.time() - self._start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        log(f"总耗时: {hours}小时{mins}分{secs}秒")
        log(f"用户名: {self.username}")
        log(f"日志已保存: {Logger.get_path()}")

        # 导出全部OCR文字
        if self._scan_all_texts:
            log_path = Logger.get_path() or ""
            base = os.path.splitext(os.path.basename(log_path))[0] if log_path else f"ocr_dump_{time.strftime('%Y%m%d_%H%M%S')}"
            ocr_dump_path = os.path.join(LOGS_DIR, f"{base}_ocr.txt")
            with open(ocr_dump_path, 'w', encoding='utf-8') as f:
                f.write(f"OCR全量文字导出 - {self.username}\n")
                f.write(f"总轮数: {self.scan_count} | 总点击: {self.clicks}\n")
                f.write(f"总耗时: {hours}h{mins}m{secs}s\n")
                f.write("=" * 50 + "\n\n")
                for entry in self._scan_all_texts:
                    f.write(f"[轮次{entry['round']}] {', '.join(entry['texts'][:30])}\n")
            log(f"OCR全量: {ocr_dump_path}")

        Logger.close()

    def _get_interval(self):
        if self.state is None or self.scan_count < Config.INITIAL_SCANS:
            return Config.SCAN_INITIAL
        return Config.SCAN_VIDEO if self.state == State.VIDEO else Config.SCAN_OTHER

    def _scan(self):
        if self.credits_done:
            return
        self.scan_count += 1

        img, color = ImageUtils.screenshot()
        sw, sh = img.size

        # OCR扫描
        texts = self.detector.scan(color)
        # 收集OCR全量文字用于最终导出
        self._scan_all_texts.append({
            'round': self.scan_count,
            'texts': [t['text'] for t in texts]
        })

        if not self.detector.has_page(color):
            log(f"第{self.scan_count}轮: 未检测到培训页面，跳过")
            return

        # 状态未知时先检测
        if self.state is None:
            self.state = self._detect_state()
            log(f"第{self.scan_count}轮 启动检测: [{_STATE_NAMES[self.state]}]")
            if self.state == State.COURSE:
                self.needs_refresh = True

        # 学分完成状态检测（每轮都检查）
        status_text = self.detector.find_completion_status()
        if status_text:
            log(f"  [学分状态] {status_text}")

        # 全局优先：弹窗确认
        confirms = self.detector.find_confirm(color)
        if confirms and self._verify_confirm_context(color, sw, sh, confirms):
            t = confirms[0] if isinstance(confirms, list) else confirms
            if isinstance(t, dict):
                cx, cy = t['cx'], t['cy']
            else:
                cx, cy, _ = t
            score = t.get('conf', 0.8) if isinstance(t, dict) else 0.8
            log(f"第{self.scan_count}轮 [{_STATE_NAMES[self.state]}] 弹窗 [{cx},{cy}]")
            self._do_click(img, cx, cy, score, 'confirm')
            time.sleep(5)
            self.needs_refresh = True
            return

        log(f"第{self.scan_count}轮 [{_STATE_NAMES[self.state]}] OCR={len(texts)}词")

        # 每轮自检
        close = self.detector.find_close()
        nexts = len(self.detector.find_next()) > 0
        dark_ratio = ImageUtils.dark_ratio(color)
        on_video = close is not None or nexts or dark_ratio > 0.3

        if on_video and self.state == State.COURSE:
            log(f"  自检修正: -> 视频")
            self.state = State.VIDEO
            self.needs_refresh = False
            self.miss_count = 0
        elif not on_video and self.state == State.VIDEO:
            log(f"  自检修正: -> 选课")
            self.state = State.COURSE
            self.course_scrolls = 0
            self.needs_refresh = True

        # 自检后才处理状态
        # 尝试获取用户名
        if not self.username:
            uname = self.detector.find_username()
            if uname:
                self.username = uname
                log(f"  用户: {self.username}")

        if self.state == State.VIDEO:
            self._do_video(img, color)
        elif self.state == State.COURSE:
            self._do_course(img, color)

    def _verify_confirm_context(self, color, sw, sh, confirms):
        """验证弹窗上下文：有白色对话框+遮罩。保留v3结构检测。"""
        # 检查白色对话框
        dialog = color[int(sh*0.25):int(sh*0.65), int(sw*0.2):int(sw*0.8)]
        white = (dialog[:,:,0]>180)&(dialog[:,:,1]>180)&(dialog[:,:,2]>180)
        if white.sum() < 5000: return False

        # 检查遮罩
        white_ys, white_xs = np.where(white)
        y_min = int(sh*0.25) + white_ys.min()
        above = color[max(0,y_min-30):y_min, int(sw*0.3):int(sw*0.7)]
        if above.size == 0: return False
        dark = (above[:,:,0]<80)&(above[:,:,1]<80)&(above[:,:,2]<80)
        return dark.sum()/above.size > 0.3

    def _detect_state(self):
        has_close = self.detector.find_close() is not None
        has_next = len(self.detector.find_next()) > 0
        has_course = self.detector.find_incomplete() is not None

        if has_close or has_next:
            return State.VIDEO
        if has_course:
            return State.COURSE

        # 使用dark_ratio兜底（和v3一致）
        img, color = ImageUtils.screenshot()
        dark_ratio = ImageUtils.dark_ratio(color)
        return State.VIDEO if dark_ratio > 0.3 else State.COURSE

    # ========== 状态处理（v3原样） ==========
    def _do_video(self, img, color):
        sw, sh = img.size

        # 视频时间轴识别（纯显示，不参与决策）
        vtime = self.detector.find_video_time()
        if vtime:
            log(f"  [视频进度] {vtime}")

        nexts = self.detector.find_next()
        if nexts:
            self.miss_count = 0
            t = nexts[0]
            cx, cy = t['cx'], t['cy']
            self._do_click(img, cx, cy, t['conf'], 'next')
            return

        # 视频播完判定：大面积全黑
        dark_ratio = ImageUtils.dark_ratio(color, x1=0.15, y1=0.30, x2=0.85, y2=0.70)
        video_ended = dark_ratio > 0.5

        if video_ended:
            self.miss_count = 0
            close = self.detector.find_close()
            if close:
                cx, cy = close['cx'], close['cy']
                log(f"  视频结束，点击关闭 ({cx},{cy})")
                self._do_click(img, cx, cy, close['conf'], 'close')
            else:
                log("  视频结束但未检测到关闭按钮，等待...")
            return

        self.miss_count = 0
        log(f"  视频播放中，等待下一节...")

    def _do_course(self, img, color):
        # 只在选课页尝试获取用户名
        if not self.username or self.username == '目☆':
            uname = self.detector.find_username()
            if uname and len(uname) >= 2:
                self.username = uname
                log(f"  用户: {self.username}")

        # === 优先级1: F5刷新（学完一个科目后先刷新） ===
        if self.needs_refresh:
            self.needs_refresh = False
            log("刷新页面(F5)，等待10秒...")
            ActionExecutor.press_key('f5')
            time.sleep(10)
            ActionExecutor.scroll_down()
            img, color = ImageUtils.screenshot()
            self.detector.scan(color)

        # === 优先级2: 学分状态检测（刷新后才查） ===
        status_text = self.detector.find_completion_status()
        if status_text:
            log(f"  [学分] {status_text}")
            # 检查是否全部完成
            if '0必修' in status_text and '0选修' in status_text:
                log("=" * 50)
                log("  ★ 全部学分已学完！")
                log("=" * 50)
                self.credits_done = True
                return

        # === 优先级3: 找未完成 ===
        inc = self.detector.find_incomplete(self.done_positions)
        if inc:
            gx, gy = inc
            self.last_clicked = (gx, gy)
            self._do_click(img, gx, gy, 0.8, 'incomplete')
            time.sleep(2)
            img2, color2 = ImageUtils.screenshot()
            self.detector.scan(color2)
            if self.detector.find_close():
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

        # 跳到底
        if self.course_scrolls == 0:
            log("  未找到未完成，跳转到页底...")
            ActionExecutor.press_key('end')
            time.sleep(1)
            self.course_scrolls = 1
            return

        # 底部再找
        img2, color2 = ImageUtils.screenshot()
        self.detector.scan(color2)
        inc2 = self.detector.find_incomplete(self.done_positions)
        if inc2:
            gx, gy = inc2
            self.last_clicked = (gx, gy)
            self._do_click(img2, gx, gy, 0.8, 'incomplete')
            time.sleep(2)
            img3, color3 = ImageUtils.screenshot()
            self.detector.scan(color3)
            if self.detector.find_close():
                self.done_positions.add((gx, gy))
                self.course_scrolls = 0
                self.state = State.VIDEO
                self.needs_refresh = False
                self.miss_count = 0
                log("  -> 视频")
                return
            else:
                self.done_positions.add((gx, gy))

        # 翻页栏（NCC模板匹配，来自v3的可靠方案）
        img4, color4 = ImageUtils.screenshot()
        sw4, sh4 = img4.size
        pg = self.detector.find_pagination(color4, sw4, sh4)
        if pg:
            px, py = pg
            log(f"  点击翻页 ({px},{py})")
            ActionExecutor.click(px, py)
            self.clicks += 1
            self.course_scrolls = 0
            self.done_positions.clear()  # 新页面，重置已完成记录
            time.sleep(3)
            # 翻页后验证：新页面是否有未完成课程
            img5, color5 = ImageUtils.screenshot()
            self.detector.scan(color5)
            inc3 = self.detector.find_incomplete(self.done_positions)
            if inc3:
                log("  翻页成功，检测到未完成课程")
            else:
                log("  翻页完成，但暂未检测到未完成")
            return

        # 回顶部找选项卡
        log("  无翻页，回顶部找选项卡...")
        ActionExecutor.press_key('home')
        time.sleep(1)

        tabs = None
        for offset in range(6):
            img5, color5 = ImageUtils.screenshot()
            self.detector.scan(color5)
            tabs = self.detector.find_tabs()
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
            self.done_positions.clear()  # 新选项卡，重置已完成记录
            time.sleep(4)
            # 切换后验证新页面
            try:
                img6, color6 = ImageUtils.screenshot()
                texts = self.detector.scan(color6)
                log(f"  选项卡已切换，OCR={len(texts)}词")
                # 列出相关文字
                rel = [t['text'] for t in texts
                       if any(kw in t['text'] for kw in ('未完成','已完成','必修','选修','学分'))]
                log(f"  相关词: {rel[:15]}")
                inc_search = self.detector.find_incomplete(self.done_positions)
                if inc_search:
                    log(f"  → 发现未完成课程")
            except Exception as e:
                log(f"  选项卡验证出错: {e}")
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

        # 点关闭后链式处理弹窗（重试3次，弹窗可能延迟出现）
        if name == 'close':
            for retry in range(3):
                time.sleep(2 + retry * 2)  # 2s, 4s, 6s 逐步等待
                img2, color2 = ImageUtils.screenshot()
                self.detector.scan(color2)
                confirms = self.detector.find_confirm(color2)
                if confirms:
                    t = confirms[0] if isinstance(confirms, list) else confirms
                    cx2, cy2 = (t['cx'], t['cy']) if isinstance(t, dict) else (t[:2] if isinstance(t, tuple) else (t.cx, t.cy))
                    log(f"  链式点击弹窗 ({cx2},{cy2}) [重试{retry+1}次]")
                    ActionExecutor.click(cx2, cy2)
                    self.clicks += 1
                    time.sleep(5)
                    self.needs_refresh = True
                    break
            else:
                log("  关闭后弹窗未出现（重试3次），下轮继续")

    def _verify_click(self, name, click_img=None):
        time.sleep(1.5)
        img2, color = ImageUtils.screenshot()
        self.detector.scan(color)

        ok = False
        if name == 'next':
            ok = ImageUtils.dark_ratio(color) > 0.2
        elif name == 'incomplete':
            ok = self.detector.find_close() is not None
        elif name == 'close':
            ok = len(self.detector.find_confirm()) > 0
        elif name == 'confirm':
            ok = len(self.detector.find_confirm()) == 0
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
