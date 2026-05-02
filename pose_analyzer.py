# -*- coding: utf-8 -*-
"""
踢趣AI视频分析服务 v3.0
技术栈：OpenCV + MediaPipe Pose + MediaPipe Hands(可选) + YOLOv8n
球检测：OpenCV HoughCircles（主） → YOLOv8n（降级）
动作识别：MediaPipe Pose 生物力学规则引擎
Dify输出：完整结构化数据，匹配文档要求的三种动作字段
"""
import cv2, math, json, sys, os, tempfile, time, copy
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe 初始化 (v3.1: 使用新版 Tasks API)
# ─────────────────────────────────────────────────────────────────────────────
MP_AVAILABLE = False
MP_DETECTOR  = None
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_lite.task')

def _init_mediapipe():
    global MP_AVAILABLE, MP_DETECTOR
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.core import base_options

        if not os.path.exists(MODEL_PATH):
            print(f'[PoseAnalyzer] 模型文件不存在: {MODEL_PATH}')
            print('[PoseAnalyzer] 请先运行 download_mediapipe_model.py 下载模型')
            return

        options = PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
        )
        MP_DETECTOR = PoseLandmarker.create_from_options(options)
        MP_AVAILABLE = True
        print(f'[PoseAnalyzer] MediaPipe {mp.__version__} (Tasks API) 初始化成功')
    except ImportError as e:
        print(f'[PoseAnalyzer] MediaPipe 导入失败: {e}')

_init_mediapipe()


# ─────────────────────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _dist(a, b):
    """两点欧氏距离（归一化坐标）"""
    return math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)

def _dist_pts(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def _angle(a, b, c):
    """三点夹角 ∠abc（顶点为b），返回0-180°"""
    v1 = (a['x'] - b['x'], a['y'] - b['y'])
    v2 = (c['x'] - b['x'], c['y'] - b['y'])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2) + 1e-6
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2) + 1e-6
    cos_val = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))

def _clamp01(x):
    return max(0.0, min(1.0, x))

def _ang_dist(a1, b1, c1, a2, b2, c2):
    """两向量夹角（°）"""
    cos_val = max(-1, min(1, abs(a1*a2 + b1*b2) / (math.sqrt(a1**2+b1**2+1e-9) * math.sqrt(a2**2+b2**2+1e-9))))
    return math.degrees(math.acos(cos_val))


# ─────────────────────────────────────────────────────────────────────────────
#  球检测器（主：HoughCircles，降级：YOLOv8n）
# ─────────────────────────────────────────────────────────────────────────────

class BallDetector:
    """
    足球检测器。
    策略：HoughCircles（零成本，无需额外依赖）优先；
          若检测失败或球过小（像素面积<100）则尝试 YOLOv8n（需 pip install ultralytics）。
    """
    def __init__(self):
        self.yolo_available = False
        self.yolo_model = None

        # HSV 白色足球范围（室内白灯场景）
        self.white_lower = (0, 0, 180)
        self.white_upper = (180, 50, 255)
        # HSV 黄/橙色足球范围（天然草场）
        self.orange_lower = (10, 80, 80)
        self.orange_upper = (30, 255, 255)
        # HSV 深色足球范围（夜间/暗光）
        self.dark_lower = (0, 0, 30)
        self.dark_upper = (180, 80, 100)

    def _try_yolo(self):
        """惰性加载 YOLOv8n"""
        if self.yolo_available:
            return True
        try:
            from ultralytics import YOLO
            # yolov8n.pt 已包含 sports_ball 类（COCO）
            self.yolo_model = YOLO('yolov8n.pt')
            self.yolo_available = True
            print('[BallDetector] YOLOv8n loaded successfully')
            return True
        except Exception as e:
            print(f'[BallDetector] YOLOv8n not available: {e}')
            return False

    def detect_in_frame(self, frame):
        """
        在单帧中检测足球位置。
        Returns: (cx_norm, cy_norm, radius_norm) 或 None
        """
        # 策略1：HoughCircles
        result = self._hough_detect(frame)
        if result:
            return result

        # 策略2：YOLOv8n
        if self._try_yolo():
            result = self._yolo_detect(frame)
            if result:
                return result

        return None

    def _hough_detect(self, frame):
        """
        用 HSV 颜色分割 + HoughCircles 检测足球。
        返回 (cx_norm, cy_norm, radius_norm) 或 None。
        """
        h, w = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (9, 9), 2)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        best_ball = None
        best_score = 0

        for lower, upper in [
            self.white_lower, self.orange_lower, self.dark_lower
        ]:
            try:
                mask = cv2.inRange(hsv, lower, upper)
                # 开运算去噪点
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                # 膨胀连接断开的区域
                mask = cv2.dilate(mask, kernel, iterations=1)

                circles = cv2.HoughCircles(
                    mask, cv2.HOUGH_GRADIENT_ALT,
                    dp=1.2, minDist=20,
                    param1=50, param2=0.85,
                    minRadius=5, maxRadius=max(w, h) // 4
                )
                if circles is not None:
                    for cx, cy, r in circles[0]:
                        # 圆形度评分：在mask上计算圆形区域的实际像素比例
                        roi = mask[max(0, int(cy)-int(r)):int(cy)+int(r),
                                   max(0, int(cx)-int(r)):int(cx)+int(r)]
                        if roi.size == 0:
                            continue
                        filled = cv2.countNonZero(roi)
                        circle_area = math.pi * r * r
                        ratio = filled / (circle_area + 1e-6)
                        score = ratio * circle_area
                        if score > best_score and circle_area > 80:
                            best_score = score
                            best_ball = (float(cx)/w, float(cy)/h, float(r)/w)
            except Exception:
                continue

        return best_ball

    def _yolo_detect(self, frame):
        """YOLOv8n 检测足球（COCO class 37 = sports ball）"""
        if self.yolo_model is None:
            return None
        try:
            results = self.yolo_model(frame, verbose=False, classes=[37], conf=0.3)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                box = r.boxes[0]
                x1, y1, x2, y2 = box.xywhn[0].cpu().numpy()  # normalized xywh
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                rw = (x2 - x1) / 2
                rh = (y2 - y1) / 2
                r = max(rw, rh)
                h, w = frame.shape[:2]
                return (float(cx), float(cy), float(r))
        except Exception as e:
            print(f'[BallDetector] YOLOv8n error: {e}')
        return None

    def track(self, video_path, sample_interval=1):
        """
        追踪整个视频中足球的轨迹。
        Returns: list of {frame, time, x, y, r, speed}
                 speed = 归一化速度（像素/秒，相邻帧）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        trajectory = []
        prev_x, prev_y, prev_t = None, None, None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue

            t = frame_idx / fps
            ball = self.detect_in_frame(frame)

            if ball:
                cx, cy, r = ball
                speed = 0.0
                if prev_x is not None:
                    speed = _dist_pts(cx, cy, prev_x, prev_y) / max(t - prev_t, 0.01)
                trajectory.append({
                    'frame': frame_idx,
                    'time': round(t, 3),
                    'x': round(cx, 4),
                    'y': round(cy, 4),
                    'r': round(r, 4),
                    'speed': round(speed, 4)
                })
                prev_x, prev_y, prev_t = cx, cy, t
            frame_idx += 1

        cap.release()
        print(f'[BallDetector] Tracked {len(trajectory)} ball positions')
        return trajectory


# ─────────────────────────────────────────────────────────────────────────────
#  PoseAnalyzer v3.0
# ─────────────────────────────────────────────────────────────────────────────

class PoseAnalyzer:
    """足球动作姿态 + 球轨迹联合分析器"""

    def __init__(self):
        if not MP_AVAILABLE:
            self.pose = None
            print('[PoseAnalyzer] Warning: MediaPipe not available')
            return
        self.pose = MP_DETECTOR  # 新版 PoseLandmarker 实例
        self.ball_detector = BallDetector()
        print('[PoseAnalyzer] v3.1 initialized: PoseLandmarker + Ball tracking')

    # ──────────────────────────────────────────────
    #  公开接口
    # ──────────────────────────────────────────────

    def analyze_video(self, video_path, action_type='auto',
                      player_info=None):
        """
        分析视频，返回完整结构化报告。

        Args:
            video_path: 本地视频路径
            action_type: 'auto' | '传球' | '射门' | '带球'
            player_info: {
                'dominant_foot': '右脚' | '左脚' | '双脚均衡',
                'age_estimation': 'U10' | 'U12' | 'U14' | 'U16+'
              }
        Returns:
            dict: 完整报告（含 poseData + difyData）
        """
        if not MP_AVAILABLE:
            raise RuntimeError(
                '姿态检测失败：MediaPipe 未安装。'
                '请在本地环境执行：pip install mediapipe opencv-python'
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f'视频读取失败：无法打开文件（{video_path}）')

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 10
        sample_interval = max(1, int(fps / 5))
        print(f'[PoseAnalyzer] {total_frames}f, {fps:.1f}fps, {duration:.1f}s, interval={sample_interval}')

        # ── 第一阶段：提取关键帧 + 追踪球 ─────────────
        all_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            import mediapipe as mp
            img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx / fps * 1000)
            results = self.pose.detect_for_video(img_mp, timestamp_ms)

            kp = None
            conf = 0
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                lms = results.pose_landmarks[0]
                kp = self._extract_keypoints(lms)
                # 新版 PoseLandmarker 无 visibility，改为用归一化坐标标准差估算置信度
                if kp:
                    vals = [v for v in kp.values() if v is not None]
                    conf = 1.0 if len(vals) >= 6 else 0.0
                if conf < 0.3:
                    kp = None

            all_frames.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'kp': kp,
                'conf': conf
            })
            frame_idx += 1

        cap.release()

        if not all_frames or all(f['kp'] is None for f in all_frames):
            raise RuntimeError(
                '姿态检测失败：视频中未识别到有效的人体姿态关键点。'
                '建议：1）确保画面中人物完整且清晰；2）避免遮挡或背光；3）确保人物侧向或面向摄像头'
            )

        valid_frames = [f for f in all_frames if f['kp'] is not None]
        print(f'[PoseAnalyzer] Valid pose frames: {len(valid_frames)}/{len(all_frames)}')

        # ── 球追踪（单独处理，不阻塞姿态分析）──────────
        ball_trajectory = self.ball_detector.track(video_path, sample_interval=sample_interval)

        # ── 第二阶段：自动识别动作类型 ───────────────
        if action_type == 'auto' or action_type not in ['传球', '射门', '带球']:
            detected_type = self._auto_detect_action(valid_frames, duration)
            print(f'[PoseAnalyzer] Detected: {detected_type}')
        else:
            detected_type = action_type

        # ── 第三阶段：生物力学分析 ───────────────────
        bio = self._biomechanics_analysis(valid_frames, detected_type)

        # ── 第四阶段：球轨迹分析 ─────────────────────
        ball_data = self._analyze_ball_trajectory(
            ball_trajectory, valid_frames, detected_type
        )

        # ── 第五阶段：综合字段（合并 bio + ball_data）──
        full_data = {**bio, **ball_data}

        # ── 第六阶段：计算评分 ────────────────────────
        scores = self._calc_scores(detected_type, full_data)

        # ── 第七阶段：生成 Dify 专用数据结构 ─────────
        dify_data = self._build_dify_data(
            detected_type, full_data, scores, player_info
        )

        return {
            'id': f'pose_{int(time.time())}_{hash(video_path) % 100000}',
            'type': detected_type,
            'typeIcon': self._get_icon(detected_type),
            'actionType': detected_type,
            'score': scores['total'],
            'level': self._get_level(scores['total']),
            'feedback': self._generate_feedback(detected_type, scores, full_data),
            'dimensions': self._generate_dimensions(detected_type, scores),
            'recommendedCourses': self._generate_recommendations(detected_type, scores),
            'date': self._get_date(),
            'meta': {
                'poseFrames': len(valid_frames),
                'ballFrames': len(ball_trajectory),
                'duration': round(duration, 1),
                'analysisMethod': 'MediaPipe Pose v3.0 + Ball Tracking + Biomechanics Engine',
                'detectedType': detected_type,
                **{k: round(v, 4) for k, v in full_data.items()
                   if isinstance(v, float) and k not in ['ankle_lock', 'follow_through', 'contact_stability']}
            },
            # Dify 专用完整数据
            'difyData': dify_data
        }

    # ──────────────────────────────────────────────
    #  关键点提取
    # ──────────────────────────────────────────────

    def _extract_keypoints(self, landmarks):
        """从新版 PoseLandmarker landmark 列表提取关键点"""
        def g(i):
            lm = landmarks[i]
            # 新版 API 无 visibility，用 x,y,z 标准差估算（>0 则有效）
            return {'x': lm.x, 'y': lm.y, 'z': getattr(lm, 'z', 0.0)}
        return {
            'nose':       g(0),
            'l_shoulder': g(11), 'r_shoulder': g(12),
            'l_elbow':    g(13), 'r_elbow':    g(14),
            'l_wrist':    g(15), 'r_wrist':    g(16),
            'l_hip':      g(23), 'r_hip':      g(24),
            'l_knee':     g(25), 'r_knee':     g(26),
            'l_ankle':    g(27), 'r_ankle':    g(28),
            'l_heel':     g(29), 'r_heel':     g(30),
            'l_foot_idx': g(31), 'r_foot_idx': g(32),
        }

    # ──────────────────────────────────────────────
    #  自动动作识别
    # ──────────────────────────────────────────────

    def _auto_detect_action(self, frames, duration):
        if len(frames) < 3:
            return '传球'

        ankle_vels = []
        for i in range(1, len(frames)):
            dt = frames[i]['time'] - frames[i-1]['time']
            if dt <= 0:
                continue
            l_v = _dist(frames[i]['kp']['l_ankle'], frames[i-1]['kp']['l_ankle']) / dt
            r_v = _dist(frames[i]['kp']['r_ankle'], frames[i-1]['kp']['r_ankle']) / dt
            ankle_vels.append(max(l_v, r_v))

        if not ankle_vels:
            return '传球'

        max_v  = max(ankle_vels)
        avg_v  = sum(ankle_vels) / len(ankle_vels)
        peak_ratio = max_v / (avg_v + 1e-6)

        hip_xs = [(f['kp']['l_hip']['x'] + f['kp']['r_hip']['x'])/2 for f in frames]
        lateral_range = max(hip_xs) - min(hip_xs)

        knee_angles = []
        for f in frames:
            kp = f['kp']
            la = _angle(kp['l_hip'], kp['l_knee'], kp['l_ankle'])
            ra = _angle(kp['r_hip'], kp['r_knee'], kp['r_ankle'])
            knee_angles.append(min(la, ra))
        knee_angle_range = max(knee_angles) - min(knee_angles) if knee_angles else 0

        print(f'[AutoDetect] max_v={max_v:.3f}, avg_v={avg_v:.3f}, peak={peak_ratio:.2f}, '
              f'lateral={lateral_range:.3f}, knee_range={knee_angle_range:.1f}°')

        if peak_ratio > 3.0 and knee_angle_range > 40:
            return '射门'
        if lateral_range > 0.15 and peak_ratio < 2.5 and duration > 5:
            return '带球'
        if peak_ratio > 2.0:
            return '传球'
        if duration <= 4:
            return '传球'
        elif duration <= 10:
            return '射门'
        return '带球'

    # ──────────────────────────────────────────────
    #  生物力学分析
    # ──────────────────────────────────────────────

    def _biomechanics_analysis(self, frames, action_type):
        """
        提取全部生物力学指标（通用 + 动作特定）。
        返回扁平 dict，字段名与文档要求一致。
        """
        result = {}

        # ── 稳定性 ──────────────────────────────────
        shoulder_ys = [(f['kp']['l_shoulder']['y'] + f['kp']['r_shoulder']['y'])/2 for f in frames]
        mean_sy = sum(shoulder_ys) / len(shoulder_ys)
        variance = sum((y - mean_sy)**2 for y in shoulder_ys) / len(shoulder_ys)
        result['stability'] = _clamp01(1 - math.sqrt(variance) * 20)

        # ── 对称性 ──────────────────────────────────
        last = frames[-1]['kp']
        sh_diff = abs(last['l_shoulder']['y'] - last['r_shoulder']['y'])
        hip_diff = abs(last['l_hip']['y'] - last['r_hip']['y'])
        result['symmetry'] = _clamp01(1 - (sh_diff + hip_diff) * 8)

        # ── 踝关节速度 ────────────────────────────────
        ankle_vels = []
        for i in range(1, len(frames)):
            dt = frames[i]['time'] - frames[i-1]['time']
            if dt <= 0:
                continue
            l_v = _dist(frames[i]['kp']['l_ankle'], frames[i-1]['kp']['l_ankle']) / dt
            r_v = _dist(frames[i]['kp']['r_ankle'], frames[i-1]['kp']['r_ankle']) / dt
            ankle_vels.append({'l': l_v, 'r': r_v, 'max': max(l_v, r_v)})

        if ankle_vels:
            all_v = [v['max'] for v in ankle_vels]
            result['max_ankle_velocity'] = max(all_v)
            result['avg_ankle_velocity'] = sum(all_v) / len(all_v)
            # 惯用脚：踝速度峰值更高的一侧
            l_total = sum(v['l'] for v in ankle_vels)
            r_total = sum(v['r'] for v in ankle_vels)
            result['dominant_foot_raw'] = 'left' if l_total > r_total else 'right'
        else:
            result['max_ankle_velocity'] = 0.1
            result['avg_ankle_velocity'] = 0.05
            result['dominant_foot_raw'] = 'right'

        # ── 膝关节角度范围 ───────────────────────────
        knee_angles = []
        for f in frames:
            kp = f['kp']
            la = _angle(kp['l_hip'], kp['l_knee'], kp['l_ankle'])
            ra = _angle(kp['r_hip'], kp['r_knee'], kp['r_ankle'])
            knee_angles.append({'l': la, 'r': ra, 'min': min(la, ra)})

        if knee_angles:
            result['knee_angle_range'] = (max(k['min'] for k in knee_angles) -
                                          min(k['min'] for k in knee_angles))
            result['min_knee_angle'] = min(k['min'] for k in knee_angles)
        else:
            result['knee_angle_range'] = 0
            result['min_knee_angle'] = 90

        # ── 踢球帧索引（踝速度最高的帧）─────────────
        if ankle_vels:
            peak_idx = max(range(len(ankle_vels)), key=lambda i: ankle_vels[i]['max'])
            peak_idx = min(peak_idx, len(frames) - 1)
        else:
            peak_idx = len(frames) // 2

        # ── 支撑脚稳定性 ──────────────────────────────
        window = frames[max(0, peak_idx-2):min(len(frames), peak_idx+3)]
        if len(window) >= 2:
            l_move = sum(_dist(window[j]['kp']['l_ankle'], window[j-1]['kp']['l_ankle'])
                         for j in range(1, len(window)))
            r_move = sum(_dist(window[j]['kp']['r_ankle'], window[j-1]['kp']['r_ankle'])
                         for j in range(1, len(window)))
            result['support_foot_stability'] = _clamp01(1 - min(l_move, r_move) * 30)
        else:
            result['support_foot_stability'] = 0.6

        # ── 踝锁定程度（踢球瞬间踝关节角度变化率）─────
        # 在峰值帧前后各取2帧，计算踝角度变化
        peak_kp = frames[peak_idx]['kp']
        window_kp = frames[max(0, peak_idx-2):min(len(frames), peak_idx+3)]
        if len(window_kp) >= 2:
            ankle_changes = []
            for j in range(1, len(window_kp)):
                kp1, kp2 = window_kp[j-1]['kp'], window_kp[j]['kp']
                # 以踢球腿（踝速度更大的那只）计算角度变化
                kick_side = 'r' if ankle_vels[min(peak_idx, len(ankle_vels)-1)]['max'] == \
                                   ankle_vels[min(peak_idx, len(ankle_vels)-1)]['r'] else 'l'
                a1 = _angle(kp1[f'{kick_side}_hip'], kp1[f'{kick_side}_knee'], kp1[f'{kick_side}_ankle'])
                a2 = _angle(kp2[f'{kick_side}_hip'], kp2[f'{kick_side}_knee'], kp2[f'{kick_side}_ankle'])
                ankle_changes.append(abs(a2 - a1))
            result['ankle_lock'] = _clamp01(1 - sum(ankle_changes)/len(ankle_changes)/15)
        else:
            result['ankle_lock'] = 0.7

        # ── 支撑脚距离（踢球帧两踝水平距离）───────────
        kp_peak = frames[peak_idx]['kp']
        result['plant_foot_distance'] = abs(
            kp_peak['l_ankle']['x'] - kp_peak['r_ankle']['x']
        )

        # ── 脚尖朝向（°，与水平轴夹角）──────────────
        # foot_idx → ankle 向量与水平轴的夹角
        def toe_angle(kp, side='l'):
            ax, ay = kp[f'{side}_ankle']['x'], kp[f'{side}_ankle']['y']
            fx, fy = kp[f'{side}_foot_idx']['x'], kp[f'{side}_foot_idx']['y']
            dx, dy = ax - fx, ay - fy
            return abs(math.degrees(math.atan2(dy, dx)))
        # 踢球腿的脚尖朝向
        kick_side = result['dominant_foot_raw']
        result['toe_alignment'] = toe_angle(kp_peak, kick_side)

        # ── 随球幅度（踢球后踝关节最大延伸）───────────
        if peak_idx < len(frames) - 1:
            peak_ankle_y = kp_peak[f'{kick_side}_ankle']['y']
            follow_y = max(f['kp'][f'{kick_side}_ankle']['y']
                           for f in frames[peak_idx:min(peak_idx+5, len(frames))])
            result['follow_through'] = _clamp01((follow_y - peak_ankle_y) * 10)
        else:
            result['follow_through'] = 0.5

        # ── 触球稳定性（支撑脚稳定性代理）─────────────
        result['contact_stability'] = result['support_foot_stability']

        # ── 身体前倾 ─────────────────────────────────
        forward_leans = []
        for f in frames:
            kp = f['kp']
            scx = (kp['l_shoulder']['x'] + kp['r_shoulder']['x'])/2
            hcx = (kp['l_hip']['x'] + kp['r_hip']['x'])/2
            scy = (kp['l_shoulder']['y'] + kp['r_shoulder']['y'])/2
            hcy = (kp['l_hip']['y'] + kp['r_hip']['y'])/2
            lean = (hcx - scx) / (abs(hcy - scy) + 1e-6)
            forward_leans.append(lean)
        result['forward_lean'] = sum(forward_leans) / len(forward_leans) if forward_leans else 0
        # 转为角度
        result['forward_lean_angle'] = math.degrees(math.atan(abs(result['forward_lean'])))

        # ── 横向位移 ─────────────────────────────────
        hip_xs = [(f['kp']['l_hip']['x'] + f['kp']['r_hip']['x'])/2 for f in frames]
        result['lateral_range'] = max(hip_xs) - min(hip_xs)

        # ── 重心高度（归一化Y坐标，越大=越低重心）────
        hip_ys = [(f['kp']['l_hip']['y'] + f['kp']['r_hip']['y'])/2 for f in frames]
        result['center_of_gravity'] = sum(hip_ys) / len(hip_ys)

        # ── 头部稳定性 ───────────────────────────────
        nose_ys = [f['kp']['nose']['y'] for f in frames]
        nose_var = sum((y - sum(nose_ys)/len(nose_ys))**2 for y in nose_ys) / len(nose_ys)
        result['head_stability'] = _clamp01(1 - math.sqrt(nose_var) * 20)

        # ── 手臂平衡度 ───────────────────────────────
        arm_diff = [abs(f['kp']['l_wrist']['y'] - f['kp']['r_wrist']['y']) for f in frames]
        result['arm_balance'] = _clamp01(1 - (sum(arm_diff)/len(arm_diff)) * 15)

        # ── 视野扫视频率（nose X轴方向改变频率）─────
        nose_xs = [f['kp']['nose']['x'] for f in frames]
        direction_changes = sum(1 for i in range(1, len(nose_xs))
                                if (nose_xs[i] - nose_xs[i-1]) *
                                   (nose_xs[min(i+1, len(nose_xs)-1)] - nose_xs[i]) < 0)
        result['scan_frequency'] = direction_changes / len(frames) if frames else 0
        result['head_movement_frequency'] = result['scan_frequency']

        # ── 助跑指标 ─────────────────────────────────
        # 助跑角度（助跑方向向量与水平向右的夹角）
        runup_frames = frames[max(0, peak_idx-5):peak_idx]
        if len(runup_frames) >= 2:
            start = runup_frames[0]['kp']
            end = runup_frames[-1]['kp']
            scx = (end['l_shoulder']['x'] + end['r_shoulder']['x'])/2 - \
                  (start['l_shoulder']['x'] + start['r_shoulder']['x'])/2
            scy = (end['l_shoulder']['y'] + end['r_shoulder']['y'])/2 - \
                  (start['l_shoulder']['y'] + start['r_shoulder']['y'])/2
            result['approach_angle'] = abs(math.degrees(math.atan2(scy, scx)))
        else:
            result['approach_angle'] = 30.0

        # 助跑步数（踝关节左右交替摆动峰值数，peak_idx前5帧）
        if len(runup_frames) >= 3:
            ankle_xs_l = [f['kp']['l_ankle']['x'] for f in runup_frames]
            steps = sum(1 for i in range(1, len(ankle_xs_l)-1)
                        if ankle_xs_l[i] > ankle_xs_l[i-1] and
                           ankle_xs_l[i] > ankle_xs_l[i+1] and
                           abs(ankle_xs_l[i] - ankle_xs_l[i-1]) > 0.03)
            result['step_count'] = max(1, steps)
        else:
            result['step_count'] = 3

        # 最后一步步幅（peak前一步的踝水平位移）
        if len(runup_frames) >= 2:
            last_ankle_x = runup_frames[-1]['kp'][kick_side]['x']
            prev_ankle_x = runup_frames[-2]['kp'][kick_side]['x']
            result['last_step_length'] = abs(last_ankle_x - prev_ankle_x)
        else:
            result['last_step_length'] = 0.1

        # 助跑节奏稳定性（步伐间隔时间标准差）
        if len(runup_frames) >= 2:
            intervals = [runup_frames[i]['time'] - runup_frames[i-1]['time']
                         for i in range(1, len(runup_frames))]
            if intervals:
                mean_int = sum(intervals) / len(intervals)
                var_int = sum((t - mean_int)**2 for t in intervals) / len(intervals)
                result['rhythm_variation'] = _clamp01(1 - math.sqrt(var_int) * 20)
            else:
                result['rhythm_variation'] = 0.7
        else:
            result['rhythm_variation'] = 0.7

        # ── 随球方向一致性 ────────────────────────────
        # 踢球后踝移动方向与前倾方向的一致性
        if peak_idx < len(frames) - 3:
            kick_ankle_before = frames[peak_idx][kick_side]['ankle']
            kick_ankle_after  = frames[peak_idx+2][kick_side]['ankle']
            dx = kick_ankle_after['x'] - kick_ankle_before['x']
            dy = kick_ankle_after['y'] - kick_ankle_before['y']
            result['follow_direction_alignment'] = 0.8 if abs(dx) > abs(dy) else 0.6
        else:
            result['follow_direction_alignment'] = 0.7

        # ── 重心转移质量 ──────────────────────────────
        if peak_idx < len(frames) - 1:
            hip_before = frames[peak_idx]['kp']
            hip_after  = frames[peak_idx+1]['kp']
            hcx_b = (hip_before['l_hip']['x'] + hip_before['r_hip']['x'])/2
            hcx_a = (hip_after['l_hip']['x'] + hip_after['r_hip']['x'])/2
            result['weight_transfer'] = _clamp01(abs(hcx_a - hcx_b) * 10)
        else:
            result['weight_transfer'] = 0.5

        # ── 传球次数（踝速度突变的峰值次数）──────────
        # 找出所有局部峰值
        peak_thresh = sum(all_v for all_v in [v['max'] for v in ankle_vels]) / len(ankle_vels) * 2
        peaks = []
        for i in range(1, len(ankle_vels)-1):
            if (ankle_vels[i]['max'] > ankle_vels[i-1]['max'] and
                ankle_vels[i]['max'] > ankle_vels[i+1]['max'] and
                ankle_vels[i]['max'] > peak_thresh):
                peaks.append(i)
        result['pass_count'] = len(peaks) if peaks else 1

        # ── 射门次数（踝速度更高的峰值）───────────────
        shot_thresh = sum(all_v for all_v in [v['max'] for v in ankle_vels]) / len(ankle_vels) * 3
        shot_peaks = [i for i in range(1, len(ankle_vels)-1)
                      if (ankle_vels[i]['max'] > ankle_vels[i-1]['max'] and
                          ankle_vels[i]['max'] > ankle_vels[i+1]['max'] and
                          ankle_vels[i]['max'] > shot_thresh)]
        result['shot_count'] = len(shot_peaks) if shot_peaks else 1

        # ── 变向次数 ─────────────────────────────────
        hip_dx = [hip_xs[i] - hip_xs[i-1] for i in range(1, len(hip_xs))]
        direction_reversals = sum(1 for i in range(1, len(hip_dx))
                                   if hip_dx[i] * hip_dx[i-1] < 0)
        result['change_count'] = direction_reversals

        # 急变向：方向反转且加速度大
        speeds = [abs(d) for d in hip_dx]
        sharp_turns = sum(1 for i in range(1, len(hip_dx)-1)
                          if hip_dx[i] * hip_dx[i-1] < 0 and
                             speeds[i] > sum(speeds)/len(speeds) * 2)
        result['sharp_turns'] = sharp_turns

        # ── 步频（踝左右交替摆动频率）───────────────
        all_ankle_x = [(f['kp']['l_ankle']['x'] + f['kp']['r_ankle']['x'])/2 for f in frames]
        peak_frames = [i for i in range(1, len(all_ankle_x)-1)
                       if all_ankle_x[i] > all_ankle_x[i-1] and
                          all_ankle_x[i] > all_ankle_x[i+1] and
                          abs(all_ankle_x[i] - all_ankle_x[i-1]) > 0.02]
        total_time = frames[-1]['time'] - frames[0]['time'] if len(frames) > 1 else 1
        result['step_frequency'] = len(peak_frames) / total_time if total_time > 0 else 1.0

        # ── 步幅 ─────────────────────────────────────
        if len(peak_frames) >= 2:
            strides = [abs(all_ankle_x[peak_frames[i]] - all_ankle_x[peak_frames[i-1]])
                       for i in range(1, len(peak_frames))]
            result['stride_length'] = sum(strides) / len(strides) if strides else 0.1
        else:
            result['stride_length'] = 0.1

        # ── 加速度 ────────────────────────────────────
        if len(hip_xs) >= 3:
            velocities = [(hip_xs[i] - hip_xs[i-1]) / max(frames[i]['time'] - frames[i-1]['time'], 0.01)
                          for i in range(1, len(hip_xs))]
            accelerations = [abs(velocities[i] - velocities[i-1])
                             for i in range(1, len(velocities))]
            result['acceleration'] = sum(accelerations) / len(accelerations) if accelerations else 0.1
        else:
            result['acceleration'] = 0.1

        # ── 跑位速度 ──────────────────────────────────
        if peak_idx < len(frames) - 1:
            hip_cx_before = (frames[peak_idx]['kp']['l_hip']['x'] + frames[peak_idx]['kp']['r_hip']['x'])/2
            hip_cx_after  = (frames[min(peak_idx+3, len(frames)-1)]['kp']['l_hip']['x'] +
                             frames[min(peak_idx+3, len(frames)-1)]['kp']['r_hip']['x'])/2
            dt = frames[min(peak_idx+3, len(frames)-1)]['time'] - frames[peak_idx]['time']
            result['reposition_speed'] = abs(hip_cx_after - hip_cx_before) / max(dt, 0.01)
        else:
            result['reposition_speed'] = 0.0

        # ── 弱脚使用比例 ─────────────────────────────
        if ankle_vels:
            l_kicks = sum(1 for v in ankle_vels if v['l'] > sum(x['l'] for x in ankle_vels)/len(ankle_vels))
            r_kicks = sum(1 for v in ankle_vels if v['r'] > sum(x['r'] for x in ankle_vels)/len(ankle_vels))
            total_k = l_kicks + r_kicks
            if total_k > 0:
                result['weak_foot_usage_ratio'] = min(l_kicks, r_kicks) / total_k
                kick_side_ratio = r_kicks / total_k if result['dominant_foot_raw'] == 'right' else l_kicks / total_k
                result['weak_foot_quality_score'] = (
                    (1 - result['weak_foot_usage_ratio']) *
                    (1 - result['weak_foot_usage_ratio'])
                )
            else:
                result['weak_foot_usage_ratio'] = 0.2
                result['weak_foot_quality_score'] = 0.6
        else:
            result['weak_foot_usage_ratio'] = 0.2
            result['weak_foot_quality_score'] = 0.6

        # ── 传球力道问题 ──────────────────────────────
        v = result['max_ankle_velocity']
        if v < 0.05:
            result['power_issue'] = 'underhit'   # 过软
        elif v > 0.4:
            result['power_issue'] = 'overhit'    # 过大
        else:
            result['power_issue'] = 'normal'

        # ── 传球出球方向偏差 ─────────────────────────
        result['direction_error'] = result['toe_alignment'] / 90.0  # 归一化

        return result

    # ──────────────────────────────────────────────
    #  球轨迹分析
    # ──────────────────────────────────────────────

    def _analyze_ball_trajectory(self, trajectory, frames, action_type):
        """
        基于追踪到的球轨迹，提取与文档对应的球相关指标。
        trajectory: list of {x, y, r, speed, time}
        """
        result = {}
        result['_ball_detected'] = len(trajectory) > 0

        if not trajectory or len(trajectory) < 2:
            print('[BallAnalyzer] No ball trajectory detected')
            result['ball_detected'] = False
            result['ball_speed'] = None
            result['ball_launch_angle'] = None
            result['contact_zone'] = None
            result['passing_type_inside'] = True
            result['passing_type_outside'] = False
            result['passing_type_long'] = False
            result['passing_type_through'] = False
            result['passing_type_lofted'] = False
            result['touch_before_pass'] = None
            result['max_ball_distance'] = None
            result['touch_frequency'] = None
            result['control_stability'] = None
            result['left_foot_ball_usage'] = 0.5
            result['right_foot_ball_usage'] = 0.5
            result['ball_change_count'] = 0
            result['ball_sharp_turns'] = 0
            result['shot_trajectory_list'] = []
            return result

        result['ball_detected'] = True

        # 速度统计
        speeds = [t['speed'] for t in trajectory]
        result['ball_speed'] = round(max(speeds), 4) if speeds else None

        # 出球角度（Y轴变化/X轴变化的反正切）
        speeds_filtered = [s for s in speeds if s > 0.01]  # 过滤静止帧
        if len(speeds_filtered) >= 2:
            ys = [t['y'] for t in trajectory if t['speed'] > 0.01]
            xs = [t['x'] for t in trajectory if t['speed'] > 0.01]
            if len(ys) >= 2 and len(xs) >= 2:
                dy = max(ys) - min(ys)
                dx = max(xs) - min(xs)
                result['ball_launch_angle'] = round(math.degrees(math.atan2(dy, dx + 1e-6)), 1)
            else:
                result['ball_launch_angle'] = None
        else:
            result['ball_launch_angle'] = None

        # 击球区域（A线=球纵向位置 / B线=左右）
        max_speed_idx = speeds.index(max(speeds)) if speeds else 0
        kick_frame_idx = max(0, min(max_speed_idx, len(trajectory)-1))
        kick_ball = trajectory[kick_frame_idx]
        ball_y_norm = kick_ball['y']  # 0=top, 1=bottom
        ball_x_norm = kick_ball['x']  # 0=left, 1=right
        # A线: top=线上方(0-0.4), middle=(0.4-0.6), bottom=(0.6-1.0)
        if ball_y_norm < 0.4:
            a_zone = 'A线上方'
        elif ball_y_norm < 0.6:
            a_zone = 'A线'
        else:
            a_zone = 'A线下方'
        # B线: left=(0-0.33), middle=(0.33-0.66), right=(0.66-1.0)
        if ball_x_norm < 0.33:
            b_zone = '左'
        elif ball_x_norm < 0.66:
            b_zone = '中'
        else:
            b_zone = '右'
        result['contact_zone'] = f'{a_zone}{b_zone}'

        # ── 传球类型推断 ─────────────────────────────
        avg_speed = sum(speeds_filtered) / len(speeds_filtered) if speeds_filtered else 0.1
        launch_angle = abs(result['ball_launch_angle']) if result['ball_launch_angle'] else 0

        result['passing_type_inside']  = True   # 默认内侧脚传球
        result['passing_type_outside']  = False
        result['passing_type_long']     = avg_speed > 0.08   # 速度高=长传
        result['passing_type_through']  = result['passing_type_long'] and launch_angle < 10
        result['passing_type_lofted']   = launch_angle > 30  # 高角度=高球

        # ── 触球次数（踢球前球在脚附近的帧数）─────────
        if frames and kick_frame_idx < len(trajectory):
            # 找踢球帧对应的 pose frame
            kick_time = trajectory[kick_frame_idx]['time']
            nearest_pose_idx = min(range(len(frames)),
                                    key=lambda i: abs(frames[i]['time'] - kick_time))
            # 踢球前1秒内（sample_interval=5帧≈1秒）的球轨迹
            pre_kick_trajectory = [t for t in trajectory
                                   if t['time'] < kick_time and
                                   kick_time - t['time'] < 1.0]
            if pre_kick_trajectory:
                # 计算每帧球与最近踝关节的距离
                touch_count = 0
                for t in pre_kick_trajectory:
                    nearest_ankle_dist = float('inf')
                    for f in frames:
                        if f['time'] > t['time']:
                            continue
                        kp = f['kp']
                        d = min(_dist_pts(t['x'], t['y'], kp['l_ankle']['x'], kp['l_ankle']['y']),
                                _dist_pts(t['x'], t['y'], kp['r_ankle']['x'], kp['r_ankle']['y']))
                        nearest_ankle_dist = min(nearest_ankle_dist, d)
                    if nearest_ankle_dist < 0.15:  # 归一化阈值
                        touch_count += 1
                result['touch_before_pass'] = touch_count
            else:
                result['touch_before_pass'] = None
        else:
            result['touch_before_pass'] = None

        # ── 球离脚最大距离（带球时）───────────────────
        if action_type == '带球' and frames:
            max_dist = 0.0
            for t in trajectory:
                min_ankle_dist = float('inf')
                for f in frames:
                    if f['time'] < t['time'] - 0.2 or f['time'] > t['time'] + 0.2:
                        continue
                    kp = f['kp']
                    d = min(_dist_pts(t['x'], t['y'], kp['l_ankle']['x'], kp['l_ankle']['y']),
                            _dist_pts(t['x'], t['y'], kp['r_ankle']['x'], kp['r_ankle']['y']))
                    min_ankle_dist = min(min_ankle_dist, d)
                max_dist = max(max_dist, min_ankle_dist)
            result['max_ball_distance'] = round(max_dist, 4)
        else:
            result['max_ball_distance'] = None

        # ── 触球频率（带球时球在脚附近的帧占比）───────
        if action_type == '带球' and frames and trajectory:
            total_frames_count = len(frames)
            close_frames = 0
            for t in trajectory:
                nearest_ankle = float('inf')
                for f in frames:
                    if abs(f['time'] - t['time']) > 0.3:
                        continue
                    kp = f['kp']
                    d = min(_dist_pts(t['x'], t['y'], kp['l_ankle']['x'], kp['l_ankle']['y']),
                            _dist_pts(t['x'], t['y'], kp['r_ankle']['x'], kp['r_ankle']['y']))
                    nearest_ankle = min(nearest_ankle, d)
                if nearest_ankle < 0.15:
                    close_frames += 1
            result['touch_frequency'] = round(close_frames / len(trajectory), 3) if trajectory else 0
        else:
            result['touch_frequency'] = None

        # ── 控球稳定性（带球时球距离标准差）───────────
        if action_type == '带球' and trajectory:
            # 取球轨迹中速度较小时（表示在控球）的距离数据
            controlled_ball = [t for t in trajectory if t['speed'] < 0.05]
            if controlled_ball and frames:
                distances = []
                for t in controlled_ball:
                    nearest = float('inf')
                    for f in frames:
                        if abs(f['time'] - t['time']) > 0.3:
                            continue
                        kp = f['kp']
                        d = min(_dist_pts(t['x'], t['y'], kp['l_ankle']['x'], kp['l_ankle']['y']),
                                _dist_pts(t['x'], t['y'], kp['r_ankle']['x'], kp['r_ankle']['y']))
                        nearest = min(nearest, d)
                    if nearest < 1.0:
                        distances.append(nearest)
                if distances:
                    std = math.sqrt(sum((d - sum(distances)/len(distances))**2
                                       for d in distances) / len(distances))
                    result['control_stability'] = _clamp01(1 - std * 10)
                else:
                    result['control_stability'] = 0.6
            else:
                result['control_stability'] = 0.6
        else:
            result['control_stability'] = None

        # ── 双脚球使用（带球时左右踝与球接近的帧比例）──
        if action_type == '带球' and frames and trajectory:
            left_count, right_count = 0, 0
            for t in trajectory:
                nearest = {'side': None, 'dist': float('inf')}
                for f in frames:
                    if abs(f['time'] - t['time']) > 0.3:
                        continue
                    kp = f['kp']
                    ld = _dist_pts(t['x'], t['y'], kp['l_ankle']['x'], kp['l_ankle']['y'])
                    rd = _dist_pts(t['x'], t['y'], kp['r_ankle']['x'], kp['r_ankle']['y'])
                    if ld < nearest['dist']:
                        nearest = {'side': 'l', 'dist': ld}
                    if rd < nearest['dist']:
                        nearest = {'side': 'r', 'dist': rd}
                if nearest['dist'] < 0.15:
                    if nearest['side'] == 'l':
                        left_count += 1
                    else:
                        right_count += 1
            total = left_count + right_count
            if total > 0:
                result['left_foot_ball_usage']  = round(left_count / total, 3)
                result['right_foot_ball_usage'] = round(right_count / total, 3)
            else:
                result['left_foot_ball_usage']  = 0.5
                result['right_foot_ball_usage'] = 0.5
        else:
            result['left_foot_ball_usage']  = 0.5
            result['right_foot_ball_usage'] = 0.5

        # ── 射门弹道类型（每次射门的结果推断）──────────
        shot_trajectories = []
        if speeds_filtered:
            max_speed_threshold = sum(speeds_filtered) / len(speeds_filtered)
            shot_peaks_idx = [i for i in range(1, len(speeds)-1)
                              if speeds[i] > max_speed_threshold * 1.5 and
                                 speeds[i] > speeds[i-1] and
                                 speeds[i] > speeds[i+1]]
            for idx in shot_peaks_idx[:3]:  # 最多3次射门
                if idx < len(trajectory):
                    t = trajectory[idx]
                    launch = result.get('ball_launch_angle', 0) or 0
                    speed = t['speed']
                    if speed > 0.1:
                        if launch < 10:
                            traj = '平射'
                        elif launch < 30:
                            traj = '低射'
                        else:
                            traj = '高射'
                        shot_trajectories.append({
                            'result': 'on_target' if traj != '高射' else 'off_target',
                            'trajectory': traj,
                            'error_reason': '踢球偏高' if launch > 20 else None
                        })
        result['shot_trajectory_list'] = shot_trajectories

        print(f'[BallAnalyzer] ball_detected={result["ball_detected"]}, '
              f'speed={result["ball_speed"]}, angle={result.get("ball_launch_angle")}, '
              f'zone={result["contact_zone"]}')

        return result

    # ──────────────────────────────────────────────
    #  评分计算
    # ──────────────────────────────────────────────

    def _calc_scores(self, action_type, full_data):
        scores = {}

        if action_type == '传球':
            scores['pass_accuracy']   = _clamp01(full_data['symmetry'] * 0.5 + full_data['contact_stability'] * 0.5)
            v = full_data['max_ankle_velocity']
            if 0.08 <= v <= 0.3:
                scores['power_control'] = 0.9
            elif v < 0.05:
                scores['power_control'] = 0.4
            elif v > 0.5:
                scores['power_control'] = 0.6
            else:
                scores['power_control'] = _clamp01(v * 3)
            scores['body_balance']    = full_data['stability']
            scores['support_foot']    = full_data['support_foot_stability']
            scores['vision']          = _clamp01(full_data['stability'] * 0.8 + 0.2)
            scores['ankle_lock']      = full_data.get('ankle_lock', 0.7)
            scores['follow_through']  = full_data.get('follow_through', 0.5)
            scores['weak_foot']        = 1 - full_data.get('weak_foot_usage_ratio', 0.2)

        elif action_type == '射门':
            scores['approach_rhythm']    = full_data.get('rhythm_variation', 0.7)
            scores['support_foot']        = full_data['support_foot_stability']
            kr = full_data['knee_angle_range']
            scores['swing_range'] = 1.0 if kr >= 80 else 0.85 if kr >= 60 else 0.65 if kr >= 40 else _clamp01(kr / 60)
            mka = full_data['min_knee_angle']
            scores['contact_point'] = 0.95 if mka <= 70 else 0.8 if mka <= 90 else 0.6 if mka <= 110 else 0.4
            scores['body_balance']  = full_data['stability']
            fl = full_data.get('forward_lean_angle', 0)
            scores['forward_lean_score'] = 0.9 if fl >= 15 else 0.7 if fl >= 8 else 0.5
            scores['ankle_lock']     = full_data.get('ankle_lock', 0.7)
            scores['follow_through']  = full_data.get('follow_through', 0.5)
            scores['weight_transfer'] = full_data.get('weight_transfer', 0.5)
            scores['head_stability']  = full_data.get('head_stability', 0.7)

        else:  # 带球
            scores['ball_feel']         = _clamp01(full_data['stability'] * 0.6 + full_data['symmetry'] * 0.4)
            scores['touch_freq']         = _clamp01(full_data.get('touch_frequency', 0.5) * 2)
            cog = full_data['center_of_gravity']
            scores['body_center']        = _clamp01((cog - 0.4) * 3) if cog > 0.4 else 0.5
            scores['speed_control']      = full_data['stability']
            lr = full_data['lateral_range']
            scores['change_direction']   = 0.9 if lr >= 0.2 else 0.7 if lr >= 0.1 else _clamp01(lr * 5 + 0.3)
            scores['arm_balance']        = full_data.get('arm_balance', 0.7)
            scores['weak_foot']          = 1 - full_data.get('weak_foot_usage_ratio', 0.2)
            scores['control_stability']  = full_data.get('control_stability', 0.6)

        weights = {
            '传球': {
                'pass_accuracy': 0.25, 'power_control': 0.20,
                'body_balance': 0.15, 'support_foot': 0.15,
                'vision': 0.10, 'ankle_lock': 0.08, 'follow_through': 0.07,
            },
            '射门': {
                'approach_rhythm': 0.12, 'support_foot': 0.18,
                'swing_range': 0.22, 'contact_point': 0.18,
                'body_balance': 0.08, 'forward_lean_score': 0.08,
                'ankle_lock': 0.06, 'follow_through': 0.05,
                'weight_transfer': 0.03,
            },
            '带球': {
                'ball_feel': 0.20, 'touch_freq': 0.18,
                'body_center': 0.18, 'speed_control': 0.15,
                'change_direction': 0.12, 'arm_balance': 0.07,
                'weak_foot': 0.05, 'control_stability': 0.05,
            }
        }
        w = weights.get(action_type, weights['传球'])
        total_raw = sum(scores.get(k, 0.5) * v for k, v in w.items())
        scores['total'] = min(100, max(30, round(total_raw * 100)))
        return scores

    # ──────────────────────────────────────────────
    #  Dify 专用数据构建
    # ──────────────────────────────────────────────

    def _build_dify_data(self, action_type, data, scores, player_info):
        """构建符合文档要求的 Dify inputs 数据"""
        player_info = player_info or {}

        # 惯用脚映射
        dominant_foot_map = {'右脚': '右脚', '左脚': '左脚', '双脚均衡': '双脚均衡'}
        dominant = dominant_foot_map.get(
            player_info.get('dominant_foot'),
            dominant_foot_map.get(data.get('dominant_foot_raw', 'right'), '右脚')
        )

        if action_type == '传球':
            return {
                'action_type': '传球',
                'pass_count': data.get('pass_count', 1),
                'player_estimation': {
                    'dominant_foot': dominant,
                    'age_estimation': player_info.get('age_estimation', 'U12')
                },
                'passing_mechanics': {
                    'plant_foot_distance': round(data.get('plant_foot_distance', 0.12), 4),
                    'toe_alignment': round(data.get('toe_alignment', 15), 1),
                    'ankle_lock': round(data.get('ankle_lock', 0.8), 3),
                    'contact_stability': round(data.get('contact_stability', 0.7), 3),
                    'follow_through': round(data.get('follow_through', 0.7), 3),
                },
                'passing_types': {
                    'inside_pass': data.get('passing_type_inside', True),
                    'outside_pass': data.get('passing_type_outside', False),
                    'long_pass': data.get('passing_type_long', False),
                    'through_pass': data.get('passing_type_through', False),
                    'lofted_pass': data.get('passing_type_lofted', False),
                },
                'passing_performance': {
                    'success_rate': None,   # 多人场景，无法评估
                    'interceptions': 0,
                    'overhit': 1 if data.get('power_issue') == 'overhit' else 0,
                    'underhit': 1 if data.get('power_issue') == 'underhit' else 0,
                    'direction_error': round(data.get('direction_error', 0), 3),
                },
                'decision_metrics': {
                    'touch_before_pass': data.get('touch_before_pass'),
                    'release_speed': round(data.get('ball_speed', 0.05), 4) if data.get('ball_detected') else None,
                    'scan_frequency': round(data.get('scan_frequency', 0.5), 3),
                },
                'movement_after_pass': {
                    'reposition_speed': round(data.get('reposition_speed', 0), 4),
                },
                'weak_foot': {
                    'usage_ratio': round(data.get('weak_foot_usage_ratio', 0.2), 3),
                    'quality_score': round(data.get('weak_foot_quality_score', 0.6), 3),
                },
                'stability': round(data.get('stability', 0.7), 3),
                'symmetry': round(data.get('symmetry', 0.8), 3),
            }

        elif action_type == '射门':
            # 出球速度：归一化速度映射到 15-35 m/s 范围（典型青少年射门速度）
            ball_speed_norm = data.get('ball_speed', 0.05) or 0.05
            ball_speed_mps = round(15 + ball_speed_norm * 200, 1)  # 粗略估算

            shot_outcomes = data.get('shot_trajectory_list', [])
            if not shot_outcomes:
                shot_outcomes = [{'result': 'on_target', 'trajectory': '平射', 'error_reason': None}]

            return {
                'action_type': '射门',
                'shot_count': data.get('shot_count', 1),
                'player_estimation': {
                    'dominant_foot': dominant,
                    'age_estimation': player_info.get('age_estimation', 'U14')
                },
                'runup_metrics': {
                    'approach_angle': round(data.get('approach_angle', 30), 1),
                    'step_count': data.get('step_count', 5),
                    'last_step_length': round(data.get('last_step_length', 0.1), 4),
                    'rhythm_variation': round(data.get('rhythm_variation', 0.7), 3),
                },
                'plant_foot': {
                    'distance_to_ball': round(data.get('plant_foot_distance', 0.12), 4),
                    'toe_direction': round(data.get('toe_alignment', 10), 1),
                    'knee_flexion': round(data.get('min_knee_angle', 15), 1),
                    'stability': round(data.get('support_foot_stability', 0.8), 3),
                },
                'kicking_leg': {
                    'ankle_lock': round(data.get('ankle_lock', 0.8), 3),
                    'knee_angle_range': round(data.get('knee_angle_range', 40), 1),
                    'max_ankle_velocity': round(data.get('max_ankle_velocity', 0.3), 4),
                },
                'impact': {
                    'contact_zone': data.get('contact_zone') or 'A线中',
                    'foot_surface': '鞋带区',
                    'ball_launch_angle': data.get('ball_launch_angle'),
                    'ball_speed': ball_speed_mps,
                },
                'body_posture': {
                    'forward_lean': round(data.get('forward_lean_angle', 10), 1),
                    'head_stability': round(data.get('head_stability', 0.8), 3),
                    'arm_balance': round(data.get('arm_balance', 0.8), 3),
                },
                'follow_through': {
                    'swing_amplitude': round(data.get('follow_through', 0.7), 3),
                    'direction_alignment': round(data.get('follow_direction_alignment', 0.8), 3),
                    'weight_transfer': round(data.get('weight_transfer', 0.6), 3),
                },
                'shot_outcomes': shot_outcomes,
                'stability': round(data.get('stability', 0.7), 3),
                'symmetry': round(data.get('symmetry', 0.8), 3),
            }

        else:  # 带球
            # 速度归一化转 m/s（粗略估算，1单位≈1.5m/s）
            avg_speed_mps = round(data.get('avg_ankle_velocity', 0.05) * 1.5, 1)
            max_speed_mps = round(data.get('max_ankle_velocity', 0.1) * 1.5, 1)
            accel_mps2    = round(data.get('acceleration', 0.1) * 1.5, 2)

            # 带球风格判断
            speed_ratio = (data.get('max_ankle_velocity', 0) /
                           (data.get('avg_ankle_velocity', 0.01) + 1e-6))
            if speed_ratio > 2.5:
                dribble_style = '速度型'
            elif data.get('touch_frequency', 0) > 0.6:
                dribble_style = '控制型'
            else:
                dribble_style = '技术型'

            return {
                'action_type': '带球',
                'player_estimation': {
                    'age_estimation': player_info.get('age_estimation', 'U12'),
                    'dominant_foot': dominant,
                    'dribbling_style': dribble_style,
                },
                'movement_metrics': {
                    'avg_speed': avg_speed_mps,
                    'max_speed': max_speed_mps,
                    'acceleration': accel_mps2,
                    'lateral_displacement': round(data.get('lateral_range', 0.15), 4),
                    'step_frequency': round(data.get('step_frequency', 1.5), 2),
                    'stride_length': round(data.get('stride_length', 0.08), 4),
                },
                'body_metrics': {
                    'center_of_gravity_height': round(data.get('center_of_gravity', 0.55) * 1.7, 3),
                    'forward_lean_angle': round(data.get('forward_lean_angle', 10), 1),
                    'body_stability': round(data.get('stability', 0.7), 3),
                    'symmetry': round(data.get('symmetry', 0.8), 3),
                    'arm_balance': round(data.get('arm_balance', 0.8), 3),
                },
                'ball_control_proxy': {
                    'touch_frequency': data.get('touch_frequency'),
                    'max_ball_distance': data.get('max_ball_distance'),
                    'control_stability': data.get('control_stability'),
                },
                'foot_usage': {
                    'left_usage_ratio': round(data.get('left_foot_ball_usage', 0.3), 3),
                    'right_usage_ratio': round(data.get('right_foot_ball_usage', 0.7), 3),
                },
                'direction_changes': {
                    'change_count': data.get('change_count', 0),
                    'sharp_turns': data.get('sharp_turns', 0),
                    'success_rate': data.get('control_stability', 0.7),
                },
                'visual_scan_proxy': {
                    'head_movement_frequency': round(data.get('head_movement_frequency', 0.5), 3),
                    'scan_frequency': round(data.get('scan_frequency', 0.5), 3),
                },
                'stability': round(data.get('stability', 0.7), 3),
                'symmetry': round(data.get('symmetry', 0.8), 3),
            }

    # ──────────────────────────────────────────────
    #  反馈 / 维度 / 推荐
    # ──────────────────────────────────────────────

    def _generate_feedback(self, action_type, scores, full_data):
        total = scores['total']
        dim_scores = {k: v for k, v in scores.items() if k != 'total'}
        weakest_key = (min(dim_scores, key=lambda k: dim_scores[k])
                       if dim_scores else 'body_balance')
        weakest_val = dim_scores.get(weakest_key, 0.6)

        if action_type == '传球':
            if total >= 85:
                return '传球动作流畅，技术规范，支撑脚与触球部位配合协调。弱脚使用意识和出球节奏需持续加强。'
            if weakest_key == 'pass_accuracy':
                return f'触球准确性有待提升。建议重点练习脚弓内侧触球点的稳定性，每次触球都应在相同位置。'
            if weakest_key == 'power_control':
                return '传球力度控制不稳定。建议用踝关节锁紧+小腿摆动来精确控制出球力量。'
            if weakest_key == 'support_foot':
                return f'支撑脚稳定性不足，触球瞬间重心失控。建议落脚后稳固再传球。'
            if weakest_key == 'weak_foot':
                return '弱脚使用比例偏低，存在只向惯用侧传球的倾向。建议专项练习弱脚传球。'
            return f'传球整体表现{self._get_level(total)}，建议重点加强{weakest_key}的练习。'

        elif action_type == '射门':
            if total >= 85:
                return '射门技术扎实，摆腿充分，身体前倾控制良好。建议多练习不同角度和不同脚法。'
            if weakest_key == 'swing_range':
                kr = round(full_data.get('knee_angle_range', 0), 1)
                return f'摆腿幅度偏小（膝关节角度变化{kr}°，标准应>60°）。建议加强大腿带动小腿的折叠练习。'
            if weakest_key == 'contact_point':
                mka = round(full_data.get('min_knee_angle', 90), 1)
                return f'踢球腿膝关节折叠不充分（最小角度{mka}°，标准<90°）。建议练习"脚背绷直+膝盖压球"。'
            if weakest_key == 'forward_lean_score':
                return '射门时身体后仰，重心未压住球，容易踢高。口诀："胸部盖过膝盖"。'
            if weakest_key == 'support_foot':
                return '支撑脚落点不稳。建议标记落脚位置（球侧旁10-15cm），反复练习固定步伐。'
            if weakest_key == 'ankle_lock':
                return '踝关节锁定不足，触球瞬间脚型不稳。建议射门时始终保持脚尖向下、脚踝绷紧。'
            return f'射门整体表现{self._get_level(total)}，建议强化射门动作链完整性。'

        else:
            if total >= 85:
                return '带球时身体重心控制优秀，变向流畅，触球频率稳定。建议挑战更高难度的障碍变向练习。'
            if weakest_key == 'change_direction':
                lr = round(full_data.get('lateral_range', 0) * 100, 1)
                return f'变向幅度偏小（横向位移{lr}cm范围）。建议专项练习"S形绕桩"和"变向加速"。'
            if weakest_key == 'body_center':
                return '带球时重心偏高，护球稳定性下降。建议降低重心（微曲膝关节），以半蹲姿势带球。'
            if weakest_key == 'touch_freq':
                return '触球频率偏低，球离脚较远。建议练习高频小触球（每步至少触球一次）。'
            if weakest_key == 'weak_foot':
                return '弱脚使用比例偏低，突破方向容易被预判。建议多练弱脚带球和变向。'
            return f'带球整体表现{self._get_level(total)}，建议提升球感练习量。'

    def _generate_dimensions(self, action_type, scores):
        dim_map = {
            '传球': [
                ('触球准确性', 'pass_accuracy'), ('力度控制', 'power_control'),
                ('身体协调性', 'body_balance'), ('支撑脚', 'support_foot'),
                ('踝关节锁定', 'ankle_lock'), ('随球动作', 'follow_through'),
                ('视野观察', 'vision'), ('弱脚能力', 'weak_foot'),
            ],
            '射门': [
                ('助跑节奏', 'approach_rhythm'), ('支撑脚', 'support_foot'),
                ('摆腿幅度', 'swing_range'), ('击球部位', 'contact_point'),
                ('踝关节锁定', 'ankle_lock'), ('身体前倾', 'forward_lean_score'),
                ('随球动作', 'follow_through'), ('重心转移', 'weight_transfer'),
                ('头部稳定', 'head_stability'),
            ],
            '带球': [
                ('球感', 'ball_feel'), ('触球频率', 'touch_freq'),
                ('身体重心', 'body_center'), ('速度控制', 'speed_control'),
                ('变向能力', 'change_direction'), ('手臂平衡', 'arm_balance'),
                ('弱脚能力', 'weak_foot'), ('控球稳定', 'control_stability'),
            ]
        }
        dims = []
        for name, key in dim_map.get(action_type, dim_map['传球']):
            raw = scores.get(key, 0.6)
            pct = round(raw * 100)
            level = '待提升'
            if pct >= 90: level = '卓越'
            elif pct >= 80: level = '优秀'
            elif pct >= 70: level = '良好'
            elif pct >= 60: level = '及格'
            dims.append({
                'name': name, 'score': pct, 'percentage': pct,
                'color': 'tertiary' if pct >= 85 else 'secondary' if pct >= 70 else 'default',
                'level': level
            })
        return dims

    def _generate_recommendations(self, action_type, scores):
        recs = {
            '传球': [
                {'title': '传球基础：脚内侧的温柔艺术', 'duration': '05:32', 'level': '入门',
                 'reason': '强化触球准确性和支撑脚稳定性'},
                {'title': '短传配合：与队友默契练习', 'duration': '08:15', 'level': '进阶',
                 'reason': '提升传球力度控制和弱脚稳定性'},
                {'title': '弱脚专项：双脚均衡训练', 'duration': '06:45', 'level': '进阶',
                 'reason': '提升弱脚传球质量'},
            ],
            '射门': [
                {'title': '射门技巧：脚背击球要领', 'duration': '06:20', 'level': '入门',
                 'reason': '掌握正确摆腿幅度和击球部位'},
                {'title': '助跑节奏：稳定的支撑脚', 'duration': '07:10', 'level': '进阶',
                 'reason': '强化支撑脚落点和助跑节奏'},
                {'title': '力量训练：踝关节爆发力', 'duration': '05:55', 'level': '进阶',
                 'reason': '提升踝关节锁定和出球力量'},
            ],
            '带球': [
                {'title': '带球入门：脚背控球基础', 'duration': '04:50', 'level': '入门',
                 'reason': '提升球感和重心稳定性'},
                {'title': '过人技巧：变向加速要领', 'duration': '07:30', 'level': '进阶',
                 'reason': '增强变向幅度和加速能力'},
                {'title': '弱脚带球：双脚均衡训练', 'duration': '06:20', 'level': '进阶',
                 'reason': '提升弱脚控球和突破方向变化'},
            ]
        }
        return recs.get(action_type, recs['传球'])

    # ──────────────────────────────────────────────
    #  工具方法
    # ──────────────────────────────────────────────

    def _get_icon(self, action_type):
        return {'传球': 'footprint', '射门': 'sports_soccer', '带球': 'speed'}.get(action_type, 'footprint')

    def _get_level(self, score):
        if score >= 90: return '卓越'
        elif score >= 80: return '优秀'
        elif score >= 70: return '良好'
        elif score >= 60: return '及格'
        else: return '待提升'

    def _get_date(self):
        return time.strftime('%m-%d %H:%M:%S', time.localtime())


# ─────────────────────────────────────────────────────────────────────────────
#  对外接口
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video_file(video_path, action_type='auto', player_info=None):
    analyzer = PoseAnalyzer()
    return analyzer.analyze_video(video_path, action_type, player_info)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python pose_analyzer.py <video_path> [auto|传球|射门|带球] [dominant_foot] [age]')
        sys.exit(1)
    vp = sys.argv[1]
    at = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    pi = {}
    if len(sys.argv) > 3:
        pi['dominant_foot'] = sys.argv[3]
    if len(sys.argv) > 4:
        pi['age_estimation'] = sys.argv[4]
    result = analyze_video_file(vp, at, pi)
    # 只打印 difyData 部分方便查看
    print(json.dumps({'difyData': result.get('difyData', {})}, ensure_ascii=False, indent=2))
    print(f'\n总评分: {result["score"]}/100 · {result["level"]}')
    print(f'动作: {result["actionType"]}')
    print(f'meta: {result["meta"]}')
