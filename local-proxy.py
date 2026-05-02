# -*- coding: utf-8 -*-
"""踢趣管理后台本地代理服务 v2.0（含姿态检测 + 大模型报告生成）"""
import sys, json, http.server, socketserver, subprocess, os, mimetypes, tempfile, base64, time
import urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')

PORT = 8888
APP_ID = 'wx3fd155759aa48454'
APP_SECRET = 'ccb1e245fc1d1b42a67726550c42f65c'
ENV_ID = 'cloudbase-3g7hxqd4d0db11a9'
FUNCTION_NAME = 'adminApi'
WEB_ADMIN_DIR = r'd:\.workbuddy\football-training\1.0.0-AI视频分析MVP\miniprogram\web-admin'

# ── Dify 大模型配置 ──────────────────────────────────────────
# 使用 Dify 工作流生成自然语言报告
# API Key 在 Dify 控制台 > 设置 > API Keys 中获取
# inputs 变量名需与 Dify 工作流中定义的变量名一致
DIFY_CONFIG = {
    'api_key': 'app-Tklo3mRIUBh2cAxb7DxTf04m',  # Dify App API Key（用户新提供）
    'base_url': 'https://api.dify.ai/v1',        # Dify 云服务地址（不改）
    'user': 'football_training_miniapp',          # 固定用户标识
}

# ── 尝试导入姿态分析器 ────────────────────────────────────
try:
    from pose_analyzer import PoseAnalyzer
    POSE_ANALYZER = PoseAnalyzer()
    print('[Proxy] PoseAnalyzer loaded successfully')
except ImportError as e:
    POSE_ANALYZER = None
    print(f'[Proxy] PoseAnalyzer not available: {e}')

mimetypes.add_type('text/html', '.html')
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/css', '.css')

def call_wx_api_once(action_data):
    """调用一次微信 API"""
    script = '''
import sys, json, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
APP_ID = 'wx3fd155759aa48454'
APP_SECRET = 'ccb1e245fc1d1b42a67726550c42f65c'
ENV_ID = 'cloudbase-3g7hxqd4d0db11a9'
FUNCTION_NAME = 'adminApi'

data = json.loads(sys.stdin.read())
body = json.dumps(data).encode()

# get token
url = f'https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={APP_ID}&secret={APP_SECRET}'
with urllib.request.urlopen(url, timeout=20) as resp:
    token_data = json.loads(resp.read().decode())
token = token_data['access_token']

# call function
invoke_url = f'https://api.weixin.qq.com/tcb/invokecloudfunction?access_token={token}&env={ENV_ID}&name={FUNCTION_NAME}'
req = urllib.request.Request(invoke_url, data=body, method='POST')
req.add_header('Content-Type', 'application/json')
with urllib.request.urlopen(req, timeout=20) as resp:
    result = resp.read().decode('utf-8')
sys.stdout.write(result)
'''
    try:
        proc = subprocess.run(
            [sys.executable, '-c', script],
            input=json.dumps(action_data).encode(),
            capture_output=True, timeout=40
        )
        if proc.returncode == 0:
            return proc.stdout.decode('utf-8')
        else:
            return json.dumps({'error': f'subprocess failed: {proc.stderr.decode("utf-8", errors="replace")}'})
    except subprocess.TimeoutExpired:
        return json.dumps({'error': '微信API请求超时'})

def call_wx_api(action_data):
    """调用微信 API，getCourses 自动分页"""
    action = action_data.get('action', '')
    
    if action == 'getCourses':
        all_courses = []
        seen_ids = set()
        all_ids = []  # 用于收集所有课程ID
        page = 1
        max_pages = 20
        
        while page <= max_pages:
            req = {'action': 'getCourses', 'data': {**action_data.get('data', {}), 'page': page, 'pageSize': 5}}
            result = call_wx_api_once(req)
            
            try:
                resp = json.loads(result)
            except:
                break
            
            inner = resp.get('resp_data', resp)
            if isinstance(inner, str):
                try: inner = json.loads(inner)
                except: break
            
            courses = inner.get('data', [])
            total = inner.get('total', 0)
            errcode = resp.get('errcode') or inner.get('errcode')
            
            # 先收集本页所有课程ID（无论成功还是报错）
            for c in courses:
                cid = c.get('_id') or c.get('id')
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_ids.append(cid)
            
            # 如果出错（尤其是超限），先收集本页ID，然后继续下页
            if errcode and errcode != 0:
                print(f'[getCourses] 第{page}页报错{errcode}，已收集本页ID，继续')
                page += 1
                continue
            
            # 正常处理（保留 content 字段，供编辑时使用）
            if not courses:
                break
            
            all_courses.extend(courses)
            
            if total > 0 and len(all_courses) >= total:
                break
            page += 1
        
        # 如果收集到的课程数少于 total，切换逐条拉取
        if total > 0 and len(all_courses) < total:
            print(f'[getCourses] 分页获取{len(all_courses)}/{total}，切换逐条拉取')
            remaining_ids = [i for i in all_ids if not any(
                (c.get('_id') or c.get('id')) == i for c in all_courses
            )]
            # 逐条拉取缺失的
            for cid in remaining_ids:
                r = call_wx_api_once({'action': 'getCourse', 'data': {'id': cid}})
                try:
                    cr = json.loads(r)
                    ic = cr.get('resp_data', cr)
                    if isinstance(ic, str): ic = json.loads(ic)
                    course = ic.get('data', {})
                    if isinstance(course, dict) and (course.get('_id') or course.get('id')):
                        all_courses.append(course)
                except:
                    pass
            print(f'[getCourses] 逐条拉取完成，共{len(all_courses)}门')
        
        return json.dumps({
            'errcode': 0, 'errmsg': 'ok',
            'resp_data': json.dumps({'success': True, 'data': all_courses, 'total': len(all_courses)}, ensure_ascii=False)
        })
    
    return call_wx_api_once(action_data)

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '3600')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            req_data = json.loads(body.decode('utf-8'))
            action = req_data.get('action', '')
            print(f'[POST] {action}')

            # 姿态检测分析端点
            if self.path.strip('/') == 'analyze-video' or action == 'analyzeVideo':
                self._handle_analyze_video(req_data)
                return

            # 大模型报告生成端点
            if self.path.strip('/') == 'generate-report' or action == 'generateReport':
                self._handle_generate_report(req_data)
                return

            result = call_wx_api(req_data)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(result.encode('utf-8'))
        except Exception as e:
            print(f'[错误] {e}')
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def _handle_analyze_video(self, req_data):
        """
        处理视频姿态分析请求 v2.0
        支持：
          1. videoUrl  - 腾讯云COS临时下载链接或普通HTTP链接
          2. videoData - base64编码的视频字节
        actionType='auto' 时由 PoseAnalyzer 自动识别动作类型
        """
        import urllib.request, ssl

        action_type = req_data.get('actionType', req_data.get('type', 'auto'))
        video_data = req_data.get('videoData', '')
        video_url  = req_data.get('videoUrl', '')
        video_id   = req_data.get('videoId', '')

        print(f'[Analyze] actionType={action_type}, videoUrl={video_url[:60] if video_url else "none"}')

        if not video_data and not video_url:
            self._send_json({'success': False, 'error': '请提供 videoUrl 或 videoData'})
            return

        video_path = None
        try:
            # ── 获取视频字节 ──────────────────────────────
            if video_url:
                print(f'[Analyze] Downloading video...')
                # 微信云存储临时下载链接需要跳过SSL验证
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
                headers = {'User-Agent': 'Mozilla/5.0'}
                req_obj = urllib.request.Request(video_url, headers=headers)
                try:
                    with urllib.request.urlopen(req_obj, timeout=60, context=ssl_ctx) as resp_obj:
                        video_bytes = resp_obj.read()
                    print(f'[Analyze] Downloaded {len(video_bytes)/1024:.0f} KB')
                except Exception as e:
                    print(f'[Analyze] Download failed: {e}')
                    self._send_json({'success': False, 'error': f'视频下载失败: {str(e)}'})
                    return
            else:
                video_bytes = base64.b64decode(video_data)
                print(f'[Analyze] Decoded base64: {len(video_bytes)/1024:.0f} KB')

            # ── 写临时文件 ────────────────────────────────
            suffix = '.mp4'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(video_bytes)
                video_path = f.name

            # ── 文件有效性校验 ────────────────────────────
            if len(video_bytes) < 5000:
                self._send_json({'success': False, 'error': '视频文件过小（<5KB），可能不是有效视频'})
                return

            # ── 调用真实姿态分析 ──────────────────────────
            if not POSE_ANALYZER:
                self._send_json({
                    'success': False,
                    'error': '姿态分析服务未初始化：本地代理缺少 PoseAnalyzer 模块，请确保 mediapipe 和 opencv-python 已安装',
                    'code': 'ANALYZER_NOT_INITIALIZED'
                })
                return

            # ── 获取 player_info（前端传来）────────────────────
            player_info_raw = req_data.get('playerInfo', {})
            player_info = {
                'dominant_foot': player_info_raw.get('dominantFoot', '右脚'),
                'age_estimation': player_info_raw.get('ageEstimation', 'U12'),
            }

            try:
                print(f'[Analyze] Running PoseAnalyzer (action_type={action_type}, player_info={player_info})...')
                t0 = time.time()
                result = POSE_ANALYZER.analyze_video(video_path, action_type, player_info)
                print(f'[Analyze] PoseAnalyzer done in {time.time()-t0:.1f}s, '
                      f'score={result.get("score")}, type={result.get("actionType")}')
            except RuntimeError as e:
                print(f'[Analyze] PoseAnalyzer error: {e}')
                self._send_json({'success': False, 'error': str(e), 'code': 'ANALYSIS_FAILED'})
                return
            except Exception as e:
                print(f'[Analyze] Unexpected error: {e}')
                self._send_json({'success': False, 'error': f'分析过程出现异常：{str(e)}', 'code': 'UNKNOWN_ERROR'})
                return

            self._send_json({'success': True, 'result': result})

        except Exception as e:
            print(f'[Analyze] Error: {e}')
            import traceback; traceback.print_exc()
            self._send_json({'success': False, 'error': str(e)})
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except Exception:
                    pass

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _handle_generate_report(self, req_data):
        """
        大模型生成自然语言诊断报告（通过 Dify 工作流）

        接收：
          {
            actionType: '带球' | '射门' | '传球',
            poseData: { stability, symmetry, score, dimensions, ... }  // pose_analyzer 输出的结构化数据
          }

        Dify inputs 字段说明（需与 Dify 工作流变量名匹配）：
          - action_type   : 动作类型（传球/射门/带球）
          - overall_score: 综合评分（0-100）
          - level         : 评级（优秀/良好/一般/待提升）
          - pose_analysis : 结构化的姿态分析数据（JSON字符串）
        """
        import ssl

        action_type = req_data.get('actionType', '带球')
        pose_data   = req_data.get('poseData', {})

        # ── 从 poseData 中提取完整的 difyData 结构 ────────
        dify_data = pose_data.get('difyData', {})

        # ── 构建 Dify 4 个输入变量 ────────────────────────
        inputs = {
            'action_type':   dify_data.get('action_type', action_type),
            'overall_score': str(pose_data.get('score', 0)),
            'level':         pose_data.get('level', '待提升'),
            'pose_analysis': json.dumps(dify_data, ensure_ascii=False),
        }

        # ── 调用 Dify /chat-messages ───────────────────────
        payload = json.dumps({
            'query': f'请生成一份{action_type}技术分析报告',
            'inputs': inputs,
            'user': DIFY_CONFIG['user'],
            'response_mode': 'blocking',   # 处理完成后返回（推荐）
            'auto_generate_name': False,
        }, ensure_ascii=False).encode('utf-8')

        ssl_ctx = ssl.create_default_context()
        headers = {
            'Authorization': f"Bearer {DIFY_CONFIG['api_key']}",
            'Content-Type': 'application/json',
        }

        try:
            url = f"{DIFY_CONFIG['base_url'].rstrip('/')}/chat-messages"
            print(f'[Dify] Calling {url} ...')
            req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=120, context=ssl_ctx) as resp:
                result = json.loads(resp.read().decode('utf-8'))

            answer = result.get('answer', '').strip()
            if answer:
                print(f'[Dify] Success, answer len={len(answer)}')
                self._send_json({
                    'success': True,
                    'report': answer,
                    'actionType': action_type,
                    'conversation_id': result.get('conversation_id', ''),
                })
            else:
                print('[Dify] Empty answer, using fallback')
                self._send_json({
                    'success': True,
                    'report': self._format_fallback_report(action_type, pose_data),
                    'actionType': action_type,
                })

        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            print(f'[Dify] HTTP Error {e.code}: {body}')
            self._send_json({
                'success': False,
                'error': f'Dify 请求失败（{e.code}），请检查工作流配置',
                'detail': body[:500],
            })
        except Exception as ex:
            print(f'[Dify] Error: {ex}')
            self._send_json({
                'success': False,
                'error': str(ex),
                'report': self._format_fallback_report(action_type, pose_data),
                'actionType': action_type,
            })

    def _format_fallback_report(self, action_type, pose_data):
        """Dify 不可用时，返回结构化数据的可读文本"""
        score = pose_data.get('score', 0)
        level = pose_data.get('level', '待提升')
        dims  = pose_data.get('dimensions', [])
        fb    = pose_data.get('feedback', '')

        lines = [
            f'# {action_type}技术分析报告\n',
            f'**综合评分：{score}/100 · 评级：{level}**\n',
            f'\n## AI诊断反馈\n{fb}\n',
            '\n## 各维度评分\n'
        ]
        for d in dims:
            bar = '█' * (d['score'] // 10) + '░' * (10 - d['score'] // 10)
            lines.append(f'- **{d["name"]}** {bar} {d["score"]}分（{d["level"]}）')

        lines.append('\n\n> 提示：Dify 工作流暂时不可用，请检查 Dify 应用状态和网络连接。')
        return '\n'.join(lines)

    def do_GET(self):
        path = self.path.strip('/')
        if path.startswith('web-admin/'):
            path = path[len('web-admin/'):]
        if path == '' or path == 'index.html':
            path = 'admin-login.html'

        local_path = os.path.join(WEB_ADMIN_DIR, path)
        if os.path.isfile(local_path):
            ext = os.path.splitext(local_path)[1]
            mime = mimetypes.guess_type(local_path)[0] or 'application/octet-stream'
            with open(local_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'404: {path}'.encode())

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True

def main():
    print('=' * 50)
    print('踢趣管理后台代理服务 v2.1（含 Dify 报告生成）')
    print('=' * 50)
    print(f'启动: http://localhost:{PORT}')
    print(f'登录: http://localhost:{PORT}/web-admin/admin-login.html')
    print(f'Dify App: app-0hbyTAUCCoNoC8BGfTHeRXZy')
    print('按 Ctrl+C 停止\n')
    with ThreadedHTTPServer(('', PORT), ProxyHandler) as httpd:
        httpd.serve_forever()

if __name__ == '__main__':
    main()
