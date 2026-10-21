"""回归测试：do_no_tool 对「文本形式工具调用 / 上游 DSML 标记泄漏」的处理。"""
import sys, re
sys.path.insert(0, '/Users/mac/Documents/Dev/galley/managed-ga/code')
import ga


class R:
    def __init__(self, c):
        self.content = c
        self.thinking = ''


def make(content):
    h = object.__new__(ga.GenericAgentHandler)
    h.working = {}
    h.current_turn = 5
    h.calls = []
    h.parent = None

    def _retry_or_exit(prompt):
        h.calls.append(('RETRY', prompt[:50]))
        yield '[STUB-RETRY]'

    def do_code_run(args, resp):
        h.calls.append(('CODE_RUN', args.get('type'), (args.get('script') or '')[:48]))
        yield '[STUB-CODE-OK]'

    h._retry_or_exit = _retry_or_exit
    h.do_code_run = do_code_run
    h._in_plan_mode = lambda: False
    return h


CASES = []

# 1. 真实 t96 内容（DeepSeek 泄漏 + 正文 code_run）
real = ("构建已通过。我发现两个关键缺口：卡片没有「断开连接」按钮，且 hub 启动的是 npm 上的旧版 canvas。\n"
        "<code_run script=\"true\">cd /Users/mac/Documents/Dev/infinite-canvas && sed -n '1,60p' canvas-agent/src/index.ts; "
        "echo \"=== deps\"</code_run>\n"
        "<parameter name=\"type\">bash</｜｜DSML｜｜ parameter>\n"
        "</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>\n")
CASES.append(('真实t96(语言+code_run+DSML)', real, 'CODE_RUN'))

# 2. 正常收尾（无任何标记）→ 应当结束本轮
CASES.append(('正常最终答复', '任务已完成，三个文件都改好了。', 'END'))

# 3. 正文只是提到 DSML 一词 → 不应触发重试
CASES.append(('正文提及DSML一词', 'DSML 是 DeepSeek 的特殊标记，Galley 需要容错。', 'END'))

# 4. 只有 DSML 垃圾、无可执行内容 → 重试
CASES.append(('仅DSML泄漏无正文工具', '好的，我继续。\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>', 'RETRY'))

# 5. 空壳 code_run（无代码体）+ DSML → 重试
CASES.append(('空壳code_run+DSML', '继续执行。\n<code_run script="true"></code_run>\n</｜｜DSML｜｜ calls>', 'RETRY'))

# 6. 原有行为：summary 半句 → 重试
CASES.append(('summary半句(原有)', '<summary>The image: orange% 0.16, dark% 99.79 — suspicious: dark 99.7</summary>', 'RETRY'))
CASES.append(('上游错误串(纯,33字)', '!!!Error: Upstream request failed', 'RETRY'))
CASES.append(('上游错误串(429额度)', '!!!Error: HTTP 429: {"code":"API_KEY_QUOTA_EXHAUSTED","message":"API key 额度已用完"}', 'RETRY'))
CASES.append(('上游错误串(401禁用)', '!!!Error: HTTP 401: {"code":"API_KEY_DISABLED","message":"API key is disabled"}', 'RETRY'))
CASES.append(('代码块里提到!!!Error', '关于日志，我确认：调用链里出现过 `!!!Error: Upstream request failed`，已修好并重跑了 3 次回归全部通过。', 'END'))

# 7. 原有行为：正常带代码块收尾的最终答复仍应正常结束
CASES.append(('带```代码块的讲解', '用法示例：\n```python\nprint(1)\nprint(2)\nprint(3)\nprint(4)\nprint(5)\nprint(6)\n```\n以上是全部内容，无需再执行。', 'END'))

ok = fail = 0
for name, content, want in CASES:
    h = make(content)
    g = h.do_no_tool({}, R(content))
    ret, ys = None, []
    while True:                      # generator 的 return 值要从 StopIteration 取
        try:
            ys.append(next(g))
        except StopIteration as e:
            ret = e.value
            break
    kinds = [c[0] for c in h.calls]
    if want == 'CODE_RUN':
        good = 'CODE_RUN' in kinds and 'RETRY' not in kinds
    elif want == 'RETRY':
        good = 'RETRY' in kinds
    else:
        good = not kinds and getattr(ret, 'next_prompt', 'x') is None
    print(('  PASS ' if good else '  FAIL ') + f'{name:28s} calls={h.calls}')
    ok += good
    fail += (not good)
print(f'\n结果: {ok} passed, {fail} failed')
sys.exit(1 if fail else 0)
