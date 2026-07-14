"""长上下文(默认 100K)下纯 decode 的 TPOT / ITL / 吞吐 / acceptance 计测脚本。

逐 token 时间戳版:记录每个流式 chunk 的到达时刻,只在"所有请求都进入 decode 之后"
的稳态窗口内统计 —— 严格排除 batch 内的 prefill ramp,得到 batch=B、KV 深度≈prompt_len
的纯 decode 指标。

窗口定义:
  W0 = max_r(各请求首 token 时刻)   # 最后一个请求也进 decode,此后引擎无 prefill
  W1 = min_r(各请求末 token 时刻)   # 最早结束的请求
  再按 --trim 各切掉两端一小段(去 ramp / drain),得 [W0', W1'] 稳态窗口。

指标(均只在窗口内算):
  * TPOT  = 窗口时长 / 窗口内产出 token 数(逐请求算再聚合;vLLM bench 同公式
            (latency-ttft)/(tokens-1),这里把区间换成纯 decode 窗口)。
  * ITL   = 窗口内相邻 chunk 到达间隔(与 vLLM bench 的 itl 同口径;spec decode 下
            一个 chunk 可能含多 token,故 ITL 为 per-chunk)。
  * 吞吐  = 窗口内总产出 token / 窗口时长。
  * acceptance length = /metrics 的 spec_decode_* 增量:1 + accepted/drafts。

用法(server 需先起好;100K 请确保 --max-model-len ≥ prompt_len+output_len):
  python3 profile_decode_100k.py --batch 8 --input-len-k 100 --output-len 512
  python3 profile_decode_100k.py --batch 8 --input-len-k 100 --output-len 512 --start-profile
"""

import asyncio
import time
import random
import argparse
import json
import statistics
import urllib.request

import aiohttp


def fetch_model_name(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
        return json.load(resp)["data"][0]["id"]


def fetch_metrics(base_url: str) -> dict[str, float]:
    """抓 /metrics,返回 spec-decode 累计计数器(所有 label 求和)。"""
    keys = (
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_drafts_total",
    )
    out = {k: 0.0 for k in keys}
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
            for line in resp.read().decode().splitlines():
                if line.startswith("#") or not line:
                    continue
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if name in out:
                    out[name] += float(line.rsplit(" ", 1)[1])
    except Exception as e:
        print(f"[warn] 抓 /metrics 失败(不影响 TPOT/吞吐): {e}", flush=True)
    return out


def create_prompts(input_len: int, num_req: int) -> list[list[int]]:
    # 每个请求独立随机 prompt,保证各自真正 prefill(不命中任何 prefix cache)。
    return [[random.randint(42, 30000) for _ in range(input_len)] for _ in range(num_req)]


async def start_profile(base_url: str):
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base_url}/start_profile") as r:
            print(f"[profile] start_profile -> {r.status}: {(await r.text()).strip()}", flush=True)


async def single_task(prompt_ids, output_len, on_first_token):
    """流式发一个请求;返回 (chunk_ts, completion_tokens)。
    chunk_ts = 每个携带 token 的 chunk 的到达绝对时刻列表(逐 token 时间戳)。"""
    data = {
        "model": args.model,
        "stream": True,
        "max_tokens": output_len,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "prompt": prompt_ids,
        "ignore_eos": True,               # 保证吐满 output_len,batch 齐步走
        "stream_options": {"include_usage": True},
    }
    url = f"{args.base_url}/v1/completions"
    chunk_ts: list[float] = []
    completion_tokens = 0
    first = True
    timeout = aiohttp.ClientTimeout(total=100000)
    connector = aiohttp.TCPConnector(limit=0)
    def handle_line(s: str):
        nonlocal completion_tokens, first
        if not s.startswith("data:"):
            return
        payload = s[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return
        # 末尾 include_usage chunk:choices 空,带真实 completion_tokens
        usage = obj.get("usage")
        if usage and usage.get("completion_tokens"):
            completion_tokens = int(usage["completion_tokens"])
        choices = obj.get("choices") or []
        if choices and choices[0].get("text"):
            chunk_ts.append(time.time())     # 逐 token(chunk)时间戳
            if first:
                first = False
                on_first_token()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
            # 读原始字节 + 手动按换行切分 SSE(绕开 readline 的 512KB 单行上限)
            buf = b""
            async for chunk in response.content.iter_any():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    handle_line(line.decode(errors="ignore").strip())
            if buf.strip():
                handle_line(buf.decode(errors="ignore").strip())
    if completion_tokens == 0:
        completion_tokens = len(chunk_ts)            # 兜底
    return chunk_ts, completion_tokens


def _pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


async def run():
    input_len = args.input_len if args.input_len else int(args.input_len_k * 1024)
    output_len = args.output_len
    batch = args.batch
    print(f"[decode] batch={batch}, prompt_len={input_len} (~{input_len/1024:.1f}k), "
          f"output_len={output_len}, trim={args.trim}", flush=True)
    print(f"[decode] 请确认 server --max-model-len ≥ {input_len + output_len}", flush=True)

    prompts = create_prompts(input_len, batch)

    state = {"n_first": 0}
    all_in_decode = asyncio.Event()

    def on_first_token():
        state["n_first"] += 1
        print(f"[decode] 首 token {state['n_first']}/{batch}", flush=True)
        if state["n_first"] == batch:
            all_in_decode.set()

    tasks = [asyncio.create_task(single_task(p, output_len, on_first_token)) for p in prompts]
    print(f"[decode] 已发出 {batch} 个请求,等待 {batch}×{input_len} 的 prefill 完成"
          f"(可能几十秒,期间无 decode 输出属正常)...", flush=True)

    # 健壮等待:进入 decode 前的两种异常都快速失败,避免死等 / 测到无效窗口。
    while not all_in_decode.is_set():
        done, _ = await asyncio.wait(tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.done() and t.exception() is not None:
                raise t.exception()          # ① 请求出错(如 400 超长)直接抛
        # ② 已有请求 decode 结束,但仍有请求没进入 decode → batch 超过 server 并发能力,
        #    永远凑不出 batch 并发窗口,立刻报错(而不是等所有首 token 后才发现窗口为空)。
        if any(t.done() for t in tasks) and not all_in_decode.is_set():
            n_done = sum(t.done() for t in tasks)
            raise RuntimeError(
                f"batch={batch} 超过 server 并发能力:已有 {n_done} 个请求 decode 结束,"
                f"但只有 {state['n_first']}/{batch} 个进入过 decode —— 请求被严重错开,"
                f"无法构成 batch={batch} 的并发窗口。\n"
                f"        请降低 --batch,或提高 server --max-num-seqs 且确认 "
                f"batch×(prompt+output) ≤ GPU KV cache size。")
    if all_in_decode.is_set():
        print(f"[decode] {batch} 个请求全部进入 decode,开始纯 decode 计测窗口...", flush=True)
    else:
        print(f"[warn] 只有 {state['n_first']}/{batch} 个请求进入 decode 就结束了"
              f"(可能 KV 容量不足被抢占/空返回,batch×(prompt+output) 逼近容量?)。"
              f"继续用有效请求计测。", flush=True)
    m0 = fetch_metrics(args.base_url)
    if args.start_profile:
        await start_profile(args.base_url)

    results = await asyncio.gather(*tasks)
    m1 = fetch_metrics(args.base_url)

    # ---- 纯 decode 窗口 ----
    # 过滤掉空返回(chunk_ts 为空)的请求,避免越界并保证窗口有效。
    valid = [r for r in results if len(r[0]) >= 2]
    n_empty = len(results) - len(valid)
    if n_empty:
        print(f"[warn] {n_empty}/{len(results)} 个请求无有效输出(已剔除)", flush=True)
    if len(valid) < 1:
        print("[error] 没有有效请求可计测(batch 太大超 KV 容量?减小 --batch 或 --output-len)", flush=True)
        return
    eff_batch = len(valid)
    first_ts = [r[0][0] for r in valid]         # 各请求首 token 时刻
    last_ts  = [r[0][-1] for r in valid]        # 各请求末 token 时刻
    W0, W1 = max(first_ts), min(last_ts)
    if W1 <= W0:
        print("[error] 无重叠纯 decode 窗口(请增大 --output-len 或减小 batch)", flush=True)
        return
    span = W1 - W0
    W0t, W1t = W0 + args.trim * span, W1 - args.trim * span
    win = max(1e-6, W1t - W0t)

    # 逐请求:窗口内 TPOT + 收集窗口内 ITL
    tpots, itls, toks_in_win_total = [], [], 0.0
    for chunk_ts, ntok in valid:
        in_win = [t for t in chunk_ts if W0t <= t <= W1t]
        if len(in_win) < 2:
            continue
        # 该请求平均每 chunk token 数(spec decode 下一个 chunk 可能含多 token)
        tok_per_chunk = ntok / max(1, len(chunk_ts))
        toks_in_win_total += len(in_win) * tok_per_chunk
        # 逐请求 TPOT(窗口内)= 首末 in-win chunk 时间跨度 / 期间产出 token 数。
        # 期间 = len(in_win)-1 个 chunk 间隔,每 chunk 约 tok_per_chunk 个 token。
        span_r = in_win[-1] - in_win[0]
        toks_span_r = (len(in_win) - 1) * tok_per_chunk
        tpots.append(span_r / max(1e-9, toks_span_r))
        # 窗口内相邻 chunk 间隔 = ITL(per-chunk)
        itls += [b - a for a, b in zip(in_win[:-1], in_win[1:])]

    thpt = toks_in_win_total / win

    d_drafts   = m1["vllm:spec_decode_num_drafts_total"]          - m0["vllm:spec_decode_num_drafts_total"]
    d_accepted = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
    d_drafttok = m1["vllm:spec_decode_num_draft_tokens_total"]    - m0["vllm:spec_decode_num_draft_tokens_total"]
    accept_len   = (1 + d_accepted / d_drafts) if d_drafts > 0 else float("nan")
    per_tok_rate = (d_accepted / d_drafttok) if d_drafttok > 0 else float("nan")

    tpot_med = statistics.median(tpots) if tpots else float("nan")
    tpot_mean = statistics.mean(tpots) if tpots else float("nan")

    print("\n================ 纯 decode 结果(仅稳态窗口)================", flush=True)
    print(f"batch (有效/请求)       : {eff_batch}/{batch}", flush=True)
    print(f"prompt_len / output_len : {input_len} / {output_len}", flush=True)
    print(f"窗口时长 / trim          : {win:.2f}s / {args.trim}", flush=True)
    print(f"--- TPOT (每输出 token) ---", flush=True)
    print(f"TPOT  中位数            : {tpot_med*1000:.2f} ms/token", flush=True)
    print(f"TPOT  均值              : {tpot_mean*1000:.2f} ms/token", flush=True)
    print(f"--- ITL (per-chunk) ---", flush=True)
    print(f"ITL   中位/均值/p99     : {_pct(itls,0.5)*1000:.2f} / "
          f"{(statistics.mean(itls)*1000 if itls else float('nan')):.2f} / "
          f"{_pct(itls,0.99)*1000:.2f} ms", flush=True)
    print(f"--- 吞吐 (decode) ---", flush=True)
    print(f"窗口吞吐               : {thpt:.1f} tok/s", flush=True)
    print(f"B / TPOT(中位) 核对     : {eff_batch/tpot_med:.1f} tok/s", flush=True)
    print(f"--- spec decode (K={args.mtp}) ---", flush=True)
    print(f"acceptance length       : {accept_len:.3f}", flush=True)
    print(f"per-token 接受率        : {per_tok_rate:.3f}", flush=True)
    print(f"drafts/accepted/drafttok: {d_drafts:.0f} / {d_accepted:.0f} / {d_drafttok:.0f}", flush=True)
    print("==========================================================\n", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="长上下文纯 decode TPOT/ITL/吞吐计测(逐 token 时间戳)")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--model", type=str, default=None, help="不传则自动从 /v1/models 获取")
    p.add_argument("--batch", type=int, default=8, help="并发请求数 = decode batch")
    p.add_argument("--input-len-k", type=float, default=100, help="prompt 长度(k),100=100*1024")
    p.add_argument("--input-len", type=int, default=0, help="prompt 绝对 token 数,设了则覆盖 -k")
    p.add_argument("--output-len", type=int, default=512, help="每请求输出 token 数(decode 步数来源)")
    p.add_argument("--trim", type=float, default=0.1, help="窗口两端各切掉的比例(去 ramp/drain)")
    p.add_argument("--mtp", type=int, default=5, help="仅用于打印标注 K,不影响请求")
    p.add_argument("--start-profile", action="store_true", help="进入 decode 窗口后触发 server trace")
    args = p.parse_args()

    if args.model is None:
        try:
            args.model = fetch_model_name(args.base_url)
        except Exception as e:
            p.error(f"无法从 {args.base_url}/v1/models 获取 model(server 起了吗?): {e}")
    print(f"[decode] using model: {args.model}", flush=True)

    asyncio.run(run())

