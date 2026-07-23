"""图片颜色 K-means 聚类可视化 — Streamlit 部署版"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

AUDIO_PATH = Path(__file__).resolve().parent / "assets" / "Windy_Hill.mp3"

# ---------------------------------------------------------------------------
# 页面与主题
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="图片颜色聚类可视化",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7f8fa 0%, #eef1f5 100%);
        border-right: 1px solid #e2e6ec;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #1a2332;
    }

    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1a2332;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .hero-sub {
        color: #5a6577;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .result-card {
        background: #f4f6f9;
        border: 1px solid #e2e6ec;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        color: #2c3544;
        line-height: 1.65;
        font-size: 0.92rem;
    }

    .swatch-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 0.75rem 0 1.25rem;
    }

    .swatch {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    div[data-testid="stButton"] > button {
        background-color: #1f6b5c !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #17564a !important;
        border: none !important;
    }

    div[data-testid="stButton"] > button:disabled {
        background-color: #c5ccd6 !important;
        color: #f5f5f5 !important;
    }

    .stSelectbox label, .stSlider label, .stRadio label,
    .stTextInput label, .stFileUploader label {
        font-weight: 600 !important;
        color: #1a2332 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_ENDPOINTS = {
    "OpenAI 官方 (api.openai.com)": "https://api.openai.com/v1/chat/completions",
    "代理平台 (openai-proxy.org)": "https://api.openai-proxy.org/v1/chat/completions",
}


# ---------------------------------------------------------------------------
# 颜色空间与 K-means
# ---------------------------------------------------------------------------
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N, 3) uint/float 0-255 → LAB"""
    rgb = rgb.astype(np.float64) / 255.0
    mask = rgb > 0.04045
    rgb = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = rgb @ m.T
    xyz /= np.array([0.95047, 1.00000, 1.08883])

    mask = xyz > 0.008856
    f = np.where(mask, np.cbrt(xyz), (7.787 * xyz) + 16 / 116)

    L = (116 * f[:, 1]) - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.column_stack([L, a, b])


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """lab: (N, 3) → rgb uint8 (N, 3)"""
    lab = np.atleast_2d(lab).astype(np.float64)
    y = (lab[:, 0] + 16) / 116
    x = lab[:, 1] / 500 + y
    z = y - lab[:, 2] / 200
    f = np.column_stack([x, y, z])

    mask = f**3 > 0.008856
    xyz = np.where(mask, f**3, (f - 16 / 116) / 7.787)
    xyz *= np.array([0.95047, 1.00000, 1.08883])

    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb = xyz @ m_inv.T
    mask = rgb > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(np.clip(rgb, 0, None), 1 / 2.4) - 0.055, 12.92 * rgb)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def run_kmeans(
    pixels: np.ndarray,
    k: int,
    space: Literal["RGB", "LAB"],
    max_iter: int = 10,
    seed: int = 42,
) -> list[dict]:
    """对降采样像素做 K-means，返回按像素量降序的聚类结果。"""
    rng = np.random.default_rng(seed)
    data = rgb_to_lab(pixels) if space == "LAB" else pixels.astype(np.float64)

    # 随机初始化质心
    indices = rng.choice(len(data), size=k, replace=False)
    centroids = data[indices].copy()

    labels = np.zeros(len(data), dtype=np.int32)
    for _ in range(max_iter):
        # 分配
        dists = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        # 更新
        for i in range(k):
            members = data[labels == i]
            if len(members) > 0:
                centroids[i] = members.mean(axis=0)

    results = []
    for i in range(k):
        cnt = int((labels == i).sum())
        if cnt == 0:
            continue
        if space == "LAB":
            rgb = lab_to_rgb(centroids[i : i + 1])[0]
        else:
            rgb = np.clip(np.round(centroids[i]), 0, 255).astype(np.uint8)
        results.append(
            {
                "rgb": tuple(int(x) for x in rgb),
                "color": f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
                "cnt": cnt,
            }
        )

    results.sort(key=lambda x: x["cnt"], reverse=True)
    return results


def extract_pixels(image: Image.Image, max_side: int = 800, step: int = 50) -> tuple[Image.Image, np.ndarray]:
    """缩放预览图并降采样提取 RGB 像素。"""
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)
    flat = arr.reshape(-1, 4)[:, :3] if arr.shape[-1] == 4 else arr.reshape(-1, 3)
    # 与原版一致：按平坦索引步长采样
    sampled = flat[::step]
    return img, sampled


def build_chart(cluster_data: list[dict], chart_type: str) -> go.Figure:
    colors = [d["color"] for d in cluster_data]
    counts = [d["cnt"] for d in cluster_data]
    labels = [f"RGB{d['rgb']}" for d in cluster_data]

    if chart_type == "饼图":
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=counts,
                    marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                    hole=0.45,
                    textinfo="none",
                    hovertemplate="%{label}<br>像素量: %{value}<extra></extra>",
                )
            ]
        )
        fig.update_layout(showlegend=True)
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=counts,
                    marker=dict(color=colors, line=dict(width=0)),
                    hovertemplate="%{x}<br>像素量: %{y}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            xaxis=dict(title="", showticklabels=False, showgrid=False),
            yaxis=dict(title="像素数量", gridcolor="#eef1f5", zeroline=False),
        )

    fig.update_layout(
        title=dict(text="聚类颜色分布", x=0.5, xanchor="center", font=dict(size=16, color="#1a2332")),
        margin=dict(t=50, b=30, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
        font=dict(family="DM Sans, sans-serif", color="#5a6577"),
    )
    return fig


def call_llm(colors_str: str, api_key: str, endpoint: str) -> str:
    prompt = (
        f"我通过 K-means 算法对一张图片提取了多个颜色聚类中心，颜色值如下：{colors_str}。"
        "你作为一名色彩学专家，请判断这几个颜色搭配在一起是否和谐，并简要给出分析理由（150字以内）。\n"
        "注意：在回复中提到某个颜色时，请务必用直观的文字描述该颜色（例如：'c1的浅蓝色'或'c3的暗红色'），不要只写代号。"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "你是一个专业的视觉设计和色彩搭配专家。"},
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    if not response.ok:
        raise RuntimeError(f"API 返回 {response.status_code}: {response.text[:300]}")
    result = response.json()
    return result["choices"][0]["message"]["content"]


def resolve_api_key(ui_key: str) -> str | None:
    key = (ui_key or "").strip()
    if key:
        return key
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key and env_key != "sk-your_api_key_here":
        return env_key
    try:
        secret = st.secrets.get("OPENAI_API_KEY", "")
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 控制面板")
    st.caption("上传图片并调节参数，即可完成颜色聚类。")

    uploaded = st.file_uploader("1. 选择图片", type=["png", "jpg", "jpeg", "webp", "bmp"])

    k_value = st.slider("2. 聚类数量 K", min_value=2, max_value=10, value=5)

    color_space = st.selectbox("3. 颜色空间", options=["RGB", "LAB"])

    chart_type = st.selectbox("4. 可视化样式", options=["饼图", "柱状图"])

    st.divider()
    st.markdown("### API 设置")
    st.caption("密钥仅保存在当前会话，不会写入服务器磁盘。")

    api_provider = st.radio(
        "接口来源",
        options=list(API_ENDPOINTS.keys()),
        index=1,
    )
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="也可在 .env 或 Streamlit Secrets 中预先配置 OPENAI_API_KEY。",
    )

    st.divider()
    st.markdown("### 背景音乐")
    if AUDIO_PATH.exists():
        st.audio(str(AUDIO_PATH), format="audio/mp3", autoplay=True)
        st.caption("配上背景音乐，看图和调色都会更轻松一点。")
    else:
        st.warning("未找到音频文件：assets/Windy_Hill.mp3")

# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
st.header("图片颜色聚类可视化")
st.markdown("提取主色调 · Plotly 交互图表 · LLM 色彩和谐度评估")

if "ai_reply" not in st.session_state:
    st.session_state.ai_reply = ""
if "analysis_fingerprint" not in st.session_state:
    st.session_state.analysis_fingerprint = ""

if uploaded is None:
    st.info("请在左侧上传一张图片开始分析。")
else:
    image = Image.open(uploaded)
    preview, pixels = extract_pixels(image)
    cluster_data = run_kmeans(pixels, k_value, color_space)  # type: ignore[arg-type]

    fingerprint = f"{uploaded.name}|{k_value}|{color_space}|{len(pixels)}"
    if fingerprint != st.session_state.analysis_fingerprint:
        st.session_state.analysis_fingerprint = fingerprint
        st.session_state.ai_reply = ""

    st.session_state.cluster_data = cluster_data
    st.session_state.preview_image = preview

    col_img, col_chart = st.columns([1, 1.35], gap="large")

    with col_img:
        st.markdown("**原图预览**")
        st.image(preview, use_container_width=True)

        swatches = "".join(
            f'<div class="swatch" title="RGB{d["rgb"]}" style="background:{d["color"]}"></div>'
            for d in cluster_data
        )
        st.markdown(f'<div class="swatch-row">{swatches}</div>', unsafe_allow_html=True)
        st.caption("聚类主色色板（按像素量降序）")

    with col_chart:
        fig = build_chart(cluster_data, chart_type)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**颜色搭配分析**")
    analyze = st.button("调用大模型判断和谐度", type="primary", use_container_width=False)

    if analyze:
        api_key = resolve_api_key(api_key_input)
        if not api_key:
            st.warning("请先在侧边栏填写 API Key，或配置环境变量 / Streamlit Secrets。")
        else:
            colors_str = "; ".join(
                f"c{i + 1}: RGB({d['rgb'][0]}, {d['rgb'][1]}, {d['rgb'][2]})"
                for i, d in enumerate(cluster_data)
            )
            endpoint = API_ENDPOINTS[api_provider]
            with st.spinner("正在请求 AI 分析，请稍候…"):
                try:
                    reply = call_llm(colors_str, api_key, endpoint)
                    st.session_state.ai_reply = reply
                except Exception as exc:
                    st.session_state.ai_reply = ""
                    st.error(f"大模型接口调用失败：{exc}")

    if st.session_state.ai_reply:
        st.markdown(
            f'<div class="result-card"><strong>AI 评估结果</strong><br/>{st.session_state.ai_reply}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-card">等待 AI 分析结果…</div>',
            unsafe_allow_html=True,
        )
