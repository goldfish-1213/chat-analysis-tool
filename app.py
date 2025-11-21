import streamlit as st
# 确保这些 import 都在最上面
import ijson
import pandas as pd  # 这里就是报错找不到的 pd，一定要有！
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from collections import Counter, defaultdict
import re
import numpy as np
from datetime import datetime
import os
import gc

# ==========================================
# 0. 基础配置
# ==========================================
st.set_page_config(page_title="ChatGPT 深度分析 31.0", layout="wide", page_icon="📊")

st.markdown("""
<style>
div[data-testid="stColorPicker"] {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 字体加载器
# ==========================================
@st.cache_resource
def get_custom_font_path():
    font_path = "msyh.ttc"
    if os.path.exists(font_path):
        return font_path
    else:
        return "arial.ttf" 

def get_custom_font_prop(size=14, weight='normal'):
    fp = get_custom_font_path()
    prop = fm.FontProperties(fname=fp)
    prop.set_size(size)
    prop.set_weight(weight)
    return prop

# ==========================================
# 2. 内置停用词表
# ==========================================
DEFAULT_STOPWORDS = {
    "---", "...", "###", "___", "***", "=>", "->", "<-", "<=", ">=", "**", "__", "~~", "==", "!=", "&&", "||", "```", "`", ">", "|", "[", "]", "{", "}", "(", ")",
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "吗", "吧", "啊", "可以", "这个", "那个", "其实", "因为", "所以", "如果", "但是", 
    "嗯", "哦", "什么", "怎么", "为", "之", "与", "及", "其", "以", "对", "而", "着", "下", "地", "子", "中", "她", "他", "们", "我们", "你们", "他们", "它们", 
    "哈哈", "哈哈哈", "嘿嘿", "呃", "嘛", "呀", "呢", "啦", "哇", "唉", "哎", "哼", "噢", "呗",
    "哪怕", "可能", "觉得", "应该", "这种", "那种", "比如", "或者", "以及", "然后", "虽然", "但是", "不过", "只是", "这样", "那样", "这里", "那里",
    "确实", "真的", "非常", "特别", "比较", "稍微", "只要", "只有", "除非", "几乎", "简直", "似乎", "好像", "大概", "也许",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", "、", "，", "。", "？", "！", "“", "”", "：", "；", "（", "）", "《", "》", "…", "—", "～",
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "when", "where", "how", "why", "which", "who", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "this", "that", "these", "those", "there", "here", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down", "out", "about", "into", "over", "after", "can", "could", "will", "would", "should", "may", "might", "must", "not", "no", "yes", "ok", "okay", "well", "just", "only", "very", "really", "too", "so", "much", "many", "some", "any", "all"
}

# ==========================================
# 3. 核心解析函数 (流式读取修复版)
# ==========================================
@st.cache_data
def parse_and_count_stream(file, stop_words):
    try:
        # 【新增】重要：将文件指针重置到开头，防止多次读取时出错
        file.seek(0) 
        
        # ijson 流式读取
        # 'item' 表示根列表下的每一项
        conversations = ijson.items(file, 'item')
        
        user_counter = Counter()
        ai_counter = Counter()
        timeline_counters = defaultdict(Counter)
        timeline_counts = defaultdict(int)

        u_count = 0
        u_total_len = 0
        a_count = 0
        a_total_len = 0

        for conversation in conversations:
            mapping = conversation.get('mapping', {})
            create_time = conversation.get('create_time')
            base_dt = datetime.fromtimestamp(create_time) if create_time else None
            
            for node_id, node_data in mapping.items():
                message = node_data.get('message')
                # 防御性检查
                if message is None: continue
                    
                if message and message.get('content') and message.get('author'):
                    role = message['author']['role']
                    content_parts = message['content'].get('parts', [])
                    text_content = "".join([part for part in content_parts if isinstance(part, str)])
                    
                    if text_content:
                        text_len = len(text_content)
                        msg_time = message.get('create_time')
                        dt = datetime.fromtimestamp(msg_time) if msg_time else base_dt
                        month_key = dt.strftime('%Y-%m') if dt else "Unknown"

                        if role == 'user':
                            u_count += 1
                            u_total_len += text_len
                            words = jieba.cut(text_content)
                            filtered = [w for w in words if len(w.strip()) > 1 and w.strip().lower() not in stop_words]
                            user_counter.update(filtered)
                            if month_key != "Unknown":
                                timeline_counters[month_key].update(filtered)
                                timeline_counts[month_key] += 1

                        elif role == 'assistant':
                            a_count += 1
                            a_total_len += text_len
                            words = jieba.cut(text_content)
                            filtered = [w for w in words if len(w.strip()) > 1 and w.strip().lower() not in stop_words]
                            ai_counter.update(filtered)
        
        # 计算平均值
        u_avg = int(u_total_len / u_count) if u_count > 0 else 0
        a_avg = int(a_total_len / a_count) if a_count > 0 else 0
        
        # 显式垃圾回收
        del conversations
        gc.collect()

        return {
            "u_counter": user_counter,
            "a_counter": ai_counter,
            "timeline_counters": timeline_counters,
            "timeline_counts": timeline_counts,
            "u_count": u_count, "u_avg": u_avg,
            "a_count": a_count, "a_avg": a_avg
        }
                            
    except Exception as e:
        st.error(f"解析出错: {e}")
        return None

# ==========================================
# 4. 颜色截断器
# ==========================================
def get_truncated_cmap(cmap_name, min_val=0.0, max_val=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap_name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n))
    )
    return new_cmap

# ==========================================
# 5. 界面侧边栏
# ==========================================
wordcloud_colormaps = {"Blues (蓝)": "Blues", "Oranges (橙)": "Oranges", "Reds (红)": "Reds", "Greens (绿)": "Greens", "Purples (紫)": "Purples", "viridis (极光)": "viridis", "magma (岩浆)": "magma", "cool (冷色)": "cool", "autumn (秋色)": "autumn"}
USER_ICON = "👾" 
AI_ICON = "🦾"

with st.sidebar:
    st.header("⚙️ 设置面板 v31.0")
    uploaded_file = st.file_uploader("1. 上传 conversations.json", type=['json'])
    
    st.markdown("---")
    st.header("🎨 视觉定制")
    
    st.write("🌈 **颜色浓度**")
    color_intensity = st.slider("去除浅色比例", 0.0, 0.6, 0.3)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"{USER_ICON} 你")
        user_wc_color = st.selectbox("你的色系", list(wordcloud_colormaps.keys()), index=0)
        
    with c2:
        st.subheader(f"{AI_ICON} AI")
        ai_wc_color = st.selectbox("AI 的色系", list(wordcloud_colormaps.keys()), index=1)

    st.markdown("---")
    max_words_limit = st.slider("📊 数据显示数量", 20, 500, 100)
    
    st.markdown("---")
    st.header("🛡️ 过滤设置")
    use_default = st.toggle("使用内置净化词表", value=True)
    custom_input = st.text_area("自定义屏蔽词", height=80)
    
    final_stopwords = set(DEFAULT_STOPWORDS) if use_default else set()
    if custom_input: final_stopwords.update([w.strip().lower() for w in re.split(r'[ ,，\n]+', custom_input) if w.strip()])

# ==========================================
# 6. 词云面板
# ==========================================
def show_wordcloud_panel(word_counts, cmap_name, title, icon, limit, min_val):
    if not word_counts: return
    
    base_cmap_name = wordcloud_colormaps[cmap_name]
    custom_cmap = get_truncated_cmap(base_cmap_name, min_val=min_val, max_val=1.0)
    fp = get_custom_font_path()

    try:
        wc = WordCloud(
            font_path=fp, 
            width=1000, height=1000,
            background_color='white', 
            colormap=custom_cmap, 
            max_words=limit, 
            contour_width=0
        ).generate_from_frequencies(word_counts)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e: st.error(f"生成失败: {e}")
    
    # 这里的 pd 必须在顶部 import pandas as pd
    with st.expander(f"📋 查看 {icon} {title} 高频词表", expanded=False):
        st.dataframe(pd.DataFrame(word_counts.most_common(limit), columns=['词语', '次数']), use_container_width=True, height=300)

# ==========================================
# 7. 柱状图面板
# ==========================================
def show_barchart_panel(word_counts, cmap_name, plain_text_title, limit):
    if not word_counts: return
    
    common_words = word_counts.most_common(limit)
    df = pd.DataFrame(common_words, columns=['Word', 'Count']).sort_values(by='Count', ascending=True)
    
    height_per_row = 0.45
    dynamic_height = max(6, len(df) * height_per_row)
    fig, ax = plt.subplots(figsize=(12, dynamic_height))
    
    font_normal = get_custom_font_prop(size=14)
    font_title = get_custom_font_prop(size=35, weight='bold')
    
    base_cmap_name = wordcloud_colormaps[cmap_name]
    cmap = get_truncated_cmap(base_cmap_name, 0.3, 0.9)
    
    for i, (word, count) in enumerate(zip(df['Word'], df['Count'])):
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap, extent=(0, count, i - 0.3, i + 0.3))
        ax.text(count + (df['Count'].max() * 0.01), i, str(count), va='center', fontsize=12)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Word'], fontproperties=font_normal)
    ax.set_title(f"{plain_text_title} Top {limit} 词频统计", pad=40, fontproperties=font_title)
    
    ax.set_ylim(-0.5, len(df) - 0.5) 
    ax.set_xlim(0, df['Count'].max() * 1.15) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.2)
    ax.tick_params(axis='x', labelsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# 8. 时光机 (深度去噪)
# ==========================================
def show_timeline_panel(res):
    st.markdown("### 📅 月度话题时光机 (深度去噪)")
    st.caption("已自动剔除全局最常用的 50 个词，只显示每月的独特话题。")
    
    timeline_counters = res["timeline_counters"]
    timeline_counts = res["timeline_counts"]
    
    if not timeline_counters: 
        st.warning("没有解析到时间数据。")
        return
        
    # 1. 计算全局噪音 (Top 50)
    global_counter = Counter()
    for c in timeline_counters.values():
        global_counter.update(c)
    global_noise_words = set([w for w, c in global_counter.most_common(50)])
    
    timeline_data = []
    sorted_months = sorted(timeline_counters.keys(), reverse=True)
    
    for month in sorted_months:
        month_counter = timeline_counters[month]
        count = timeline_counts[month]
        
        filtered_counter = Counter()
        for w, c in month_counter.items():
            if w not in global_noise_words:
                filtered_counter[w] = c
        
        top_n = filtered_counter.most_common(10)
        top_str = " | ".join([f"{w}" for w, c in top_n])
        
        timeline_data.append({
            "月份": month, 
            "本月特色话题 (Top 10)": top_str, 
            "对话条数": count
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    st.dataframe(df_timeline, use_container_width=True, height=600)

# ==========================================
# 主界面
# ==========================================
st.title("🛸 ChatGPT 深度分析 31.0")

if uploaded_file:
    # 调用解析
    res = parse_and_count_stream(uploaded_file, final_stopwords)
    
    if res:
        st.markdown("### 🧬 聊天基因报告")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{USER_ICON} 你的总发言", f"{res['u_count']} 条")
        col2.metric(f"{USER_ICON} 你的平均长度", f"{res['u_avg']} 字/条")
        col3.metric(f"{AI_ICON} AI 的总回复", f"{res['a_count']} 条")
        delta_val = res['a_avg'] - res['u_avg']
        col4.metric(f"{AI_ICON} AI 的平均长度", f"{res['a_avg']} 字/条", delta=f"{delta_val} (表达欲)" if delta_val > 0 else None)
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["🎨 词云 & 词表", "📊 实力对比", "📅 时光机"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1: st.subheader(f"{USER_ICON} 你的词云"); show_wordcloud_panel(res['u_counter'], user_wc_color, "用户", USER_ICON, max_words_limit, color_intensity)
            with c2: st.subheader(f"{AI_ICON} AI 的词云"); show_wordcloud_panel(res['a_counter'], ai_wc_color, "AI", AI_ICON, max_words_limit, color_intensity)
        with tab2:
            c1, c2 = st.columns(2)
            with c1: show_barchart_panel(res['u_counter'], user_wc_color, "用户", max_words_limit)
            with c2: show_barchart_panel(res['a_counter'], ai_wc_color, "AI", max_words_limit)
        with tab3: 
            show_timeline_panel(res)
else: st.write("👈 请在左侧上传文件开始")