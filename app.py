import streamlit as st
import json
import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import re
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from datetime import datetime

# ==========================================
# 0. 基础配置 & CSS
# ==========================================
st.set_page_config(page_title="ChatGPT 深度分析 19.0", layout="wide", page_icon="🛸")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
<style>
div[data-testid="stColorPicker"] {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    width: 100%;
}
div[data-testid="stColorPicker"] > label {
    width: 100%;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 内置停用词表
# ==========================================
DEFAULT_STOPWORDS = {
    "---", "...", "###", "___", "***", "=>", "->", "<-", "<=", ">=", "**", "__", "~~", "==", "!=", "&&", "||", "```", "`", ">", "|", "[", "]", "{", "}", "(", ")",
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "吗", "吧", "啊", "可以", "这个", "那个", "其实", "因为", "所以", "如果", "但是", 
    "嗯", "哦", "什么", "怎么", "为", "之", "与", "及", "其", "以", "对", "而", "着", "下", "地", "子", "中", "它", "她", "他", "们", "我们", "你们", "他们", "它们", 
    "哈哈", "哈哈哈", "嘿嘿", "呃", "嘛", "呀", "呢", "啦", "哇", "唉", "哎", "哼", "噢", "呗",
    "哪怕", "可能", "觉得", "应该", "这种", "那种", "比如", "或者", "以及", "然后", "虽然", "但是", "不过", "只是", "这样", "那样", "这里", "那里",
    "确实", "真的", "非常", "特别", "比较", "稍微", "只要", "只有", "除非", "几乎", "简直", "似乎", "好像", "大概", "也许",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", "、", "，", "。", "？", "！", "“", "”", "：", "；", "（", "）", "《", "》", "…", "—", "～",
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "when", "where", "how", "why", "which", "who", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "this", "that", "these", "those", "there", "here", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down", "out", "about", "into", "over", "after", "can", "could", "will", "would", "should", "may", "might", "must", "not", "no", "yes", "ok", "okay", "well", "just", "only", "very", "really", "too", "so", "much", "many", "some", "any", "all"
}

# ==========================================
# 2. 核心解析函数
# ==========================================
@st.cache_data
def parse_data(file):
    try:
        data = json.load(file)
    except:
        st.error("文件格式不对，请确保上传的是 JSON 文件")
        return [], [], []
    
    user_data_list = []
    ai_data_list = [] 
    
    for conversation in data:
        mapping = conversation.get('mapping', {})
        create_time = conversation.get('create_time')
        base_dt = datetime.fromtimestamp(create_time) if create_time else None
        
        for node_id, node_data in mapping.items():
            message = node_data.get('message')
            if message and message.get('content') and message.get('author'):
                role = message['author']['role']
                msg_time = message.get('create_time')
                dt = datetime.fromtimestamp(msg_time) if msg_time else base_dt
                
                content_parts = message['content'].get('parts', [])
                text_content = "".join([part for part in content_parts if isinstance(part, str)])
                
                if text_content and dt:
                    item = {"text": text_content, "time": dt}
                    if role == 'user': user_data_list.append(item)
                    elif role == 'assistant': ai_data_list.append(item)
    return user_data_list, ai_data_list

# ==========================================
# 3. 统计函数
# ==========================================
def calculate_stats(data_list):
    if not data_list: return 0, 0, 0
    lengths = [len(d['text']) for d in data_list]
    total_len = sum(lengths)
    avg_len = total_len / len(lengths)
    max_len = max(lengths)
    return len(data_list), int(avg_len), max_len

# ==========================================
# 4. Emoji 形状生成器
# ==========================================
@st.cache_data
def create_emoji_mask(emoji_char):
    if emoji_char.startswith("⬛"): return None
    char = emoji_char.split(" ")[0]
    img_size = (1000, 1000)
    img = Image.new("RGBA", img_size, (255,255,255,0)) 
    draw = ImageDraw.Draw(img)
    font_path = "seguiemj.ttf"
    try:
        font = ImageFont.truetype(font_path, 800)
    except:
        try: font = ImageFont.truetype("C:/Windows/Fonts/seguiemj.ttf", 800)
        except: return None
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except: w, h = 800, 800
    x, y = (img_size[0] - w) / 2, (img_size[1] - h) / 2 - 50
    draw.text((x, y), char, font=font, embedded_color=True) 
    alpha = img.split()[-1]
    alpha_array = np.array(alpha)
    final_mask = np.full(alpha_array.shape, 255, dtype=np.uint8)
    final_mask[alpha_array > 0] = 0
    return final_mask

# ==========================================
# 5. 颜色截断器
# ==========================================
def get_truncated_cmap(cmap_name, min_val=0.0, max_val=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap_name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n))
    )
    return new_cmap

# ==========================================
# 6. 界面侧边栏
# ==========================================
emoji_options = ["⬛ 矩形 (无形状)", "🧠 大脑", "🗨 气泡", "☁ 云朵", "❤ 爱心", "🦊 狐狸", "🐟 小鱼", "🐱 猫咪", "🦉 猫头鹰", "🐶 小狗"]
wordcloud_colormaps = {"Blues (蓝)": "Blues", "Oranges (橙)": "Oranges", "Reds (红)": "Reds", "Greens (绿)": "Greens", "Purples (紫)": "Purples", "viridis (极光)": "viridis", "magma (岩浆)": "magma", "cool (冷色)": "cool", "autumn (秋色)": "autumn"}
USER_ICON = "👾" 
AI_ICON = "🦾"

with st.sidebar:
    st.title("⚙️ 设置面板 v19.0")
    uploaded_file = st.file_uploader("1. 上传 conversations.json", type=['json'])
    
    st.markdown("---")
    st.header("🎨 视觉定制")
    
    st.write("🌈 **颜色浓度**")
    st.caption("调节滑块，让颜色更深或更浅 👇")
    color_intensity = st.slider("去除浅色比例", 0.0, 0.6, 0.3)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<h4 style='text-align: center;'>{USER_ICON} 你</h4>", unsafe_allow_html=True)
        user_shape = st.selectbox("你的形状", emoji_options, index=1)
        user_wc_color = st.selectbox("你的色系", list(wordcloud_colormaps.keys()), index=0)
        
    with c2:
        st.markdown(f"<h4 style='text-align: center;'>{AI_ICON} AI</h4>", unsafe_allow_html=True)
        ai_shape = st.selectbox("AI 的形状", emoji_options, index=2)
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
# 7. 词云面板
# ==========================================
def show_wordcloud_panel(data_list, cmap_name, shape_mask, title, icon, limit, min_val):
    if not data_list: return
    text_list = [d['text'] for d in data_list]
    full_text = " ".join(text_list)
    words = jieba.lcut(full_text)
    filtered_words = [w.strip() for w in words if len(w.strip()) > 1 and w.strip().lower() not in final_stopwords]
    word_counts = Counter(filtered_words)
    
    base_cmap_name = wordcloud_colormaps[cmap_name]
    custom_cmap = get_truncated_cmap(base_cmap_name, min_val=min_val, max_val=1.0)

    try:
        wc = WordCloud(font_path='msyh.ttc', width=1000, height=1000, background_color='white', colormap=custom_cmap, max_words=limit, stopwords=final_stopwords, mask=shape_mask, contour_width=0).generate_from_frequencies(word_counts)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e: st.error(f"生成失败: {e}")
    
    with st.expander(f"📋 查看 {icon} {title} 高频词表", expanded=False):
        st.dataframe(pd.DataFrame(word_counts.most_common(limit), columns=['词语', '次数']), use_container_width=True, height=300)

# ==========================================
# 8. 柱状图面板 (水平渐变 + 大间距)
# ==========================================
def show_barchart_panel(data_list, cmap_name, role_icon, limit):
    if not data_list: return
    text_list = [d['text'] for d in data_list]
    words = [w.strip() for w in jieba.lcut(" ".join(text_list)) if len(w.strip()) > 1 and w.strip().lower() not in final_stopwords]
    common_words = Counter(words).most_common(limit)
    if not common_words: return
    
    df = pd.DataFrame(common_words, columns=['Word', 'Count']).sort_values(by='Count', ascending=True)
    
    # 【增加呼吸感】每行高度预算增加到 0.45 (之前是0.25)
    height_per_row = 0.45
    dynamic_height = max(6, len(df) * height_per_row)
    
    fig, ax = plt.subplots(figsize=(12, dynamic_height))
    
    # 获取渐变 colormap
    base_cmap_name = wordcloud_colormaps[cmap_name]
    # 截断一下，保证左边的浅色也能看见 (从 0.3 开始)
    cmap = get_truncated_cmap(base_cmap_name, 0.3, 0.9)
    
    # 【核心黑科技】绘制单柱水平渐变
    # 遍历每一行数据
    for i, (word, count) in enumerate(zip(df['Word'], df['Count'])):
        # 创建渐变纹理 (1行 256列)
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        # 使用 imshow 绘制图片作为柱子
        # extent = [x_min, x_max, y_min, y_max]
        # y轴位置是 i，我们给它 0.6 的厚度 (上下各0.3)
        ax.imshow(gradient, aspect='auto', cmap=cmap, extent=(0, count, i - 0.3, i + 0.3))
        
        # 添加数字标签 (位置在 count 右侧一点点)
        ax.text(count + (df['Count'].max() * 0.01), i, str(count), va='center', fontsize=12)

    # 设置 Y 轴标签 (词语)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Word'], fontsize=14)
    
    # 美化图表
    ax.set_title(f"{role_icon} Top {limit} 词频统计", fontsize=20, pad=20)
    ax.set_ylim(-0.5, len(df) - 0.5) # 确保上下留白合适
    ax.set_xlim(0, df['Count'].max() * 1.15) # X轴留出空间给数字
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # 去掉左边框，更现代
    ax.spines['bottom'].set_visible(False) # 去掉底边框
    
    # 只保留横向虚线，辅助阅读
    ax.grid(axis='x', linestyle='--', alpha=0.2)
    ax.tick_params(axis='x', labelsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# 9. 时光机
# ==========================================
def show_timeline_panel(user_list):
    st.markdown("### 📅 月度话题时光机 (深度去噪)")
    st.caption("已自动剔除全局最常用的 50 个词，只显示每月的独特话题。")
    
    df_u = pd.DataFrame(user_list)
    if df_u.empty: 
        st.warning("没有解析到时间数据。")
        return
    
    all_text = " ".join(df_u['text'].tolist())
    all_words = jieba.lcut(all_text)
    all_filtered = [w.strip() for w in all_words if len(w.strip()) > 1 and w.strip().lower() not in final_stopwords]
    global_counter = Counter(all_filtered)
    global_noise_words = set([w for w, c in global_counter.most_common(50)])
    
    df_u['month'] = df_u['time'].dt.to_period('M')
    monthly_groups = df_u.groupby('month')
    
    timeline_data = []
    for month, group in monthly_groups:
        month_text = " ".join(group['text'].tolist())
        words = jieba.lcut(month_text)
        filtered = [w.strip() for w in words if len(w.strip()) > 1 and w.strip().lower() not in final_stopwords and w.strip() not in global_noise_words]
        top_n = Counter(filtered).most_common(10) 
        top_str = " | ".join([f"{w}" for w, c in top_n])
        timeline_data.append({"月份": str(month), "本月特色话题 (Top 10)": top_str, "对话条数": len(group)})
    
    df_timeline = pd.DataFrame(timeline_data).sort_values(by="月份", ascending=False)
    st.dataframe(df_timeline, use_container_width=True, height=600)

# ==========================================
# 主界面
# ==========================================
st.title("🛸 ChatGPT 深度分析 19.0")

if uploaded_file:
    user_mask = create_emoji_mask(user_shape)
    ai_mask = create_emoji_mask(ai_shape)
    user_data, ai_data = parse_data(uploaded_file)
    
    u_count, u_avg, u_max = calculate_stats(user_data)
    a_count, a_avg, a_max = calculate_stats(ai_data)
    st.markdown("### 🧬 聊天基因报告")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{USER_ICON} 你的总发言", f"{u_count} 条")
    col2.metric(f"{USER_ICON} 你的平均长度", f"{u_avg} 字/条")
    col3.metric(f"{AI_ICON} AI 的总回复", f"{a_count} 条")
    col4.metric(f"{AI_ICON} AI 的平均长度", f"{a_avg} 字/条", delta=f"{a_avg - u_avg} (表达欲)" if a_avg > u_avg else None)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🎨 词云 & 词表", "📊 实力对比", "📅 时光机"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1: st.subheader(f"{USER_ICON} 你的词云"); show_wordcloud_panel(user_data, user_wc_color, user_mask, "用户", USER_ICON, max_words_limit, color_intensity)
        with c2: st.subheader(f"{AI_ICON} AI 的词云"); show_wordcloud_panel(ai_data, ai_wc_color, ai_mask, "AI", AI_ICON, max_words_limit, color_intensity)
    with tab2:
        c1, c2 = st.columns(2)
        with c1: show_barchart_panel(user_data, user_wc_color, USER_ICON, max_words_limit)
        with c2: show_barchart_panel(ai_data, ai_wc_color, AI_ICON, max_words_limit)
    with tab3: 
        show_timeline_panel(user_data)
else: st.write("👈 请在左侧上传文件开始")