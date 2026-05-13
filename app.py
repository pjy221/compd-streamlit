import streamlit as st
from PIL import Image
import sqlite3
import pandas as pd
import os
import io
import hashlib
from urllib.parse import quote, unquote
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import json
import csv
import tempfile

# 尝试导入 wordcloud
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ======================
# 配置与初始化
# ======================
DB_PATH = "compounds.db"
IMG_DIR = "img"
UPLOAD_DIR = "uploaded_files"
ADMIN_PASSWORD_HASH = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# ======================
# 数据库管理 (包含自动升级逻辑)
# ======================
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # --- 修改：添加 qualitative_ions 字段 ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS compounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cas_number TEXT,
        compound_name_cn TEXT,
        compound_name_en TEXT,
        molecular_weight TEXT,
        molecular_formula TEXT,
        description TEXT,
        threshold_threshold TEXT,
        threshold_detection TEXT,
        threshold_recognition TEXT,
        ion_fragments TEXT,
        qualitative_ions TEXT,          -- ✅ 新增字段
        odor TEXT,
        ri_polar TEXT,
        ri_semi_nonpolar TEXT,
        ri_nonpolar TEXT,
        category TEXT,
        detected_samples TEXT,
        has_aroma INTEGER
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS user_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        action_type TEXT,
        query_params TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        ip_address TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        filename TEXT,
        total_items INTEGER,
        matched_items INTEGER,
        match_rate REAL,
        upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        raw_data_summary TEXT,
        file_path TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS admin_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT
    )
    """)

    c.execute("SELECT count(*) FROM admin_users WHERE username='admin'")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
                  ('admin', ADMIN_PASSWORD_HASH))

    conn.commit()
    conn.close()


init_db()


# ======================
# 辅助函数
# ======================
def log_action(action_type, query_info, session_id=None):
    if not session_id:
        session_id = st.session_state.get("session_id", "unknown")
    if isinstance(query_info, dict):
        query_str = json.dumps(query_info, ensure_ascii=False)
    else:
        query_str = str(query_info)
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_logs (session_id, action_type, query_params, ip_address)
        VALUES (?, ?, ?, ?)
    """, (session_id, action_type, query_str[:2000], "127.0.0.1"))
    conn.commit()
    conn.close()


def log_upload(filename, total, matched, raw_summary, file_path, session_id=None):
    if not session_id:
        session_id = st.session_state.get("session_id", "unknown")
    rate = (matched / total * 100) if total > 0 else 0
    summary_str = json.dumps(raw_summary, ensure_ascii=False)[:1000] if isinstance(raw_summary, dict) else str(
        raw_summary)[:1000]
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO uploads (session_id, filename, total_items, matched_items, match_rate, raw_data_summary, file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (session_id, filename, total, matched, rate, summary_str, file_path))
    conn.commit()
    conn.close()


def search_compounds(cas_number="", compound_name_cn="", category="", has_aroma="", compound_name_en="",
                     detected_samples=""):
    conn = get_connection()
    query = "SELECT * FROM compounds WHERE 1=1"
    params = []
    if cas_number:
        query += " AND cas_number LIKE ?"
        params.append(f"%{cas_number}%")
    if compound_name_cn:
        query += " AND compound_name_cn LIKE ?"
        params.append(f"%{compound_name_cn}%")
    if category:
        query += " AND category LIKE ?"
        params.append(f"%{category}%")
    if compound_name_en:
        query += " AND compound_name_en LIKE ?"
        params.append(f"%{compound_name_en}%")
    if detected_samples:
        query += " AND detected_samples LIKE ?"
        params.append(f"%{detected_samples}%")
    if has_aroma == "带香气":
        query += " AND has_aroma = 1"
    elif has_aroma == "不带香气":
        query += " AND (has_aroma IS NULL OR has_aroma = 0)"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def batch_search_cas(cas_list):
    if not cas_list:
        return pd.DataFrame()
    conn = get_connection()
    clean_list = list(set([str(c).strip() for c in cas_list if str(c).strip()]))
    if not clean_list:
        return pd.DataFrame()
    placeholders = ','.join(['?'] * len(clean_list))
    query = f"SELECT * FROM compounds WHERE cas_number IN ({placeholders})"
    df = pd.read_sql_query(query, conn, params=clean_list)
    conn.close()
    return df


def display_image(cas):
    img_path = os.path.join(IMG_DIR, f"{cas}.png")
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            st.image(img, caption=f"结构图：{cas}", width=300)
        except Exception as e:
            st.error(f"图片加载失败：{e}")
    else:
        st.info(f"图片不存在：{cas}.png")


# ======================
# CAS 预处理（仅用于修复上传文件）
# ======================
def fix_cas_protect(input_file, output_file, cas_column, delimiter=',', has_header=True, encoding='utf-8'):
    """
    将 CAS 列的值包装为 Excel 文本公式 ="..."，防止被识别为日期。
    此函数仅用于修复用户上传的原始文件。
    """
    with open(input_file, 'r', newline='', encoding=encoding) as infile:
        reader = csv.reader(infile, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        return

    if isinstance(cas_column, str):
        if not has_header:
            raise ValueError("文件无表头，请使用列索引（int）定位 CAS 列")
        header = rows[0]
        try:
            col_idx = header.index(cas_column)
        except ValueError:
            raise ValueError(f"列名 '{cas_column}' 不在表头中")
    else:
        col_idx = cas_column
        if col_idx < 0 or col_idx >= len(rows[0]):
            raise ValueError(f"列索引 {col_idx} 超出范围")

    start_row = 1 if has_header else 0
    for i in range(start_row, len(rows)):
        row = rows[i]
        if col_idx < len(row):
            original = row[col_idx].strip()
            if original:
                row[col_idx] = f'="{original}"'

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)


def preprocess_uploaded_csv(uploaded_file, cas_column_name="CAS#"):
    """
    对上传的 CSV 文件进行预处理：
      1. 保存原文件，调用 fix_cas_protect 生成修复后的临时 CSV
      2. 读取修复后的 CSV，并从公式中提取真实 CAS 号
    返回处理后的 DataFrame（已提取真实 CAS 值）。
    """
    # 保存原始上传文件
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_in:
        tmp_in.write(uploaded_file.getvalue())
        tmp_in_path = tmp_in.name
    # 生成修复后的临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8-sig') as tmp_out:
        tmp_out_path = tmp_out.name
    try:
        fix_cas_protect(tmp_in_path, tmp_out_path, cas_column=cas_column_name, delimiter=',', has_header=True,
                        encoding='utf-8')
        # 读取修复后的 CSV（所有单元格保持文本）
        df_repaired = pd.read_csv(tmp_out_path, dtype=str, keep_default_na=False)
    finally:
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)

    # 从公式 ="xxx" 中提取真实 CAS 号
    if cas_column_name in df_repaired.columns:
        def extract_cas(val):
            val = str(val).strip()
            if val.startswith('="') and val.endswith('"'):
                return val[2:-1]
            return val

        df_repaired[cas_column_name] = df_repaired[cas_column_name].apply(extract_cas)
    return df_repaired


# ======================
# 页面：隐藏的管理后台
# ======================
def render_admin_dashboard():
    st.set_page_config(page_title="系统管理后台", layout="wide")
    if "is_admin_logged_in" not in st.session_state:
        st.session_state.is_admin_logged_in = False
    if not st.session_state.is_admin_logged_in:
        st.markdown("## 管理员登录")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            password = st.text_input("请输入访问密码", type="password", placeholder="默认密码: password")
            if st.button("登录"):
                input_hash = hashlib.sha256(password.encode()).hexdigest()
                if input_hash == ADMIN_PASSWORD_HASH:
                    st.session_state.is_admin_logged_in = True
                    st.success("登录成功！")
                    st.rerun()
                else:
                    st.error("密码错误")
        st.stop()
    st.markdown("## 系统数据管理")
    if st.button("退出登录"):
        st.session_state.is_admin_logged_in = False
        st.rerun()
    tab2, tab3, tab4 = st.tabs(["活跃度与转化", "用户上传记录", "所有用户查询日志"])
    conn = get_connection()
    with tab2:
        st.markdown("### 活跃度与转化分析")
        logs_all = pd.read_sql_query("SELECT session_id, action_type, timestamp FROM user_logs", conn)
        if not logs_all.empty:
            logs_all['timestamp'] = pd.to_datetime(logs_all['timestamp'])
            recent_logs = logs_all[logs_all['timestamp'] > (datetime.now() - timedelta(days=7))]
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            total_searches = len(logs_all[logs_all['action_type'] == 'search'])
            unique_sessions = logs_all['session_id'].nunique()
            detail_views = len(logs_all[logs_all['action_type'] == 'detail_view'])
            uploads_df = pd.read_sql_query("SELECT * FROM uploads", conn)
            batch_users = uploads_df['session_id'].nunique() if not uploads_df.empty else 0
            with kpi_col1:
                st.metric("总查询次数", total_searches)
            with kpi_col2:
                st.metric("独立用户数 (UV)", unique_sessions)
            with kpi_col3:
                conv_rate = (detail_views / total_searches * 100) if total_searches > 0 else 0
                st.metric("详情转化率", f"{conv_rate:.1f}%")
            with kpi_col4:
                st.metric("批量上传用户", batch_users)
            st.subheader("近7天查询趋势")
            if not recent_logs.empty:
                daily_counts = recent_logs[recent_logs['action_type'] == 'search'].groupby(
                    recent_logs['timestamp'].dt.date).size().reset_index(name='counts')
                daily_counts['timestamp'] = pd.to_datetime(daily_counts['timestamp'])
                fig_line = px.line(daily_counts, x='timestamp', y='counts', markers=True, title="每日查询量走势",
                                   labels={'timestamp': '日期', 'counts': '次数'})
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("近7天无数据。")
            st.markdown(f"""
            #### 分析解读
            - **转化率分析**: 当前转化率为 **{conv_rate:.1f}%**。
              - 若低于 20%：说明搜索结果列表可能缺乏吸引力，或搜索不够精准。
              - 若高于 50%：说明用户目的性很强，搜索非常精准。
            - **用户行为**: 共有 **{batch_users}** 位用户使用了批量上传功能，占比 **{batch_users / unique_sessions * 100 if unique_sessions > 0 else 0:.1f}%**。
            """)
        else:
            st.info("暂无足够数据生成报表。")
    with tab3:
        st.markdown("### 用户上传样本与匹配详情")
        uploads_df = pd.read_sql_query("SELECT * FROM uploads ORDER BY upload_time DESC", conn)
        if not uploads_df.empty:
            display_cols = ['upload_time', 'session_id', 'filename', 'total_items', 'matched_items', 'match_rate',
                            'file_path']
            available_cols = [col for col in display_cols if col in uploads_df.columns]
            rename_map = {
                'upload_time': '上传时间',
                'session_id': '用户ID',
                'filename': '文件名',
                'total_items': '总条数',
                'matched_items': '匹配条数',
                'match_rate': '匹配率 (%)',
                'file_path': '文件路径'
            }
            df_display = uploads_df[available_cols].copy().rename(columns=rename_map)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            if 'match_rate' in uploads_df.columns:
                fig_hist = px.histogram(uploads_df, x="match_rate", nbins=20, title="上传样本匹配率分布",
                                        labels={"match_rate": "匹配率 (%)"}, color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("### 文件内容预览")
            if 'filename' in uploads_df.columns:
                selected_file = st.selectbox("选择要预览的文件", uploads_df['filename'].tolist(),
                                             format_func=lambda x: f"{x}")
                if selected_file:
                    row = uploads_df[uploads_df['filename'] == selected_file].iloc[0]
                    file_path = row.get('file_path') if 'file_path' in row else None
                    if file_path and os.path.exists(file_path):
                        st.info(f"文件物理路径：`{file_path}`")
                        try:
                            if file_path.endswith('.csv'):
                                df_preview = pd.read_csv(file_path)
                                st.write(f"**前 10 行预览 (共 {len(df_preview)} 行):**")
                                st.dataframe(df_preview.head(10), use_container_width=True)
                                csv_buffer = io.StringIO()
                                df_preview.to_csv(csv_buffer, index=False)
                                st.download_button("下载原始文件", csv_buffer.getvalue(), file_name=selected_file,
                                                   mime="text/csv")
                            elif file_path.endswith('.txt'):
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                st.text_area("文件内容预览", content, height=300)
                                st.download_button("下载原始文件", content, file_name=selected_file, mime="text/plain")
                            else:
                                st.warning("暂不支持预览此文件格式，但可下载。")
                                with open(file_path, 'rb') as f:
                                    st.download_button("下载原始文件", f.read(), file_name=selected_file)
                        except Exception as e:
                            st.error(f"读取文件失败：{e}")
                    else:
                        st.error("文件未在服务器找到（可能已被删除、路径错误或旧数据无路径）。")
        else:
            st.info("暂无用户上传记录。")
    with tab4:
        st.markdown("### 详细用户操作日志")
        logs_detail = pd.read_sql_query("SELECT * FROM user_logs ORDER BY timestamp DESC LIMIT 100", conn)
        if not logs_detail.empty:
            logs_detail['action_type_cn'] = logs_detail['action_type'].map({
                'search': '搜索',
                'detail_view': '查看详情',
                'batch_upload': '批量上传'
            }).fillna(logs_detail['action_type'])
            show_df = logs_detail[['timestamp', 'session_id', 'action_type_cn', 'query_params', 'ip_address']].copy()
            show_df.columns = ['操作时间', '用户会话ID', '操作类型', '查询参数详情', 'IP地址']
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无日志。")
    conn.close()


# ======================
# 主程序
# ======================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"user_{os.urandom(4).hex()}"

query_params = st.query_params
page = query_params.get("page", "")
detail_cas = query_params.get("detail", None)

if page == "admin":
    render_admin_dashboard()
    st.stop()

if detail_cas:
    cas = unquote(detail_cas)
    log_action("detail_view", {"cas": cas})
    conn = get_connection()
    df_detail = pd.read_sql_query("SELECT * FROM compounds WHERE cas_number = ?", conn, params=[cas])
    conn.close()
    if df_detail.empty:
        st.set_page_config(page_title="未找到", layout="centered")
        st.error("未找到该化合物记录。")
        if st.button("返回查询列表"):
            st.query_params.clear()
            st.rerun()
    else:
        row = df_detail.iloc[0].to_dict()
        st.set_page_config(page_title=f"详情 - {row.get('compound_name_cn', cas)}", layout="wide")
        st.markdown("### 化合物详情")
        if st.button("返回查询列表"):
            st.query_params.clear()
            st.rerun()
        col_img, col_detail = st.columns([1, 2])
        with col_img:
            display_image(cas)
        with col_detail:
            # --- 修改：更新字段标签和顺序 ---
            fields = [
                ("CAS号", "cas_number"),
                ("中文名", "compound_name_cn"),
                ("英文名", "compound_name_en"),
                ("分子量", "molecular_weight"),
                ("分子式", "molecular_formula"),
                ("描述与性状", "description"),                # ✅ 原“描述”
                ("阈值-阈值", "threshold_threshold"),
                ("阈值-觉察 (d)", "threshold_detection"),
                ("阈值-识别 (r)", "threshold_recognition"),
                ("离子碎片", "ion_fragments"),
                ("定性特征离子", "qualitative_ions"),        # ✅ 新增
                ("气味", "odor"),
                ("保留指数-极性", "ri_polar"),
                ("保留指数-半非极性", "ri_semi_nonpolar"),
                ("保留指数-非极性", "ri_nonpolar"),
                ("分类", "category"),
                ("检出样品", "detected_samples"),
                ("是否有气味", "has_aroma")                  # ✅ 原“是否有香气”
            ]
            for label, key in fields:
                val = row.get(key, "")
                if key == "has_aroma":
                    val = "是" if val == 1 else "否"
                st.text(f"{label}: {val}")
    st.stop()

# ======================
# 正常主页（重构为三个 Tab）
# ======================
st.set_page_config(page_title="海天项目成果共享平台", layout="wide")
st.markdown(
    """
    <div style="text-align: left; margin-bottom: 1rem; border-bottom: 2px solid #eee; padding-bottom: 10px;">
        <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">
            海天项目成果共享平台
        </div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #34495e;">
            发酵样品香气快速解析系统
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("注：阈值单位为 mg/kg；括号内为年份；若无特殊说明，介质为水。")

# 初始化 session state
if "query" not in st.session_state:
    st.session_state.query = {
        "cas_number": "", "compound_name_cn": "", "category": "", "has_aroma": "",
        "compound_name_en": "", "detected_samples": "", "batch_mode": False, "batch_cas_list": []
    }
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "csv_processed" not in st.session_state:
    st.session_state.csv_processed = False

tab_batch, tab_single, tab_coming_soon = st.tabs(["定性分析", "单条查询", "其他"])

# ======================
# Tab 1: 定性分析-批量导入查询（仅支持CSV）
# ======================
with tab_batch:
    st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold;line-height: 2.4; color: #333;">
            批量导入查询（CSV格式）
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="font-size: 18px; font-weight: normal; color: #333;">
            上传设备导出的原始CSV文件，系统将自动解析并生成可预览、下载的查询结果。<br>
注：系统会自动修复 Excel 将 CAS 号误识别为日期的问题。<br>
<br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 文件上传区域（仅CSV）
    uploaded_csv = st.file_uploader("上传含 'CAS#' 列的 CSV 表格 (.csv)", type=["csv"], key="csv_uploader_batch")

    # --- 处理 CSV（使用预处理修复 CAS 列）---
    if uploaded_csv is not None and not st.session_state.get("csv_processed", False):
        try:
            # 调用预处理函数，自动修复日期误判问题
            df_input = preprocess_uploaded_csv(uploaded_csv, cas_column_name="CAS#")
            if "CAS#" not in df_input.columns:
                st.error("CSV 文件必须包含 'CAS#' 列！")
            else:
                cas_series = df_input["CAS#"].dropna().astype(str).str.strip()
                cas_list = cas_series[cas_series != ""].tolist()
                if not cas_list:
                    st.warning("CAS# 列中没有有效数据。")
                else:
                    df_db_results = batch_search_cas(cas_list)
                    matched_count = df_db_results.shape[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_filename = f"{timestamp}_{uploaded_csv.name}"
                    save_path = os.path.join(UPLOAD_DIR, safe_filename)
                    # 保存预处理后的数据
                    df_input.to_csv(save_path, index=False, encoding='utf-8-sig')
                    log_upload(safe_filename, len(cas_list), matched_count, df_input.head(5).to_dict(), save_path)
                    if df_db_results.empty:
                        df_merged = df_input.copy()
                        df_merged['匹配状态'] = '未匹配'
                    else:
                        df_db_results["cas_number"] = df_db_results["cas_number"].astype(str)
                        df_input["CAS#"] = df_input["CAS#"].astype(str)
                        df_merged = pd.merge(df_input, df_db_results, left_on="CAS#", right_on="cas_number", how="left")
                        df_merged['匹配状态'] = df_merged['compound_name_cn'].apply(
                            lambda x: '匹配成功' if pd.notna(x) else '未匹配')
                        if "cas_number" in df_merged.columns:
                            df_merged.drop(columns=["cas_number"], inplace=True)
                    columns_to_remove = ["id", "molecular_formula", "compound_name_en", "ri_semi_nonpolar",
                                         "ri_nonpolar"]
                    cols_to_drop = [col for col in columns_to_remove if col in df_merged.columns]
                    if cols_to_drop:
                        df_merged = df_merged.drop(columns=cols_to_drop)
                    st.session_state.csv_merged_df = df_merged
                    st.session_state.csv_processed = True
                    st.session_state.csv_filename = safe_filename
                    st.success(f"成功处理 CSV：共 {len(cas_list)} 个，匹配 {matched_count} 条。")
                    st.rerun()
        except Exception as e:
            st.error(f"CSV 处理失败：{e}")
            st.session_state.csv_processed = False

    # 显示批量查询结果
    if "csv_merged_df" in st.session_state:
        st.subheader("批量查询结果")
        df_merged = st.session_state.csv_merged_df
        match_rate = (df_merged['匹配状态'] == '匹配成功').sum() / len(df_merged) * 100
        st.metric("本次上传匹配率", f"{match_rate:.1f}%")
        st.dataframe(df_merged, use_container_width=True, hide_index=True)

        # 下载 CSV 文件（使用制表符前缀强制 Excel 识别为文本）
        output = io.BytesIO()
        df_download = df_merged.copy()

        # 为 CAS 号添加制表符前缀（Excel 打开时不会显示制表符，但会强制识别为文本）
        if "CAS#" in df_download.columns:
            def clean_cas_value(val):
                val_str = str(val).strip()
                if val_str.startswith('="') and val_str.endswith('"'):
                    val_str = val_str[2:-1]
                # 添加制表符前缀（使用 \t），强制 Excel 识别为文本
                return f'\t{val_str}'


            df_download["CAS#"] = df_download["CAS#"].apply(clean_cas_value)

        # 写入 CSV，使用 quoting=1 确保所有字段加引号
        df_download.to_csv(output, index=False, encoding='utf-8-sig', quoting=1)

        st.download_button(
            label="下载合并后的结果文件",
            data=output.getvalue(),
            file_name=f"result_{st.session_state.csv_filename}",
            mime="text/csv",
            help="下载的文件中 CAS 号已强制格式化为文本，用 Excel 打开时会正常显示为文本格式"
        )

        if st.button("清除结果", key="clear_batch_result"):
            for k in ["csv_merged_df", "csv_processed", "csv_filename"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

# ======================
# Tab 2: 单条查询
# ======================
with tab_single:
    col1, col2, col3 = st.columns(3)
    with col1:
        cas_number = st.text_input("CAS号", value=st.session_state.query["cas_number"], key="input_cas")
        has_aroma = st.selectbox("香气", ["", "带香气", "不带香气"], key="input_aroma",
                                 index=["", "带香气", "不带香气"].index(st.session_state.query["has_aroma"]) if
                                 st.session_state.query["has_aroma"] in ["", "带香气", "不带香气"] else 0)
    with col2:
        compound_name_cn = st.text_input("中文名", value=st.session_state.query["compound_name_cn"], key="input_cn")
        category = st.text_input("种类", value=st.session_state.query["category"], key="input_cat")
    with col3:
        compound_name_en = st.text_input("英文名", value=st.session_state.query["compound_name_en"], key="input_en")
        detected_samples = st.text_input("检出样品", value=st.session_state.query["detected_samples"],
                                         key="input_detected")

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("查询"):
            log_action("search", {"cas": cas_number, "cn": compound_name_cn, "en": compound_name_en})
            st.session_state.query.update({
                "cas_number": cas_number, "compound_name_cn": compound_name_cn, "category": category,
                "has_aroma": has_aroma, "compound_name_en": compound_name_en,
                "detected_samples": detected_samples, "batch_mode": False
            })
            st.session_state.file_processed = False
            st.session_state.csv_processed = False
            if "csv_merged_df" in st.session_state:
                del st.session_state.csv_merged_df
            st.rerun()
    with btn_col2:
        if st.button("清除", type="secondary"):
            st.session_state.query = {
                "cas_number": "", "compound_name_cn": "", "category": "", "has_aroma": "",
                "compound_name_en": "", "detected_samples": "", "batch_mode": False, "batch_cas_list": []
            }
            st.session_state.file_processed = False
            st.session_state.csv_processed = False
            for key in ["input_cas", "input_aroma", "input_cn", "input_cat", "input_en", "input_detected"]:
                if key in st.session_state: del st.session_state[key]
            if "csv_merged_df" in st.session_state:
                del st.session_state.csv_merged_df
            st.rerun()

# ======================
# 显示单条查询结果（仅在单条查询 tab 中显示）
# ======================
with tab_single:
    df = pd.DataFrame()
    if st.session_state.query.get("batch_mode", False):
        cas_list = st.session_state.query["batch_cas_list"]
        df = batch_search_cas(cas_list) if cas_list else pd.DataFrame()
    else:
        q = st.session_state.query
        search_params = {
            "cas_number": q.get("cas_number", ""),
            "compound_name_cn": q.get("compound_name_cn", ""),
            "category": q.get("category", ""),
            "has_aroma": q.get("has_aroma", ""),
            "compound_name_en": q.get("compound_name_en", ""),
            "detected_samples": q.get("detected_samples", "")
        }
        if any(search_params.values()):
            df = search_compounds(**search_params)

    if not df.empty:
        df["has_aroma_display"] = df["has_aroma"].apply(lambda x: "是" if x == 1 else "否")
        st.subheader(f"查询结果（共 {len(df)} 条）")
        for idx, row in df.iterrows():
            cas = row["cas_number"]
            name = row.get("compound_name_cn", "未知名称")
            desc = row.get("description", "")
            aroma = row.get("has_aroma_display", "否")
            with st.container(border=True):
                col_left, col_right = st.columns([4, 1])
                with col_left:
                    st.markdown(f"**{name}** （CAS: `{cas}`）")
                    if desc:
                        st.caption(desc)
                    st.markdown(
                        f"**分类**: {row.get('category', '—')} | **检出样品**: {row.get('detected_samples', '—')} | **有香气**: {aroma}")
                with col_right:
                    if st.button("查看详情", key=f"view_{cas}_{idx}"):
                        st.query_params["detail"] = quote(cas)
                        st.rerun()
    else:
        has_search_params = any([
            st.session_state.query.get("cas_number"),
            st.session_state.query.get("compound_name_cn"),
            st.session_state.query.get("category"),
            st.session_state.query.get("has_aroma"),
            st.session_state.query.get("compound_name_en"),
            st.session_state.query.get("detected_samples")
        ])
        if not st.session_state.query.get("batch_mode") and has_search_params:
            st.info("未找到匹配的记录。")

# ======================
# Tab 3: 功能未开放
# ======================
with tab_coming_soon:
    st.markdown('<p style="font-size: 16px; font-weight: normal; ">还未开发定量功能，敬请期待</p>',
                unsafe_allow_html=True)