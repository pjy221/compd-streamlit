import streamlit as st
from PIL import Image
import sqlite3
import pandas as pd
import os

# ======================
# 配置
# ======================
DB_PATH = "compounds.db"
IMG_DIR = "img"  # 和 app.py 同级目录下的 img 文件夹

# 确保图片目录存在（部署时 GitHub 要包含 img/）
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# ======================
# 数据库连接与查询
# ======================
def get_connection():
    return sqlite3.connect(DB_PATH)

def search_compounds(
    cas_number="",
    compound_name_cn="",
    category="",
    has_aroma="",  # "", "带香气", "不带香气"
    compound_name_en=""
):
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

    # 特殊处理 has_aroma
    if has_aroma == "带香气":
        query += " AND has_aroma = 1"
    elif has_aroma == "不带香气":
        query += " AND (has_aroma IS NULL OR has_aroma = 0)"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def batch_search_cas(cas_list):
    """批量查询 CAS 列表"""
    if not cas_list:
        return pd.DataFrame()
    conn = get_connection()
    placeholders = ','.join(['?'] * len(cas_list))
    query = f"SELECT * FROM compounds WHERE cas_number IN ({placeholders})"
    df = pd.read_sql_query(query, conn, params=cas_list)
    conn.close()
    return df

# ======================
# 显示工具函数
# ======================
def display_image(cas):
    """在 Streamlit 中显示图片"""
    img_path = os.path.join(IMG_DIR, f"{cas}.png")
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            st.image(img, caption=f"结构图: {cas}.png", width=300)
        except Exception as e:
            st.error(f"图片加载失败: {e}")
    else:
        st.info(f"图片不存在: {cas}.png")

def show_detail_modal(row):
    """Streamlit 没有弹窗，改用展开详情"""
    with st.expander(f"🔍 查看详情: {row['cas_number']}"):
        cols = st.columns([1, 2])
        with cols[0]:
            st.markdown("### 基本信息")
            fields = [
                ("CAS号", "cas_number"),
                ("分子量", "molecular_weight"),
                ("分子式", "molecular_formula"),
                ("英文名", "compound_name_en"),
                ("中文名", "compound_name_cn"),
                ("描述", "description"),
                ("阈值-阈值", "threshold_threshold"),
                ("阈值-觉察 (d)", "threshold_detection"),
                ("阈值-识别 (r)", "threshold_recognition"),
                ("离子碎片", "ion_fragments"),
                ("气味", "odor"),
                ("保留指数-半标准非极性", "ri_semi_nonpolar"),
                ("保留指数-非极性", "ri_nonpolar"),
                ("保留指数-极性", "ri_polar"),
                ("分类", "category"),
                ("检出样品", "detected_samples"),
                ("是否有香气", "has_aroma_display")
            ]
            for label, key in fields:
                val = row.get(key, "")
                if key == "has_aroma_display":
                    val = "是" if row.get("has_aroma") == 1 else "否"
                st.text(f"{label}: {val}")

        with cols[1]:
            st.markdown("### 分子结构图")
            display_image(row["cas_number"])

# ======================
# 主界面
# ======================
st.set_page_config(page_title="化合物数据库查询系统", layout="wide")
st.title("🧪 化合物数据库查询系统")

# 提示信息
st.caption("注：阈值单位为mg/kg；括号内为年份；若无特殊说明，介质为水。")

# 查询条件（两行布局）
col1, col2, col3 = st.columns(3)
with col1:
    cas_number = st.text_input("CAS号", key="cas")
    has_aroma = st.selectbox("香气", ["", "带香气", "不带香气"], key="aroma")
with col2:
    compound_name_cn = st.text_input("中文名", key="cn")
    category = st.text_input("种类", key="cat")
with col3:
    compound_name_en = st.text_input("英文名", key="en")

# 按钮区
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
with btn_col1:
    do_search = st.button("🔍 查询")
with btn_col2:
    do_clear = st.button("🧹 清除")
with btn_col3:
    uploaded_file = st.file_uploader("📂 批量查询 (上传 CAS 列表.txt)", type=["txt"])

# 清除逻辑
if do_clear:
    st.session_state.clear()
    st.rerun()

# 批量查询
if uploaded_file is not None:
    try:
        cas_list = [line.decode("utf-8").strip() for line in uploaded_file if line.strip()]
        cas_list = [c for c in cas_list if c]
        if cas_list:
            df_batch = batch_search_cas(cas_list)
            found_cas = set(df_batch["cas_number"].tolist())
            missing = [c for c in cas_list if c not in found_cas]
            st.success(f"批量查询完成：{len(found_cas)}/{len(cas_list)} 个匹配")
            if missing:
                st.warning(f"未找到的 CAS: {', '.join(missing)}")
            df = df_batch
        else:
            st.warning("文件为空")
            df = pd.DataFrame()
    except Exception as e:
        st.error(f"批量查询失败: {e}")
        df = pd.DataFrame()
elif do_search:
    # 单条查询
    df = search_compounds(
        cas_number=cas_number,
        compound_name_cn=compound_name_cn,
        category=category,
        has_aroma=has_aroma,
        compound_name_en=compound_name_en
    )
else:
    df = pd.DataFrame()

# 显示结果
if not df.empty:
    # 添加显示用的 has_aroma_display 列（用于表格展示）
    df["has_aroma_display"] = df["has_aroma"].apply(lambda x: "是" if x == 1 else "否")

    # 选择要显示的列（顺序和名称对齐 Tkinter）
    display_columns = {
        "cas_number": "CAS",
        "molecular_weight": "分子量",
        "molecular_formula": "分子式",
        "compound_name_en": "Compound Name",
        "compound_name_cn": "名称",
        "description": "描述",
        "threshold_threshold": "阈值-阈值",
        "threshold_detection": "阈值-觉察d",
        "threshold_recognition": "阈值-识别r",
        "ion_fragments": "离子碎片",
        "odor": "气味",
        "ri_semi_nonpolar": "保留指数-半标准非极性",
        "ri_nonpolar": "保留指数-非极性",
        "ri_polar": "保留指数-极性",
        "category": "分类",
        "detected_samples": "检出样品",
        "has_aroma_display": "是否有香气"
    }

    df_display = df[list(display_columns.keys())].rename(columns=display_columns)

    st.subheader(f"查询结果（共 {len(df)} 条）")
    st.dataframe(df_display, use_container_width=True)

    # 允许用户点击某一行查看详情 + 图片
    selected_cas = st.selectbox("👉 选择一条记录查看结构图和详情", df["cas_number"].tolist(), key="detail_select")
    if selected_cas:
        row = df[df["cas_number"] == selected_cas].iloc[0].to_dict()
        col_img, col_detail = st.columns([1, 2])
        with col_img:
            display_image(selected_cas)
        with col_detail:
            show_detail_modal(row)

else:
    if do_search or uploaded_file is not None:
        st.info("未找到匹配的记录。")