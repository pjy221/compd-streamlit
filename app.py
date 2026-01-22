import streamlit as st
from PIL import Image
import sqlite3
import pandas as pd
import os
from rapidfuzz import fuzz  # ← 新增依赖
import re

# ======================
# 配置
# ======================
DB_PATH = "compounds.db"
IMG_DIR = "img"

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
    placeholders = ','.join(['?'] * len(cas_list))
    query = f"SELECT * FROM compounds WHERE cas_number IN ({placeholders})"
    df = pd.read_sql_query(query, conn, params=cas_list)
    conn.close()
    return df


# ======================
# 新增：模糊过滤函数（基于 rapidfuzz）
# ======================
def fuzzy_filter_dataframe(df, query_text, threshold=65):
    if df.empty or not query_text.strip():
        return df

    # 清洗查询（可选）
    query_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query_text.lower()).strip()
    if not query_clean:
        query_clean = query_text.lower()

    def get_best_score(row):
        texts = [
            str(row.get("compound_name_cn", "")),
            str(row.get("compound_name_en", "")),
            str(row.get("description", "")),
            str(row.get("category", ""))
        ]
        scores = []
        for t in texts:
            t = t.lower()
            scores.extend([
                fuzz.token_sort_ratio(query_clean, t),
                fuzz.partial_ratio(query_clean, t),
                fuzz.token_set_ratio(query_clean, t)
            ])
        return max(scores) if scores else 0

    df["_fuzzy_score"] = df.apply(get_best_score, axis=1)
    filtered_df = df[df["_fuzzy_score"] >= threshold].copy()
    filtered_df = filtered_df.sort_values("_fuzzy_score", ascending=False).head(30)
    return filtered_df.drop(columns=["_fuzzy_score"])


# ======================
# 显示工具函数
# ======================
def display_image(cas):
    img_path = os.path.join(IMG_DIR, f"{cas}.png")
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            st.image(img, caption=f"结构图: {cas}.png", width=300)
        except Exception as e:
            st.error(f"图片加载失败: {e}")
    else:
        st.info(f"图片不存在: {cas}.png")


# ======================
# 主界面
# ======================
st.set_page_config(page_title="化合物数据库查询系统", layout="wide")
st.title("化合物数据库查询系统")
st.caption("注：阈值单位为mg/kg；括号内为年份；若无特殊说明，介质为水。")

# 初始化查询状态（含 free_text）
if "query" not in st.session_state:
    st.session_state.query = {
        "cas_number": "",
        "compound_name_cn": "",
        "category": "",
        "has_aroma": "",
        "compound_name_en": "",
        "free_text": "",  # ← 新增自由关键词字段
        "batch_mode": False,
        "batch_cas_list": []
    }

# 用于跟踪是否已处理上传文件的session_state变量
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# 用于跟踪选中行的session_state变量
if "selected_rows" not in st.session_state:
    st.session_state.selected_rows = []

# 查询条件输入（绑定到 session_state）
col1, col2, col3 = st.columns(3)
with col1:
    cas_number = st.text_input("CAS号", value=st.session_state.query["cas_number"], key="input_cas")
    has_aroma = st.selectbox(
        "香气",
        ["", "带香气", "不带香气"],
        index=["", "带香气", "不带香气"].index(st.session_state.query["has_aroma"])
        if st.session_state.query["has_aroma"] in ["", "带香气", "不带香气"] else 0,
        key="input_aroma"
    )
with col2:
    compound_name_cn = st.text_input("中文名", value=st.session_state.query["compound_name_cn"], key="input_cn")
    category = st.text_input("种类", value=st.session_state.query["category"], key="input_cat")
with col3:
    compound_name_en = st.text_input("英文名", value=st.session_state.query["compound_name_en"], key="input_en")
    free_text = st.text_input(
        "自由关键词（支持模糊匹配）",
        value=st.session_state.query["free_text"],
        key="input_free"
    )

# 按钮区
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
with btn_col1:
    if st.button("查询"):
        st.session_state.query.update({
            "cas_number": cas_number,
            "compound_name_cn": compound_name_cn,
            "category": category,
            "has_aroma": has_aroma,
            "compound_name_en": compound_name_en,
            "free_text": free_text,  # ← 保存自由关键词
            "batch_mode": False
        })
        st.session_state.file_processed = False
        st.session_state.selected_rows = []
        st.rerun()
with btn_col2:
    if st.button("清除", type="secondary"):
        st.session_state.query = {
            "cas_number": "",
            "compound_name_cn": "",
            "category": "",
            "has_aroma": "",
            "compound_name_en": "",
            "free_text": "",  # ← 清除
            "batch_mode": False,
            "batch_cas_list": []
        }
        st.session_state.file_processed = False
        st.session_state.selected_rows = []

        # 清除输入框的特定 session_state 键
        for key in ["input_cas", "input_aroma", "input_cn", "input_cat", "input_en", "input_free"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
with btn_col3:
    uploaded_file = st.file_uploader("批量查询 (上传 CAS 列表.txt)", type=["txt"], key="file_uploader")

# 处理批量上传
if uploaded_file is not None and not st.session_state.file_processed:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        cas_list = [line.strip() for line in content.splitlines() if line.strip()]
        if cas_list:
            st.session_state.query.update({
                "batch_mode": True,
                "batch_cas_list": cas_list,
                "cas_number": "",
                "compound_name_cn": "",
                "category": "",
                "has_aroma": "",
                "compound_name_en": "",
                "free_text": ""
            })
            st.session_state.file_processed = True
            st.session_state.selected_rows = []
            st.success(f"成功读取 {len(cas_list)} 个CAS号")
            st.rerun()
    except Exception as e:
        st.error(f"文件读取失败: {e}")

# 执行查询
if st.session_state.query.get("batch_mode", False):
    cas_list = st.session_state.query["batch_cas_list"]
    df = batch_search_cas(cas_list) if cas_list else pd.DataFrame()
    if not df.empty:
        found_cas = set(df["cas_number"].tolist())
        missing = [c for c in cas_list if c not in found_cas]
        st.success(f"批量查询完成：{len(found_cas)}/{len(cas_list)} 个匹配")
        if missing:
            st.warning(f"未找到的 CAS: {', '.join(missing)}")
    else:
        st.warning("未找到任何匹配记录。")
else:
    # 单条查询（含模糊关键词）
    q = st.session_state.query
    has_conditions = any([
        q["cas_number"], q["compound_name_cn"], q["category"],
        q["has_aroma"], q["compound_name_en"]
    ])

    if has_conditions or q["free_text"].strip():
        # 先用结构化条件初筛
        df = search_compounds(
            cas_number=q["cas_number"],
            compound_name_cn=q["compound_name_cn"],
            category=q["category"],
            has_aroma=q["has_aroma"],
            compound_name_en=q["compound_name_en"]
        )
        # 再用自由关键词模糊过滤
        if q["free_text"].strip():
            df = fuzzy_filter_dataframe(df, q["free_text"], threshold=60)
    else:
        df = pd.DataFrame()

# 显示结果
if not df.empty:
    df["has_aroma_display"] = df["has_aroma"].apply(lambda x: "是" if x == 1 else "否")

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

    event = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if event.selection:
        st.session_state.selected_rows = event.selection.rows

    if st.session_state.selected_rows:
        selected_index = st.session_state.selected_rows[0]
        row = df.iloc[selected_index].to_dict()
        cas = row["cas_number"]

        col_img, col_detail = st.columns([1, 2])
        with col_img:
            display_image(cas)
        with col_detail:
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

else:
    if (
            st.session_state.query.get("batch_mode") or
            any([
                st.session_state.query["cas_number"],
                st.session_state.query["compound_name_cn"],
                st.session_state.query["category"],
                st.session_state.query["has_aroma"],
                st.session_state.query["compound_name_en"],
                st.session_state.query["free_text"]
            ])
    ):
        st.info("未找到匹配的记录。")