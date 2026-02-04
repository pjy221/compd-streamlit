import streamlit as st
from PIL import Image
import sqlite3
import pandas as pd
import os
import io

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
        compound_name_en="",
        detected_samples=""  # ← 新增参数：检出样品模糊查询
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
    if detected_samples:  # ← 新增条件
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
    placeholders = ','.join(['?'] * len(cas_list))
    query = f"SELECT * FROM compounds WHERE cas_number IN ({placeholders})"
    df = pd.read_sql_query(query, conn, params=cas_list)
    conn.close()
    return df


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
st.set_page_config(page_title="海天政府项目研究-化合物数据库查询系统", layout="wide")
st.markdown(
    """
    <div style="text-align: left; margin-bottom: 1rem;">
        <div style="font-size: 2rem; font-weight: bold; line-height: 1.6;">
            海天政府项目研究
        </div>
        <div style="font-size: 2rem; font-weight: bold; line-height: 1.6; ">
            化合物数据库查询系统
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("注：阈值单位为mg/kg；括号内为年份；若无特殊说明，介质为水。")

# 初始化查询状态
if "query" not in st.session_state:
    st.session_state.query = {
        "cas_number": "",
        "compound_name_cn": "",
        "category": "",
        "has_aroma": "",
        "compound_name_en": "",
        "detected_samples": "",
        "batch_mode": False,
        "batch_cas_list": []
    }

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "csv_processed" not in st.session_state:
    st.session_state.csv_processed = False

if "selected_rows" not in st.session_state:
    st.session_state.selected_rows = []

# 查询条件输入
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
    detected_samples = st.text_input("检出样品", value=st.session_state.query["detected_samples"], key="input_detected")

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
            "detected_samples": detected_samples,
            "batch_mode": False
        })
        st.session_state.file_processed = False
        st.session_state.csv_processed = False
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
            "detected_samples": "",
            "batch_mode": False,
            "batch_cas_list": []
        }
        st.session_state.file_processed = False
        st.session_state.csv_processed = False
        st.session_state.selected_rows = []
        for key in ["input_cas", "input_aroma", "input_cn", "input_cat", "input_en", "input_detected"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
with btn_col3:
    uploaded_file = st.file_uploader("批量查询 (上传 CAS 列表.txt)", type=["txt"], key="file_uploader")
    uploaded_csv = st.file_uploader("批量查询 (上传含 CAS# 的 CSV)", type=["csv"], key="csv_uploader")

# 处理 .txt 批量上传
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
                "detected_samples": ""
            })
            st.session_state.file_processed = True
            st.session_state.selected_rows = []
            st.success(f"成功读取 {len(cas_list)} 个CAS号")
            st.rerun()
    except Exception as e:
        st.error(f"文件读取失败: {e}")

# 处理 CSV 上传
if uploaded_csv is not None and not st.session_state.get("csv_processed", False):
    try:
        df_input = pd.read_csv(uploaded_csv, dtype=str)

        if "CAS#" not in df_input.columns:
            st.error("CSV 文件必须包含 'CAS#' 列！")
        else:
            cas_series = df_input["CAS#"].dropna().astype(str).str.strip()
            cas_list = cas_series[cas_series != ""].tolist()

            if not cas_list:
                st.warning("CAS# 列中没有有效数据。")
            else:
                df_db_results = batch_search_cas(cas_list)

                if df_db_results.empty:
                    st.warning("数据库中未找到任何匹配的 CAS 号。")
                    df_merged = df_input.copy()
                else:
                    df_db_results["cas_number"] = df_db_results["cas_number"].astype(str)
                    df_input["CAS#"] = df_input["CAS#"].astype(str)
                    df_merged = pd.merge(
                        df_input,
                        df_db_results,
                        left_on="CAS#",
                        right_on="cas_number",
                        how="left"
                    )
                    if "cas_number" in df_merged.columns:
                        df_merged.drop(columns=["cas_number"], inplace=True)

                # ===== 删除指定列（导出时隐藏）=====
                columns_to_remove = [
                    "id",
                    "cas_number",
                    "molecular_formula",
                    "compound_name_en",
                    "ri_semi_nonpolar",
                    "ri_nonpolar"
                ]
                cols_to_drop = [col for col in columns_to_remove if col in df_merged.columns]
                if cols_to_drop:
                    df_merged = df_merged.drop(columns=cols_to_drop)

                st.session_state.csv_merged_df = df_merged
                st.session_state.csv_processed = True
                st.session_state.csv_filename = uploaded_csv.name

                st.success(f"成功处理 CSV 文件，共 {len(cas_list)} 个 CAS 号，匹配到 {df_db_results.shape[0]} 条记录。")
                st.rerun()

    except Exception as e:
        st.error(f"CSV 处理失败: {e}")
        st.session_state.csv_processed = False

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
    q = st.session_state.query
    if any([
        q["cas_number"],
        q["compound_name_cn"],
        q["category"],
        q["has_aroma"],
        q["compound_name_en"],
        q["detected_samples"]
    ]):
        df = search_compounds(
            cas_number=q["cas_number"],
            compound_name_cn=q["compound_name_cn"],
            category=q["category"],
            has_aroma=q["has_aroma"],
            compound_name_en=q["compound_name_en"],
            detected_samples=q["detected_samples"]
        )
    else:
        df = pd.DataFrame()

# 显示单条/批量查询结果
if not df.empty:
    df["has_aroma_display"] = df["has_aroma"].apply(lambda x: "是" if x == 1 else "否")

    # ===== 修改：只显示需要的列（隐藏指定字段）=====
    display_columns = {
        "molecular_weight": "分子量",
        "compound_name_cn": "名称",
        "description": "描述",
        "threshold_threshold": "阈值-阈值",
        "threshold_detection": "阈值-觉察d",
        "threshold_recognition": "阈值-识别r",
        "ion_fragments": "离子碎片",
        "odor": "气味",
        "ri_polar": "保留指数-极性",
        "category": "分类",
        "detected_samples": "检出样品",
        "has_aroma_display": "是否有香气"
    }

    # 确保只保留存在的列
    available_cols = [col for col in display_columns.keys() if col in df.columns]
    df_display = df[available_cols].rename(columns={k: v for k, v in display_columns.items() if k in available_cols})

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
        cas = row.get("cas_number", "")

        col_img, col_detail = st.columns([1, 2])
        with col_img:
            if cas:
                display_image(cas)
            else:
                st.info("无 CAS 号，无法加载图片")
        with col_detail:
            st.markdown("### 基本信息")
            # ===== 修改：详情页也隐藏指定字段 =====
            fields = [
                ("分子量", "molecular_weight"),
                ("中文名", "compound_name_cn"),
                ("描述", "description"),
                ("阈值-阈值", "threshold_threshold"),
                ("阈值-觉察 (d)", "threshold_detection"),
                ("阈值-识别 (r)", "threshold_recognition"),
                ("离子碎片", "ion_fragments"),
                ("气味", "odor"),
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
                st.session_state.query["detected_samples"]
            ])
    ):
        st.info("未找到匹配的记录。")

# ===== 显示 CSV 合并结果 + 导出按钮 ===== v
if "csv_merged_df" in st.session_state:
    st.subheader("CSV 批量查询结果")
    df_merged = st.session_state.csv_merged_df
    st.dataframe(df_merged, use_container_widt任何人幕僚的是h=True)

    output = io.BytesIO()
    df_merged.to_csv(output, index=False, encoding='utf-8-sig')
    csv_data = output.getvalue()

    st.download_button(
        label="📥 下载合并后的 CSV 文件",
        data=csv_data,
        file_name=f"merged_{st.session_state.csv_filename}",
        mime="text/csv"
    )

    if st.button("清除 CSV 结果"):
        del st.session_state.csv_merged_df
        del st.session_state.csv_processed
        del st.session_state.csv_filename
        st.rerun()