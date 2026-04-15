import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import matplotlib

# 【关键1】必须设置为纯后台画图模式，禁止弹出桌面窗口
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import os
from PIL import Image

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "STHeiti", "WenQuanYi Micro Hei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ====================================================================
# 原版核心类，完全保留
# ====================================================================
class AdvancedSEMAnalyzer:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.scale_pixels = None
        self.scale_real = None
        self.pore_samples = []
        self.polymer_samples = []
        self.selection_start = None
        self.selection_end = None
        self.porosity = None
        self.pore_sizes = None
        self.is_trained = False
        self.training_accuracy = None
        self.binary_image = None
        self.prediction = None
        self.threshold = None
        self.pore_gray_mean = None
        self.poly_gray_mean = None
        self.is_pore_darker = True

    def load_image_from_matrix(self, image_array):
        if image_array is None:
            raise ValueError("无法读取图像文件")
        if len(image_array.shape) == 3:
            self.original_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            self.original_image = image_array
        self.image = self.original_image.astype(np.float32)

    def set_scale(self, pixels, real_length):
        self.scale_pixels = pixels
        self.scale_real = real_length

    def train_threshold_classifier(self):
        if len(self.pore_samples) < 5 or len(self.polymer_samples) < 5:
            return False, "需要至少标注5个孔洞样本和5个聚合物样本"

        pore_vals = np.array([self.image[x, y] for x, y in self.pore_samples])
        poly_vals = np.array([self.image[x, y] for x, y in self.polymer_samples])

        self.pore_gray_mean = float(np.mean(pore_vals))
        self.poly_gray_mean = float(np.mean(poly_vals))
        self.threshold = (self.pore_gray_mean + self.poly_gray_mean) / 2.0

        if self.pore_gray_mean < self.poly_gray_mean:
            self.is_pore_darker = True
            pore_ok = int(np.sum(pore_vals < self.threshold))
            poly_ok = int(np.sum(poly_vals >= self.threshold))
        else:
            self.is_pore_darker = False
            pore_ok = int(np.sum(pore_vals >= self.threshold))
            poly_ok = int(np.sum(poly_vals < self.threshold))

        total = len(pore_vals) + len(poly_vals)
        acc = (pore_ok + poly_ok) / total * 100
        self.training_accuracy = acc
        self.is_trained = True
        return True, "训练完成"

    def calculate_porosity(self):
        if not self.is_trained:
            success, msg = self.train_threshold_classifier()
            if not success:
                return None, msg

        if self.selection_start and self.selection_end:
            r1 = min(self.selection_start[0], self.selection_end[0])
            r2 = max(self.selection_start[0], self.selection_end[0])
            c1 = min(self.selection_start[1], self.selection_end[1])
            c2 = max(self.selection_start[1], self.selection_end[1])
            analysis_img = self.image[r1:r2, c1:c2]
            self.analysis_region = (r1, c1, r2 - r1, c2 - c1)
        else:
            analysis_img = self.image
            self.analysis_region = (0, 0, self.image.shape[0], self.image.shape[1])

        if self.is_pore_darker:
            pred = (analysis_img >= self.threshold).astype(np.uint8)
        else:
            pred = (analysis_img < self.threshold).astype(np.uint8)

        self.prediction = pred
        self.binary_image = pred * 255

        h, w = analysis_img.shape[:2]
        pore_pixels = int(np.sum(pred == 0))
        porosity = pore_pixels / (h * w) * 100

        pore_sizes = []
        if self.scale_pixels and self.scale_real:
            contours, _ = cv2.findContours(
                (pred == 0).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5:
                    diam = 2 * np.sqrt(area / np.pi)
                    pore_sizes.append(diam * self.scale_real / self.scale_pixels)

        self.porosity = porosity
        self.pore_sizes = pore_sizes
        return porosity, "计算成功"

    def _build_report(self):
        fig = plt.figure(figsize=(22, 16))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.38, wspace=0.32)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.original_image, cmap="gray")
        ax1.set_title("原始图像")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.original_image, cmap="gray")
        for x, y in self.pore_samples:
            ax2.plot(y, x, "ro", markersize=4)
        for x, y in self.polymer_samples:
            ax2.plot(y, x, "bo", markersize=4)
        if self.selection_start and self.selection_end:
            ax2.add_patch(Rectangle(
                (self.selection_start[1], self.selection_start[0]),
                self.selection_end[1] - self.selection_start[1],
                self.selection_end[0] - self.selection_start[0],
                fill=False, edgecolor="lime", linewidth=2
            ))
        ax2.set_title("标注图像 (红=孔洞, 蓝=聚合物)")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(self.binary_image, cmap="gray")
        ax3.set_title("分类结果 (黑=孔洞, 白=聚合物)\n孔隙率 = {:.2f}%".format(self.porosity))
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[1, 0])
        if self.pore_sizes:
            ax4.hist(self.pore_sizes, bins=25, color="steelblue", edgecolor="black")
            mean_d = np.mean(self.pore_sizes)
            ax4.axvline(mean_d, color="red", ls="--", lw=1.5, label="均值 {:.2f} um".format(mean_d))
            ax4.legend()
        ax4.set_xlabel("孔径 (um)")
        ax4.set_ylabel("数量")
        ax4.set_title("孔径分布")
        ax4.grid(True, alpha=0.25)

        ax5 = fig.add_subplot(gs[1, 1])
        pore_g = [float(self.image[x, y]) for x, y in self.pore_samples]
        poly_g = [float(self.image[x, y]) for x, y in self.polymer_samples]
        ax5.hist(pore_g, bins=20, alpha=0.6, color="red", label="孔洞样本", edgecolor="black")
        ax5.hist(poly_g, bins=20, alpha=0.6, color="blue", label="聚合物样本", edgecolor="black")
        if self.threshold is not None:
            ax5.axvline(self.threshold, color="green", ls="--", lw=2, label="阈值 {:.1f}".format(self.threshold))
        ax5.set_xlabel("灰度值")
        ax5.legend()
        ax5.set_title("样本灰度分布")
        ax5.grid(True, alpha=0.25)

        ax6 = fig.add_subplot(gs[1, 2])
        bars = ax6.bar(["样本准确率"], [self.training_accuracy if self.training_accuracy else 0], color="steelblue",
                       width=0.4)
        ax6.text(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height() + 1,
                 "{:.1f}%".format(self.training_accuracy if self.training_accuracy else 0), ha="center", fontsize=12)
        ax6.set_ylabel("准确率 (%)")
        ax6.set_title("分类准确率")
        ax6.set_ylim(0, 115)

        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis("off")
        txt = "--- 分析摘要 ---\n\n"
        img_w = self.original_image.shape[1]
        img_h = self.original_image.shape[0]
        txt += "图像尺寸 : {} x {} px\n".format(img_w, img_h)
        if self.scale_real:
            res = self.scale_real / self.scale_pixels
            txt += "标尺 : {} um\n".format(self.scale_real)
            txt += "分辨率 : {:.4f} um/px\n\n".format(res)
        else:
            txt += "标尺 : 未设置\n\n"
        txt += ">>> 孔隙率 = {:.2f}%\n\n".format(self.porosity)
        if self.pore_sizes:
            txt += "检测到的孔 : {}\n".format(len(self.pore_sizes))
            txt += "平均孔径 : {:.2f} um\n".format(np.mean(self.pore_sizes))
            txt += "中位孔径 : {:.2f} um\n".format(np.median(self.pore_sizes))
            txt += "最小/最大 : {:.2f} / {:.2f} um\n".format(np.min(self.pore_sizes), np.max(self.pore_sizes))
        ax9.text(0.05, 0.95, txt, transform=ax9.transAxes, fontsize=10, va="top", fontfamily="SimHei",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        fig.suptitle("SEM 孔隙率分析 - 沈阳化工大学", fontsize=18, fontweight="bold", y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig


# ====================================================================
# UI 与交互逻辑
# ====================================================================
st.set_page_config(layout="wide", page_title="SEM 分析系统")

st.markdown(
    """
    <style>
    div[data-testid="stWidgetLabel"] p { font-size: 20px !important; font-weight: bold !important; color: #003366 !important; }
    div[role="radiogroup"] label p { font-size: 20px !important; }
    div[data-testid="stButton"] button p { font-size: 20px !important; font-weight: bold !important; }
    input { font-size: 20px !important; }
    </style>
    """, unsafe_allow_html=True
)

if "analyzer" not in st.session_state:
    st.session_state.analyzer = AdvancedSEMAnalyzer()
if "img_matrix" not in st.session_state:
    st.session_state.img_matrix = None

col_logo, col_title = st.columns([1, 15])
with col_logo:
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo", "syuct.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
with col_title:
    st.markdown("<div style='display: flex; align-items: center; height: 100%; padding-top: 10px;'><h2 style='color: #003366; margin: 0;'>沈阳化工大学 <span style='color: #333; font-size: 24px; font-weight: normal; margin-left: 15px;'>SEM 孔隙率智能分析系统</span></h2></div>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("选择 SEM 图像", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        try:
            pil_upload = Image.open(uploaded_file)
            
            # 【终极防弹算法】：应对各种位深的科研图像
            if pil_upload.mode.startswith('I') or pil_upload.mode == 'F':
                arr = np.array(pil_upload, dtype=np.float32)
                # 安全归一化
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                pil_upload = Image.fromarray(arr).convert("RGB")
            else:
                pil_upload = pil_upload.convert("RGB")

            img_bgr = cv2.cvtColor(np.array(pil_upload), cv2.COLOR_RGB2BGR)

            max_width = 800
            if img_bgr.shape[1] > max_width:
                ratio = max_width / img_bgr.shape[1]
                new_height = int(img_bgr.shape[0] * ratio)
                img_bgr = cv2.resize(img_bgr, (max_width, new_height))

            st.session_state.img_matrix = img_bgr
            st.session_state.analyzer = AdvancedSEMAnalyzer()
            st.session_state.analyzer.load_image_from_matrix(img_bgr)
            st.session_state.current_file = uploaded_file.name
        except Exception as e:
            st.error(f"读取图像失败: {e}")
            st.stop()

    col_img, col_controls = st.columns([2, 1])

    with col_controls:
        st.markdown("### 控制面板")
        mode = st.radio("选择鼠标操作:", ["设置标尺 (绿)", "框选区域 (黄)", "添加孔洞 (红)", "添加聚合物 (蓝)"])

        drawing_mode = "line"
        stroke_color = "green"
        if mode == "设置标尺 (绿)": drawing_mode, stroke_color = "line", "green"
        elif mode == "框选区域 (黄)": drawing_mode, stroke_color = "rect", "yellow"
        elif mode == "添加孔洞 (红)": drawing_mode, stroke_color = "point", "red"
        elif mode == "添加聚合物 (蓝)": drawing_mode, stroke_color = "point", "blue"

        scale_input = st.number_input("标尺实际长度 (um):", value=0)

        if st.button("计算孔隙率", type="primary", use_container_width=True):
            with st.spinner("正在计算..."):
                porosity, msg = st.session_state.analyzer.calculate_porosity()
                if porosity is not None:
                    st.success(f"计算完成！孔隙率: {porosity:.2f}%")
                    fig = st.session_state.analyzer._build_report()
                    st.pyplot(fig)
                else:
                    st.error(f"计算失败: {msg}")

        if st.button("清除所有标注", use_container_width=True):
            st.session_state.analyzer = AdvancedSEMAnalyzer()
            st.session_state.analyzer.load_image_from_matrix(st.session_state.img_matrix)
            st.rerun()

    with col_img:
        # 新增原生预览器作为双保险
        st.caption("📷 图像已成功加载：")
        
        img_h, img_w = st.session_state.img_matrix.shape[:2]
        # 强制转换为 RGB 确保画板绝对兼容
        pil_img_for_canvas = Image.fromarray(cv2.cvtColor(st.session_state.img_matrix, cv2.COLOR_BGR2RGB), 'RGB')

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color=stroke_color,
            background_image=pil_img_for_canvas,
            update_streamlit=True,
            height=int(img_h),
            width=int(img_w),
            drawing_mode=drawing_mode,
            key=f"sem_canvas_{st.session_state.current_file}",
        )

        if canvas_result.json_data is not None:
            pores, polys = [], []
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "circle":
                    r, c = int(obj["top"]), int(obj["left"])
                    if obj["stroke"] == "red": pores.append((r, c))
                    elif obj["stroke"] == "blue": polys.append((r, c))
                elif obj["type"] == "line" and obj["stroke"] == "green":
                    pix_len = np.sqrt((obj["x2"] - obj["x1"]) ** 2 + (obj["y2"] - obj["y1"]) ** 2)
                    st.session_state.analyzer.set_scale(pix_len, scale_input)
                elif obj["type"] == "rect" and obj["stroke"] == "yellow":
                    r1, c1 = int(obj["top"]), int(obj["left"])
                    r2, c2 = int(obj["top"] + obj["height"]), int(obj["left"] + obj["width"])
                    st.session_state.analyzer.selection_start = (r1, c1)
                    st.session_state.analyzer.selection_end = (r2, c2)

            st.session_state.analyzer.pore_samples = pores
            st.session_state.analyzer.polymer_samples = polys

        st.markdown(f"**孔洞样本**: {len(st.session_state.analyzer.pore_samples)} | **聚合物样本**: {len(st.session_state.analyzer.polymer_samples)}")

st.markdown("<br><br><br><div style='text-align: center; color: #888888; font-size: 14px; padding: 20px 0; border-top: 1px solid #eee;'>作者：材料学院马驰老师，版权所有，如需使用，请与作者联系。</div>", unsafe_allow_html=True)
