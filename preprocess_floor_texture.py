"""
地板材质预处理脚本
功能：
1. 透视畸变校正
2. 无缝纹理处理（边缘融合）
3. 光照均匀化
4. 分辨率与尺寸标准化
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image


class FloorTextureProcessor:
    """地板纹理处理器"""

    def __init__(
        self,
        target_size: int = 512,
        blend_width: int = 64,
        blur_size: int = 101,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8
    ):
        """
        初始化处理器

        Args:
            target_size: 目标尺寸（2的幂次方）
            blend_width: 边缘融合宽度
            blur_size: 光照均匀化的高斯模糊核大小
            clahe_clip: CLAHE 对比度限制
            clahe_grid: CLAHE 网格大小
        """
        self.target_size = target_size
        self.blend_width = blend_width
        self.blur_size = blur_size
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

    # ==================== 1. 透视校正 ====================

    def detect_corners_interactive(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        交互式选择四个角点（用于手动校正）
        返回四个角点坐标，顺序：左上、右上、右下、左下
        """
        corners = []
        img_display = img.copy()
        window_name = "Select 4 corners (TL, TR, BR, BL) - Press 'r' to reset, 'q' to quit"

        def mouse_callback(event, x, y, flags, param):
            nonlocal corners, img_display
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])
                cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
                if len(corners) > 1:
                    cv2.line(img_display, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
                if len(corners) == 4:
                    cv2.line(img_display, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
                cv2.imshow(window_name, img_display)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, img_display)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # 重置
                corners = []
                img_display = img.copy()
                cv2.imshow(window_name, img_display)
            elif key == ord('q') or key == 27:  # 退出
                cv2.destroyAllWindows()
                return None
            elif key == 13 and len(corners) == 4:  # Enter确认
                cv2.destroyAllWindows()
                return np.array(corners, dtype=np.float32)

        return None

    def detect_corners_auto(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        自动检测矩形角点（适用于有明显边界的地板照片）
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 膨胀边缘以连接断开的线
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 找最大轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 多边形近似
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            # 排序角点：左上、右上、右下、左下
            corners = self._order_corners(corners)
            return corners

        return None

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """将角点按左上、右上、右下、左下顺序排列"""
        # 按y坐标排序，分成上下两组
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]

        # 上面两点按x排序
        top_left, top_right = top_points[np.argsort(top_points[:, 0])]
        # 下面两点按x排序
        bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def correct_perspective(
        self,
        img: np.ndarray,
        corners: Optional[np.ndarray] = None,
        auto_detect: bool = True
    ) -> np.ndarray:
        """
        透视校正

        Args:
            img: 输入图像
            corners: 四个角点坐标，如果为None则自动检测或使用整张图
            auto_detect: 是否尝试自动检测角点

        Returns:
            校正后的图像
        """
        h, w = img.shape[:2]

        if corners is None and auto_detect:
            corners = self.detect_corners_auto(img)

        if corners is None:
            # 无法检测到角点，使用整张图片
            return img

        # 计算目标矩形的宽高
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])

        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_left, height_right))

        # 目标角点
        dst_corners = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(corners, dst_corners)

        # 应用透视变换
        corrected = cv2.warpPerspective(img, matrix, (max_width, max_height))

        return corrected

    # ==================== 2. 无缝纹理处理 ====================

    def make_seamless_blend(self, img: np.ndarray) -> np.ndarray:
        """
        使用渐变混合方法创建无缝纹理
        将图像四边与对边进行渐变融合
        """
        h, w = img.shape[:2]
        blend_w = min(self.blend_width, w // 4, h // 4)

        result = img.copy().astype(np.float32)

        # 创建水平方向的渐变权重
        gradient_h = np.linspace(0, 1, blend_w).reshape(1, -1, 1)

        # 左右边缘融合
        left_strip = result[:, :blend_w]
        right_strip = result[:, -blend_w:]
        blended_lr = left_strip * (1 - gradient_h) + right_strip * gradient_h
        result[:, :blend_w] = blended_lr
        result[:, -blend_w:] = blended_lr

        # 创建垂直方向的渐变权重
        gradient_v = np.linspace(0, 1, blend_w).reshape(-1, 1, 1)

        # 上下边缘融合
        top_strip = result[:blend_w, :]
        bottom_strip = result[-blend_w:, :]
        blended_tb = top_strip * (1 - gradient_v) + bottom_strip * gradient_v
        result[:blend_w, :] = blended_tb
        result[-blend_w:, :] = blended_tb

        return np.clip(result, 0, 255).astype(np.uint8)

    def make_seamless_mirror(self, img: np.ndarray) -> np.ndarray:
        """
        使用镜像翻转方法创建无缝纹理
        将图像扩展为2x2并镜像，然后裁剪中心
        """
        h, w = img.shape[:2]

        # 水平翻转
        img_h_flip = cv2.flip(img, 1)
        # 垂直翻转
        img_v_flip = cv2.flip(img, 0)
        # 双向翻转
        img_hv_flip = cv2.flip(img, -1)

        # 拼接成2x2
        top_row = np.hstack([img, img_h_flip])
        bottom_row = np.hstack([img_v_flip, img_hv_flip])
        expanded = np.vstack([top_row, bottom_row])

        # 裁剪中心区域
        center_x, center_y = w, h
        result = expanded[center_y - h//2:center_y + h//2 + h%2,
                         center_x - w//2:center_x + w//2 + w%2]

        return result

    def make_seamless_fft(self, img: np.ndarray) -> np.ndarray:
        """
        使用频域处理创建无缝纹理
        通过FFT去除边缘不连续性
        """
        result = np.zeros_like(img, dtype=np.float32)

        for c in range(3):
            channel = img[:, :, c].astype(np.float32)

            # FFT变换
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)

            h, w = channel.shape
            cy, cx = h // 2, w // 2

            # 创建低通滤波器，保留低频成分
            mask = np.zeros((h, w), np.float32)
            r = min(h, w) // 4
            cv2.circle(mask, (cx, cy), r, 1, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)

            # 应用滤波
            fshift_filtered = fshift * (1 - mask * 0.3)

            # 逆变换
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            result[:, :, c] = np.abs(img_back)

        return np.clip(result, 0, 255).astype(np.uint8)

    # ==================== 3. 光照均匀化 ====================

    def normalize_lighting(self, img: np.ndarray) -> np.ndarray:
        """高斯除法去除低频光照渐变"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        blurred = cv2.GaussianBlur(l_channel, (self.blur_size, self.blur_size), 0)
        mean_val = np.mean(l_channel)
        normalized = (l_channel / (blurred + 1e-6)) * mean_val
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        lab[:, :, 0] = normalized
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """CLAHE 增强局部对比度"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def auto_white_balance(self, img: np.ndarray) -> np.ndarray:
        """自动白平衡（灰度世界假设）"""
        result = img.astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3

        result[:, :, 0] *= avg_gray / (avg_b + 1e-6)
        result[:, :, 1] *= avg_gray / (avg_g + 1e-6)
        result[:, :, 2] *= avg_gray / (avg_r + 1e-6)

        return np.clip(result, 0, 255).astype(np.uint8)

    def adjust_saturation(self, img: np.ndarray, factor: float = 1.1) -> np.ndarray:
        """调整饱和度"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def fix_lighting(self, img: np.ndarray, white_balance: bool = True) -> np.ndarray:
        """完整光照处理流程"""
        result = self.normalize_lighting(img)
        result = self.apply_clahe(result)
        if white_balance:
            result = self.auto_white_balance(result)
        return result

    # ==================== 4. 尺寸标准化 ====================

    def crop_to_square(self, img: np.ndarray) -> np.ndarray:
        """裁剪为正方形（取中心区域）"""
        h, w = img.shape[:2]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[top:top + min_dim, left:left + min_dim]

    def resize_to_power_of_two(self, img: np.ndarray, size: Optional[int] = None) -> np.ndarray:
        """调整为2的幂次方尺寸"""
        if size is None:
            size = self.target_size
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)

    def standardize_size(self, img: np.ndarray) -> np.ndarray:
        """标准化尺寸：裁剪为正方形 + 调整为目标尺寸"""
        img = self.crop_to_square(img)
        img = self.resize_to_power_of_two(img)
        return img

    # ==================== 5. 完整处理流程 ====================

    def process(
        self,
        img: np.ndarray,
        perspective_corners: Optional[np.ndarray] = None,
        auto_perspective: bool = False,
        seamless_method: str = "blend",
        fix_light: bool = True,
        white_balance: bool = True
    ) -> np.ndarray:
        """
        完整处理流程

        Args:
            img: 输入图像
            perspective_corners: 透视校正角点
            auto_perspective: 是否自动检测透视
            seamless_method: 无缝处理方法 (blend/mirror/fft/none)
            fix_light: 是否进行光照均匀化
            white_balance: 是否进行白平衡

        Returns:
            处理后的图像
        """
        result = img.copy()

        # 1. 透视校正
        if perspective_corners is not None or auto_perspective:
            result = self.correct_perspective(result, perspective_corners, auto_perspective)

        # 2. 尺寸标准化
        result = self.standardize_size(result)

        # 3. 光照均匀化
        if fix_light:
            result = self.fix_lighting(result, white_balance)

        # 4. 无缝处理
        if seamless_method == "blend":
            result = self.make_seamless_blend(result)
        elif seamless_method == "mirror":
            result = self.make_seamless_mirror(result)
        elif seamless_method == "fft":
            result = self.make_seamless_fft(result)

        return result


def process_directory(
    input_dir: Path,
    output_dir: Path,
    processor: FloorTextureProcessor,
    seamless_method: str = "blend",
    auto_perspective: bool = False,
    fix_light: bool = True,
    white_balance: bool = True,
    prefix: str = "floor"
):
    """批量处理目录中的图片"""
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        print(f"目录中没有找到图片: {input_dir}")
        return

    print(f"找到 {len(image_files)} 张图片")
    print(f"输出目录: {output_dir}")
    print(f"目标尺寸: {processor.target_size}x{processor.target_size}")
    print(f"无缝方法: {seamless_method}")
    print("-" * 50)

    for idx, img_path in enumerate(sorted(image_files), start=1):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[错误] 无法读取: {img_path.name}")
                continue

            original_size = img.shape[:2][::-1]

            # 处理图片
            result = processor.process(
                img,
                auto_perspective=auto_perspective,
                seamless_method=seamless_method,
                fix_light=fix_light,
                white_balance=white_balance
            )

            # 保存
            new_name = f"{prefix}_{idx:02d}.jpg"
            output_path = output_dir / new_name
            cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

            print(f"[{idx:02d}] {img_path.name} {original_size} -> "
                  f"{processor.target_size}x{processor.target_size}")

        except Exception as e:
            print(f"[错误] {img_path.name}: {e}")

    print("-" * 50)
    print(f"完成！共处理 {len(image_files)} 张纹理")


def create_tiled_preview(img: np.ndarray, tiles: int = 3) -> np.ndarray:
    """创建平铺预览图，用于检查无缝效果"""
    h, w = img.shape[:2]
    preview = np.zeros((h * tiles, w * tiles, 3), dtype=np.uint8)
    for i in range(tiles):
        for j in range(tiles):
            preview[i*h:(i+1)*h, j*w:(j+1)*w] = img
    return preview


def main():
    parser = argparse.ArgumentParser(
        description="地板材质预处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单张图片
  python preprocess_floor_texture.py -i floor.jpg -o output.jpg

  # 批量处理目录
  python preprocess_floor_texture.py -i ./raw_textures -o ./processed

  # 使用镜像方法创建无缝纹理
  python preprocess_floor_texture.py -i ./raw -o ./out --seamless mirror

  # 交互式透视校正
  python preprocess_floor_texture.py -i floor.jpg -o out.jpg --interactive
        """
    )

    parser.add_argument("-i", "--input", required=True,
                        help="输入图片或目录路径")
    parser.add_argument("-o", "--output", required=True,
                        help="输出图片或目录路径")
    parser.add_argument("--size", type=int, default=512,
                        choices=[256, 512, 1024, 2048],
                        help="目标尺寸 (默认: 512)")
    parser.add_argument("--seamless", default="blend",
                        choices=["blend", "mirror", "fft", "none"],
                        help="无缝处理方法 (默认: blend)")
    parser.add_argument("--no-lighting", action="store_true",
                        help="跳过光照均匀化")
    parser.add_argument("--no-white-balance", action="store_true",
                        help="跳过白平衡")
    parser.add_argument("--auto-perspective", action="store_true",
                        help="自动检测透视并校正")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式选择透视校正角点")
    parser.add_argument("--blend-width", type=int, default=64,
                        help="边缘融合宽度 (默认: 64)")
    parser.add_argument("--prefix", default="floor",
                        help="输出文件名前缀 (默认: floor)")
    parser.add_argument("--preview", action="store_true",
                        help="生成平铺预览图")

    args = parser.parse_args()

    # 创建处理器
    processor = FloorTextureProcessor(
        target_size=args.size,
        blend_width=args.blend_width
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        # 批量处理目录
        process_directory(
            input_path,
            output_path,
            processor,
            seamless_method=args.seamless,
            auto_perspective=args.auto_perspective,
            fix_light=not args.no_lighting,
            white_balance=not args.no_white_balance,
            prefix=args.prefix
        )
    else:
        # 处理单张图片
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"错误: 无法读取图片 {input_path}")
            sys.exit(1)

        corners = None
        if args.interactive:
            corners = processor.detect_corners_interactive(img)
            if corners is None:
                print("取消透视校正")

        result = processor.process(
            img,
            perspective_corners=corners,
            auto_perspective=args.auto_perspective,
            seamless_method=args.seamless,
            fix_light=not args.no_lighting,
            white_balance=not args.no_white_balance
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"已保存: {output_path}")

        if args.preview:
            preview = create_tiled_preview(result)
            preview_path = output_path.with_stem(output_path.stem + "_preview")
            cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"预览图: {preview_path}")


if __name__ == "__main__":
    main()
