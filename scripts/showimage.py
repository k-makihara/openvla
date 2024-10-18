from PIL import Image, ImageOps
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def letterbox_pad(image, target_size, padding_fill_value=(123, 116, 103)):
    # 画像のサイズ
    width, height = image.size
    target_width, target_height = target_size
    
    # アスペクト比を計算
    scale = min(target_width / width, target_height / height)
    
    # 新しい画像サイズを計算
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 画像をリサイズ（アスペクト比を維持）
    resized_image = image.resize((new_width, new_height), Image.BICUBIC)
    
    # パディングの計算
    pad_width = (target_width - new_width) // 2
    pad_height = (target_height - new_height) // 2
    
    # Letterboxパディングを追加
    padded_image = ImageOps.expand(
        resized_image,
        border=(pad_width, pad_height, target_width - new_width - pad_width, target_height - new_height - pad_height),
        fill=padding_fill_value
    )
    
    return padded_image

def resize_image(image, size):
    # 指定されたサイズにリサイズ（補間はbicubic）
    return image.resize((size, size), Image.BICUBIC)

def center_crop(image, target_size):
    width, height = image.size
    target_width, target_height = target_size

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2
    
    return image.crop((left, top, right, bottom))

with open("/home/koshimakihara/Downloads/products/questionnaire_vqa_max1.json", "r") as f:
    examples = json.load(f)

image_dir = "/home/koshimakihara/Downloads/products"

for i in range(len(examples)):
    image_path_left = Path(examples[i]["image_paths"]["left_image"])
    image_path_right = Path(examples[i]["image_paths"]["right_image"])

    # 画像を開く
    left_image = Image.open(image_dir / image_path_left).convert("RGB")
    right_image = Image.open(image_dir / image_path_right).convert("RGB")

    # キャンバスのサイズを計算（横幅は画像の合計幅、高さは一番高い画像の高さ）
    total_width = left_image.width + right_image.width
    max_height = max(left_image.height, right_image.height)

    # 新しい画像（キャンバス）を作成
    concatenated_image = Image.new("RGB", (total_width, max_height))

    # 左画像を貼り付け
    concatenated_image.paste(left_image, (0, 0))

    # 右画像を左画像の横に貼り付け
    concatenated_image.paste(right_image, (left_image.width, 0))

    # 処理を順に適用
    # 1. LetterboxPad: 224x224のターゲットサイズ、指定されたパディングカラー
    letterbox_padded_image = letterbox_pad(concatenated_image, target_size=(224, 224), padding_fill_value=(123, 116, 103))

    # 2. Resize: 224x224サイズにリサイズ
    resized_image = resize_image(letterbox_padded_image, size=224)

    # 3. CenterCrop: 224x224の中心からトリミング
    center_cropped_image = center_crop(resized_image, target_size=(224, 224))

    # 結果を表示
    #center_cropped_image.show()
    plt.imshow(np.array(center_cropped_image))
    plt.show()

    # 結果を保存
    #center_cropped_image.save('output_image.jpg')
