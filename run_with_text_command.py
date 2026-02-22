#!/usr/bin/env python3
"""
文字指令系統展示腳本
- 將使用者輸入的文字指令轉換為嵌入向量
- 模擬將此向量作為模型輸入的過程
"""

from sentence_transformers import SentenceTransformer
import torch
import numpy as np

def get_text_embedding(instruction_text, model_name="clip-ViT-L-14"):
    """
    將文字指令轉換為嵌入向量
    """
    print(f"🧠 正在使用 '{model_name}' 模型來理解您的指令...")
    
    # 載入預訓練的 CLIP-style 模型
    model = SentenceTransformer(model_name)
    
    # 將文字編碼為嵌入向量
    embedding = model.encode(instruction_text, convert_to_tensor=True)
    
    print(f"✅ 指令 '{instruction_text}' 已轉換為嵌入向量")
    print(f"   向量維度：{embedding.shape}") # 應該是 [768]
    
    return embedding

def main():
    print("="*60)
    print("🗣️ 歡迎使用 VLA 文字指令系統")
    print("="*60)
    print("您可以輸入任何操作指令，例如：")
    print("  - 英文: 'pick up the mango and place it in the bowl'")
    print("  - 中文: '請把芒果拿起來，然後放到碗裡'")
    print("\n輸入 'exit' 或 'q' 來離開程式。\n")

    while True:
        # 讓使用者輸入指令
        user_command = input("👉 請輸入您的指令：")
        
        if user_command.lower() in ['exit', 'q']:
            print("👋 程式結束。")
            break
        
        if not user_command:
            continue
        
        # 1. 取得指令的嵌入向量
        instruction_embedding = get_text_embedding(user_command)
        
        # 2. 模擬將其作為模型輸入
        # 在真實的 VLA 中，這個向量會與影像特徵、機器人狀態特徵一起被融合
        print("\n--- 模擬模型推斷流程 ---")
        print("1. ✅ 文字指令已編碼")
        print("2. 👁️ (模擬) 取得攝影機影像，並用 ViT 編碼")
        print("3. 🤚 (模擬) 取得 GelSight 影像，並用 CNN 編碼")
        print("4. 🦾 (模擬) 取得機器人關節狀態，並用 MLP 編碼")
        
        # 5. 將所有特徵融合
        # fusion_input = torch.cat([instruction_embedding, vision_features, tactile_features, ...])
        print("\n5. ⚙️ (模擬) 將文字向量與其他感測器特徵融合...")
        print(f"   融合後的特徵將會被輸入到決策網路 (Transformer)。")
        
        # 6. 儲存嵌入向量以供真實模型使用
        save_path = "./instruction_embedding_custom.npy"
        np.save(save_path, instruction_embedding.cpu().numpy())
        print(f"\n💾 您的指令嵌入向量已儲存到：{save_path}")
        print("   您可以將這個檔案載入到真實的推斷腳本中！")
        print("----------------------------\n")
        

if __name__ == "__main__":
    main()