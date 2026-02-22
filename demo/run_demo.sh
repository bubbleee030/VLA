#!/bin/bash
# 一鍵執行 Demo 腳本

echo "======================================"
echo "  VLA Demo 啟動器"
echo "======================================"
echo ""
echo "請選擇你要執行的功能："
echo ""
echo "1) 視覺化資料集（最簡單，不需訓練）"
echo "2) 互動式指令測試"
echo "3) 快速訓練模型"
echo "4) 視覺化模型推論結果"
echo "5) 檢查資料集狀態"
echo "q) 退出"
echo ""
echo "======================================"
read -p "請輸入選項 [1-5, q]: " choice

case $choice in
    1)
        echo ""
        echo "🎨 啟動資料集視覺化..."
        echo ""
        read -p "要視覺化幾個 episodes？ [預設 5]: " num_episodes
        num_episodes=${num_episodes:-5}
        
        read -p "是否跳過影片生成（節省時間）？ [y/N]: " skip_videos
        
        if [[ $skip_videos == "y" || $skip_videos == "Y" ]]; then
            python3 simple_visualize_data.py --num_episodes $num_episodes --skip_videos
        else
            python3 simple_visualize_data.py --num_episodes $num_episodes
        fi
        
        echo ""
        echo "✓ 完成！結果儲存在 ./data_visualization/"
        ;;
        
    2)
        echo ""
        echo "🤖 啟動互動式 demo..."
        echo ""
        python3 interactive_demo.py
        ;;
        
    3)
        echo ""
        echo "🎓 開始快速訓練..."
        echo "這會訓練 1000 steps，約需 10-30 分鐘"
        echo ""
        read -p "確定要開始訓練嗎？ [y/N]: " confirm
        
        if [[ $confirm == "y" || $confirm == "Y" ]]; then
            bash quick_train_demo.sh
        else
            echo "已取消"
        fi
        ;;
        
    4)
        echo ""
        echo "📊 視覺化模型推論..."
        echo ""
        
        # 列出可用的 checkpoints
        echo "可用的 checkpoints:"
        echo ""
        find ../outputs ../checkpoints -name "*.ckpt" -type f 2>/dev/null | head -10
        echo ""
        
        read -p "請輸入 checkpoint 路徑: " ckpt_path
        
        if [ -z "$ckpt_path" ]; then
            echo "❌ 未提供 checkpoint 路徑"
            exit 1
        fi
        
        if [ ! -f "$ckpt_path" ]; then
            echo "❌ 找不到 checkpoint: $ckpt_path"
            exit 1
        fi
        
        read -p "要視覺化幾個 episodes？ [預設 5]: " num_episodes
        num_episodes=${num_episodes:-5}
        
        python3 visualize_inference.py \
            --checkpoint "$ckpt_path" \
            --dataset_path ../data/datasets/mango \
            --num_episodes $num_episodes
        
        echo ""
        echo "✓ 完成！結果儲存在 ./demo_outputs/"
        ;;
        
    5)
        echo ""
        echo "📁 資料集狀態檢查..."
        echo ""
        
        dataset_path="../data/datasets/mango"
        
        if [ -d "$dataset_path" ]; then
            echo "✓ 資料集路徑存在: $dataset_path"
            echo ""
            
            episode_count=$(find "$dataset_path" -maxdepth 1 -type d -name "episode_*" | wc -l)
            echo "📊 Episodes 總數: $episode_count"
            echo ""
            
            echo "前 10 個 episodes:"
            ls -1 "$dataset_path" | grep "episode_" | head -10
            echo ""
            
            echo "範例 episode (episode_0) 內容:"
            ls -lh "$dataset_path/episode_0/" 2>/dev/null || echo "  episode_0 不存在"
            echo ""
            
            if [ -f "$dataset_path/episode_0/instruction.txt" ]; then
                echo "範例指令:"
                cat "$dataset_path/episode_0/instruction.txt"
            fi
        else
            echo "❌ 找不到資料集: $dataset_path"
            echo ""
            echo "請確認資料集位置是否正確"
        fi
        ;;
        
    q|Q)
        echo ""
        echo "👋 再見！"
        exit 0
        ;;
        
    *)
        echo ""
        echo "❌ 無效的選項: $choice"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "  執行完畢"
echo "======================================"
