import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from dataloader import load_and_preprocess_data, TimeSeriesDataset
from cnn import CNN1D

def evaluate_with_details(model, data_loader, device, file_paths, label_encoder):
    """
    PyTorch 모델 평가 및 각 클래스별 확률이 포함된 DataFrame 반환
    """
    import os
    import pandas as pd
    import torch.nn.functional as F

    model.eval()
    results = []

    class_names = label_encoder.classes_

    with torch.no_grad():
        sample_idx = 0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Softmax 확률
            probabilities = F.softmax(outputs, dim=1)

            # 예측 클래스
            predictions = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)

            for j in range(batch_size):

                if sample_idx >= len(file_paths):
                    break

                file_name = os.path.basename(file_paths[sample_idx])

                actual_label = label_encoder.inverse_transform(
                    [labels[j].cpu().item()]
                )[0]

                pred_label = label_encoder.inverse_transform(
                    [predictions[j].cpu().item()]
                )[0]

                row_data = {
                    "File_Name": file_name,
                    "Actual_Material": actual_label,
                    "Predicted_Material": pred_label,
                    "Result": "O"
                    if labels[j].item() == predictions[j].item()
                    else "X"
                }

                # 클래스별 확률 추가
                for col_idx, class_name in enumerate(class_names):
                    row_data[f"Prob_{class_name}"] = round(
                        probabilities[j, col_idx].cpu().item(),
                        4
                    )

                results.append(row_data)
                sample_idx += 1

    return pd.DataFrame(results)

def main():
    # 1. 설정
    TRAIN_DATA_DIR = "./fire_Debrs/KR EXP"
    K_FOLDS = 5 
    MAX_LENGTH = 2400
    BATCH_SIZE = 16    
    EPOCHS = 50        
    LEARNING_RATE = 0.001 
    normalization = "MaxAbs"
    log_transform = True
    SAVE_EXCEL_NAME = "./fire_Debrs/KR_EXP_LOG_MAX_LOG_5CV_2400_Results_final.xlsx"
    # 2. 데이터 로딩 (파일명 포함)
    print("데이터 로딩 및 전처리를 시작합니다...")
    X_all, y_all, label_encoder, all_paths = load_and_preprocess_data(
        TRAIN_DATA_DIR, MAX_LENGTH, bining=False,
        log_transform=log_transform, normalization=normalization, encoder=None
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label_encoder.classes_)
    
    fold_results = [] # 각 Fold의 상세 결과 저장용
    fold_final_accuracies = []

    # 3. K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    print(f"\n--- {K_FOLDS}-Fold 교차 검증 시작 ---")
    
    # 엑셀 저장을 위한 ExcelWriter 설정
    with pd.ExcelWriter(SAVE_EXCEL_NAME, engine='xlsxwriter') as writer:
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            print(f"\n>> Fold {fold+1}/{K_FOLDS}")
            
            # 데이터 분할
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            val_paths = [all_paths[i] for i in val_idx] # 해당 Fold의 테스트 파일명
            
            train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
            
            # 모델 및 최적화 설정
            model = CNN1D(num_classes=num_classes).to(device)                   
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Training
            for epoch in range(EPOCHS):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # 상세 평가 (맞은 것, 틀린 것 분류)
            result_df = evaluate_with_details(model, val_loader, device, val_paths, label_encoder)
            
            # Fold별 시트 저장 (Sheet 1~5)
            result_df.to_excel(writer,
                   sheet_name=f'Fold_{fold+1}',
                   index=False)

            acc = (result_df["Result"] == "O").mean() * 100
            
            fold_final_accuracies.append(acc)
            print(f"   Fold {fold+1} 완료 - Val Acc: {acc:.2f}%")

        # 6번째 시트: 전체 요약 결과 (Summary)
        summary_df = pd.DataFrame({
            'Fold': [f'Fold {i+1}' for i in range(K_FOLDS)],
            'Accuracy (%)': fold_final_accuracies
        })
        summary_df.loc[len(summary_df)] = ['Average', np.mean(fold_final_accuracies)]
        summary_df.loc[len(summary_df)] = ['Std Dev', np.std(fold_final_accuracies)]
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n✅ 분석 완료! 엑셀 파일이 저장되었습니다: {SAVE_EXCEL_NAME}")

    # 최종 결과 출력
    print("\n" + "="*40)
    print(f"📊 평균 최종 정확도: {np.mean(fold_final_accuracies):.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()