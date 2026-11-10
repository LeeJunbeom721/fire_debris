import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from dataloader import load_and_preprocess_data, TimeSeriesDataset
from cnn import CNN1D
import torch.nn.functional as F

def evaluate_with_details(model, data_loader, device, file_paths, label_encoder):
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
    # 1. 경로 및 설정
    TRAIN_DATA_DIR = "./fire_Debrs/KR EXP" 
    TEST_DATA_DIR = "./fire_Debrs/UCF EXP"
    EXCEL_FILE_PATH = './fire_Debrs/KR_EXP_UCF_EXP+UCF_EXP_2400_5RS.xlsx'
    
    os.makedirs(os.path.dirname(EXCEL_FILE_PATH), exist_ok=True)
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE = 2400, 16, 50, 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 데이터 로딩 (고정 데이터 & 타겟 데이터)
    print("데이터 로딩 중...")
    X_fixed_base, y_fixed_base, label_encoder, _ = load_and_preprocess_data(TRAIN_DATA_DIR, MAX_LENGTH, False, False, "MaxAbs")
    X_target_all, y_target_all, _, p_target_all = load_and_preprocess_data(TEST_DATA_DIR, MAX_LENGTH, False, False, "MaxAbs", encoder=label_encoder)
    # 3. 5회 반복 루프
    with pd.ExcelWriter(EXCEL_FILE_PATH, engine='xlsxwriter') as writer:
        summary_list = []

        for i in range(1, 6):
            print(f"\n🚀 Round {i}/5: 데이터 병합 및 재학습 시작")
            
            # Korea Internal Data를 7:3으로 분할
            X_sub_train, X_sub_test, y_sub_train, y_sub_test, _, p_sub_test = train_test_split(
                X_target_all, y_target_all, p_target_all, test_size=0.3, random_state=i*100, stratify=y_target_all
            )

            # [핵심] 고정 데이터 + 이번 라운드에 뽑힌 70%의 타겟 데이터를 합침
            X_combined_train = np.concatenate([X_fixed_base, X_sub_train], axis=0)
            y_combined_train = np.concatenate([y_fixed_base, y_sub_train], axis=0)

            print(f"학습 데이터 크기: {len(X_combined_train)} (고정: {len(X_fixed_base)} + 추가: {len(X_sub_train)})")

            # 모델 초기화 및 학습
            model = CNN1D(num_classes=len(label_encoder.classes_)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            
            train_loader = DataLoader(TimeSeriesDataset(X_combined_train, y_combined_train), batch_size=BATCH_SIZE, shuffle=True)
            
            for epoch in range(EPOCHS):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

           # 테스트 (나머지 30% 데이터 사용)
            test_loader = DataLoader(
                TimeSeriesDataset(X_sub_test, y_sub_test),
                batch_size=BATCH_SIZE,
                shuffle=False
            )

            df_details = evaluate_with_details(
                model,
                test_loader,
                device,
                p_sub_test,
                label_encoder
            )

            # Round별 결과 저장
            df_details.to_excel(
                writer,
                sheet_name=f'Test_Round_{i}',
                index=False
            )

            # Accuracy 계산
            acc = (df_details["Result"] == "O").mean()

            summary_list.append({
                "Round": f"Round_{i}",
                "Accuracy": acc
            })

            print(f"Round {i} Accuracy: {acc:.4f}")

        # Summary 시트 저장
        summary_df = pd.DataFrame(summary_list)

        summary_df.loc[len(summary_df)] = [
            "Average",
            summary_df["Accuracy"].mean()
        ]

        summary_df.loc[len(summary_df)] = [
            "Std Dev",
            summary_df["Accuracy"].std()
        ]

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

    print("\n" + "=" * 40)
    print(f"평균 Accuracy : {summary_df['Accuracy'][:-2].mean():.4f}")
    print(f"표준편차 : {summary_df['Accuracy'][:-2].std():.4f}")
    print("=" * 40)

    print(f"\n✅ 모든 과정 완료! 결과 저장: {EXCEL_FILE_PATH}")

if __name__ == "__main__":
    main()