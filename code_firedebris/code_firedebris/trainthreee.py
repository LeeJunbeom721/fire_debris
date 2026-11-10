import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
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
    # 1. 경로 및 설정
    TRAIN2_DATA_DIR = "./fire_Debrs/UCF SIM"
    TEST_DATA_DIR = "./fire_Debrs/KR EXP"
    TRAIN_DATA_DIR = "./fire_Debrs/UCF EXP" 
    EXCEL_FILE_PATH = './UCF_sim_exp_KR_exp+KR_exp_2400-final.xlsx'
    
    os.makedirs(os.path.dirname(EXCEL_FILE_PATH), exist_ok=True)
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE = 2400, 16, 50, 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 데이터 로딩 (고정 데이터 & 타겟 데이터)
    print("데이터 로딩 중...")
    X_fixed1, y_fixed1, label_encoder, _ = load_and_preprocess_data(TRAIN_DATA_DIR, MAX_LENGTH, False, False, "MaxAbs")
    X_fixed2, y_fixed2, _, _ = load_and_preprocess_data(TRAIN2_DATA_DIR, MAX_LENGTH, False, False, "MaxAbs", encoder=label_encoder)
    X_target_all, y_target_all, _, p_target_all = load_and_preprocess_data(TEST_DATA_DIR, MAX_LENGTH, False, False, "MaxAbs", encoder=label_encoder)

    # 고정 학습 데이터 병합
    X_fixed_base = np.concatenate([X_fixed1, X_fixed2], axis=0)
    y_fixed_base = np.concatenate([y_fixed1, y_fixed2], axis=0)

    # 3. 5회 반복 루프
    with pd.ExcelWriter(EXCEL_FILE_PATH, engine='xlsxwriter') as writer:
        summary_list = []

        for i in range(1, 6):
            print(f"\n🚀 Round {i}/5: 데이터 병합 및 재학습 시작")

            X_sub_train, X_sub_test, y_sub_train, y_sub_test, p_sub_train, p_sub_test = train_test_split(
                X_target_all,
                y_target_all,
                p_target_all,
                test_size=0.3,
                random_state=i*100,
                stratify=y_target_all
            )

            X_combined_train = np.concatenate([X_fixed_base, X_sub_train], axis=0)
            y_combined_train = np.concatenate([y_fixed_base, y_sub_train], axis=0)

            print(f"학습 데이터 크기 : {len(X_combined_train)}")

            model = CNN1D(num_classes=len(label_encoder.classes_)).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(
                TimeSeriesDataset(X_combined_train, y_combined_train),
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            for epoch in range(EPOCHS):
                model.train()

                for inputs, labels in train_loader:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    loss.backward()

                    optimizer.step()

            # -------------------------------
            # Test Loader 생성
            # -------------------------------
            test_loader = DataLoader(
                TimeSeriesDataset(X_sub_test, y_sub_test),
                batch_size=BATCH_SIZE,
                shuffle=False
            )

            # 상세 평가
            result_df = evaluate_with_details(
                model,
                test_loader,
                device,
                p_sub_test,
                label_encoder
            )

            result_df.to_excel(
                writer,
                sheet_name=f'Round_{i}',
                index=False
            )

            acc = (result_df["Result"] == "O").mean() * 100

            summary_list.append({
                "Round": i,
                "Accuracy (%)": acc
            })

            print(f"Round {i} Accuracy : {acc:.2f}%")

        # -------------------------------
        # Summary Sheet
        # -------------------------------
        summary_df = pd.DataFrame(summary_list)

        summary_df.loc[len(summary_df)] = [
            "Average",
            summary_df["Accuracy (%)"].mean()
        ]

        summary_df.loc[len(summary_df)] = [
            "Std",
            summary_df["Accuracy (%)"].std()
        ]

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

    print(f"\n저장 완료 : {EXCEL_FILE_PATH}")

    print("=" * 50)
    print(f"Average Accuracy : {summary_df['Accuracy (%)'][:-2].mean():.2f}%")
    print(f"Std : {summary_df['Accuracy (%)'][:-2].std():.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()