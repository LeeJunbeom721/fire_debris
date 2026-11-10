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
import glob
from tqdm import tqdm
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict_with_details(model, X_test, device, file_paths, label_encoder, batch_size=16):
    model.eval()
    class_names = label_encoder.classes_
    results = []

    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # [N, 1, L]

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            for j in range(batch.size(0)):
                idx = i + j
                row = {
                    "File_Name": os.path.basename(file_paths[idx]),
                    "Predicted_Material": label_encoder.inverse_transform([preds[j].cpu().item()])[0],
                }
                for c, name in enumerate(class_names):
                    row[f"Prob_{name}"] = round(probs[j, c].cpu().item(), 4)
                results.append(row)

    return pd.DataFrame(results)

def load_test_data_no_label(test_dir, max_length):
    # 하위 폴더 구조와 무관하게 csv/xlsx 전부 수집
    file_paths = sorted(
        glob.glob(os.path.join(test_dir, '**', '*.csv'), recursive=True) +
        glob.glob(os.path.join(test_dir, '**', '*.xlsx'), recursive=True)
    )

    X = np.zeros((len(file_paths), max_length), dtype=np.float32)

    for i, f_path in enumerate(tqdm(file_paths, desc="Loading mixture test data")):
        if f_path.endswith('.csv'):
            df = pd.read_csv(f_path, header=None)
        else:
            df = pd.read_excel(f_path, header=None)

        data = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0).values

        # zero-padding / truncation (dataloader와 동일)
        temp = np.zeros(max_length, dtype=np.float32)
        length = min(len(data), max_length)
        temp[:length] = data[:length]
        data = temp

        # MaxAbs 정규화 (dataloader와 동일, 샘플 단위)
        s_max = np.max(np.abs(data))
        if s_max > 0:
            data = data / s_max

        X[i, :] = data

    return X, file_paths

def main():
        
    DATA_DIR_1 = "./fire_Debrs/KR EXP" 
    DATA_DIR_2 = "./fire_Debrs/UCF SIM"
    DATA_DIR_3 = "./fire_Debrs/UCF EXP"
    TEST_DIR = "./fire_Debrs/추론용 데이터(확률추론)"
    OUTPUT_DIR = "./mixture_test-seed113"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE = 2400, 16, 50, 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 모든 개별 데이터셋 미리 로드 (DATA_DIR_2, DATA_DIR_3 추가)
    print("기반 데이터셋 로딩 중...")
    # 기준이 될 label_encoder 학습 및 1번 데이터 로드
    X_1, y_1, label_encoder, _ = load_and_preprocess_data(DATA_DIR_1, MAX_LENGTH, False, False, "MaxAbs")
    
    #일관된 레이블 인코딩을 위해 기준 label_encoder를 주입하여 2, 3번 데이터 로드
    X_2, y_2, _, _ = load_and_preprocess_data(DATA_DIR_2, MAX_LENGTH, False, False, "MaxAbs", encoder=label_encoder)
    X_3, y_3, _, _ = load_and_preprocess_data(DATA_DIR_3, MAX_LENGTH, False, False, "MaxAbs", encoder=label_encoder)
    
    # 3. 고정 평가(Test) 데이터 로드 (5-Round 코드처럼 경로 리스트 p_test 확보)
    print("평가용 Mixture 데이터셋 로딩 중...")
    X_test, p_test = load_test_data_no_label(TEST_DIR, MAX_LENGTH)
    print(f"평가용 Mixture 파일 {len(p_test)}개 로드 완료")
    print(f"인코더 클래스: {label_encoder.classes_}")
   


    # 고정할 테스트 데이터 이름 키워드
    test_name_suffix = "Mixture"

    # 💡 7개 시나리오 정의 및 실제 로드한 변수 매핑
    # 매핑 구조: "출력 파일명용": [실제 데이터 배열 리스트, 실제 레이블 배열 리스트]
    scenarios = {
        "KR_EXP": ([X_1], [y_1]),
        "UCF_SIM": ([X_2], [y_2]),
        "UCF_EXP": ([X_3], [y_3]),
        "KR_EXP+UCF SIM": ([X_1, X_2], [y_1, y_2]),
        "KR_EXP+UCF_EXP": ([X_1, X_3], [y_1, y_3]),
        "UCF_SIM+UCF_EXP": ([X_2, X_3], [y_2, y_3]),
        "KR_EXP+UCF_EXP+UCF_SIM": ([X_1, X_2, X_3], [y_1, y_2, y_3])
    }
    
    # 4. 시나리오별 반복 루프 시작
    for scenario_name, (x_list, y_list) in scenarios.items():
        print(f"\n🚀 시나리오 시작: {scenario_name}")
        
        excel_filename = f"{scenario_name} + {test_name_suffix}.xlsx"
        excel_file_path = os.path.join(OUTPUT_DIR, excel_filename)
        
        # 💡 [핵심 구현] 시나리오 조합에 맞게 리스트 안에 든 numpy 배열들을 하나로 합쳐줍니다.
        X_combined_train = np.concatenate(x_list, axis=0)
        y_combined_train = np.concatenate(y_list, axis=0)
        
        print(f"   > 학습 데이터 크기: {len(X_combined_train)}개 조합 완료")
        
        train_dataset = TimeSeriesDataset(X_combined_train, y_combined_train)
        set_seed(113)                                          # ← 모델 전에 먼저
        g = torch.Generator(); g.manual_seed(113) 
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
        
        # 모델 및 학습 요소 초기화
        set_seed(113)
        model = CNN1D(num_classes=len(label_encoder.classes_)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # 학습 루프 (Train Loop)
        print(f"   > '{scenario_name}' 모델 학습 진행 중...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"      Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}")
        
        # 상세 평가 및 개별 엑셀 파일 저장
        print(f"   > 평가 및 개별 엑셀 파일 저장 중...")
        df_result = predict_with_details(model, X_test, device, p_test, label_encoder)
        # 엑셀 파일로 개별 저장
        df_result.to_excel(excel_file_path, index=False)
        print(f"   💾 저장 완료: {excel_file_path}")
            
    print(f"\n📊 요청하신 7가지 조합의 실험 및 개별 엑셀 파일 저장이 모두 완료되었습니다!")

if __name__ == "__main__":
    main()