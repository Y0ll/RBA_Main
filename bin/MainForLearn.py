import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.pipeline import Pipeline
import ipaddress
from lightgbm import LGBMClassifier
import time
import warnings
import joblib
from flask import Flask, request, jsonify
import io

app = Flask(__name__)
warnings.filterwarnings('ignore')

User_ID = ''

classifiers = {
            'lgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=12,
                learning_rate=0.05,
                num_leaves=63,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        }
def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return 0  


def enhanced_preprocessing(train_df, test_df=None):
    if test_df is None:
        train_df = train_df.copy()
        known_countries = set(train_df['Country'].unique())
        train_df['Is_Unknown_Country'] = 0
        return train_df, known_countries
    else:
        train_df = train_df.copy()
        test_df = test_df.copy()
        known_countries = set(train_df['Country'].unique())

        train_df['Is_Unknown_Country'] = 0
        test_df['Is_Unknown_Country'] = test_df['Country'].apply(
            lambda x: 1 if x not in known_countries else 0
        )

        most_common_country = train_df['Country'].mode()[0]
        test_df['Country'] = test_df['Country'].apply(
            lambda x: x if x in known_countries else most_common_country
        )

        return train_df, test_df, known_countries


def calculate_rba_score(auc, prediction_time_ms, auc_weight=0.7, speed_weight=0.3):
    max_acceptable_time = 100 
    speed_score = max(0, 1 - (prediction_time_ms / max_acceptable_time))
    combined_score = (auc * auc_weight) + (speed_score * speed_weight)
    return combined_score, speed_score


def load_and_preprocess_data_ml(data):
    print("Загрузка данных для ML моделей...")

    data['Login_Hour'] = pd.to_datetime(data['Login_Timestamp']).dt.hour
    data['Is_Account_Takeover'] = data['Is_Account_Takeover'].astype(np.uint8)
    data['Is_Attack_IP'] = data['Is_Attack_IP'].astype(np.uint8)
    data['Login_Successful'] = data['Login_Successful'].astype(np.uint8)
    data = data.drop(columns=["Round_Trip_Time", 'Region', 'City', 'Login_Timestamp', 'index'])

    data['User_Agent_String'], _ = pd.factorize(data['User_Agent_String'])
    data['Browser_Name_and_Version'], _ = pd.factorize(data['Browser_Name_and_Version'])
    data['OS_Name_and_Version'], _ = pd.factorize(data['OS_Name_and_Version'])
    data['IP_Address'] = data['IP_Address'].apply(ip_to_int)

    features = data.drop(['Is_Attack_IP', 'Is_Account_Takeover'], axis=1)
    labels = data['Is_Account_Takeover']

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    X_train_processed, X_test_processed, known_countries = enhanced_preprocessing(X_train, X_test)

    print(f"ML данные загружены: {X_train_processed.shape[0]} train, {X_test_processed.shape[0]} test")
    return X_train_processed, X_test_processed, y_train, y_test


def make_pipeline(classifier_key, preprocessor):
    if classifier_key in classifiers and classifier_key != 'statistical_rba':
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifiers[classifier_key])
        ])
        return clf
    else:
        return classifiers.get(classifier_key)


def find_risk_level_thresholds(model, X_val, y_val, model_name):
    print(f"Поиск порогов для трех уровней риска ({model_name})...")

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)
        if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1] 
    else:
        y_proba = model.decision_function(X_val)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    legit_probs = y_proba[y_val == 0]
    attack_probs = y_proba[y_val == 1]

    print(f"    Статистика вероятностей:")
    print(f"      Легитимные: min={legit_probs.min():.3f}, max={legit_probs.max():.3f}, mean={legit_probs.mean():.3f}")
    print(f"      Атаки: min={attack_probs.min():.3f}, max={attack_probs.max():.3f}, mean={attack_probs.mean():.3f}")

    low_medium_threshold = np.percentile(legit_probs, 80) if len(legit_probs) > 0 else 0.3
    medium_high_threshold = np.percentile(attack_probs, 25) if len(attack_probs) > 0 else 0.7
    if low_medium_threshold >= medium_high_threshold:
        print("     Автокоррекция порогов: низкий порог ≥ высокого")
      
        legit_mean = legit_probs.mean() if len(legit_probs) > 0 else 0.2
        attack_mean = attack_probs.mean() if len(attack_probs) > 0 else 0.6

        low_medium_threshold = min(legit_mean + 0.1, 0.4)  
        medium_high_threshold = max(attack_mean - 0.1, 0.6)  

        if low_medium_threshold >= medium_high_threshold:
            low_medium_threshold = 0.3
            medium_high_threshold = 0.7
            print("   Использованы фиксированные пороги: 0.3 и 0.7")

    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    binary_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5

    low_medium_threshold = min(low_medium_threshold, 0.45)  
    medium_high_threshold = max(medium_high_threshold, 0.55) 

    min_gap = 0.15
    if medium_high_threshold - low_medium_threshold < min_gap:
        gap_needed = min_gap - (medium_high_threshold - low_medium_threshold)
        low_medium_threshold = max(0.1, low_medium_threshold - gap_needed / 2)
        medium_high_threshold = min(0.9, medium_high_threshold + gap_needed / 2)

    print(f" Найдены пороги для {model_name}:")
    print(f"    Низкий риск: < {low_medium_threshold:.4f} (только пароль)")
    print(f"    Средний риск: {low_medium_threshold:.4f} - {medium_high_threshold:.4f} (требуется 2FA)")
    print(f"    Высокий риск: > {medium_high_threshold:.4f} (блокировка)")


    return {
        'low_medium': low_medium_threshold,
        'medium_high': medium_high_threshold,
        'binary': binary_threshold
    }, y_proba

@app.route('/learn', methods=['POST'])
def handle_request():
    json_data = f'{request.data.decode("utf-8")}'

    try:
        data = pd.read_json(io.StringIO(json_data))
        print("=" * 80)
        print(" ЗАГРУЗКА ДАННЫХ ДЛЯ СРАВНЕНИЯ МОДЕЛЕЙ")
        print("=" * 80)

        X_train_ml, X_test_ml, y_train_ml, y_test_ml = load_and_preprocess_data_ml(data)
        User_ID = data.iloc[0]['User_ID']
        print(f"\n СВОДКА ПО ДАННЫМ:")
        print(f"ML данные:        {X_train_ml.shape[0]} train, {X_test_ml.shape[0]} test")
        print(f"Распределение классов в ML тесте: {sum(y_test_ml == 0)} легитимных, {sum(y_test_ml == 1)} атак")

        categorical_cols = ['Country', 'Device_Type']
        numeric_cols = ['ASN', 'Login_Hour', 'IP_Address', 'User_Agent_String',
                        'Browser_Name_and_Version', 'OS_Name_and_Version', 'Is_Unknown_Country']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

        results_df = pd.DataFrame(columns=['Model', 'AUC', 'Accuracy', 'Prediction_Time_MS',
                                           'Requests_Per_Second', 'Normalized_Speed_Score', 'RBA_Score'])

        print("\n" + "=" * 80)
        print(" СРАВНЕНИЕ МОДЕЛЕЙ ДЛЯ RBA СИСТЕМЫ")
        print("=" * 80)

        for classifier_key in classifiers:
            print(f"\n Тестируем {classifier_key}...")

            pipeline = make_pipeline(classifier_key, preprocessor)

            print("    Используем ML данные...")

            pipeline.fit(X_train_ml, y_train_ml)

            sample_data = X_test_ml.iloc[:1]

            _ = pipeline.predict_proba(sample_data)

            start_single = time.time()
            for _ in range(1000):
                _ = pipeline.predict_proba(sample_data)
            single_prediction_time_ms = (time.time() - start_single) / 1000 * 1000  х

            requests_per_second = 1000 / (single_prediction_time_ms / 1000) if single_prediction_time_ms > 0 else 100000

            dtpredictions = pipeline.predict(X_test_ml)
            probs = pipeline.predict_proba(X_test_ml)[:, 1]

            y_test_current = y_test_ml

            auc_score = roc_auc_score(y_test_current, probs)
            accuracy = accuracy_score(y_test_current, dtpredictions)

            rba_score, normalized_speed_score = calculate_rba_score(
                auc_score,
                single_prediction_time_ms,
                auc_weight=0.9,
                speed_weight=0.1
            )

            results_df = pd.concat([results_df, pd.DataFrame([{
                'Model': classifier_key,
                'AUC': auc_score,
                'Accuracy': accuracy,
                'Prediction_Time_MS': single_prediction_time_ms,
                'Requests_Per_Second': requests_per_second,
                'Normalized_Speed_Score': normalized_speed_score,
                'RBA_Score': rba_score
            }])], ignore_index=True)

            print(f" {classifier_key.upper()}")
            print(f"   AUC: {auc_score:.6f}")
            print(f"   Accuracy: {accuracy:.6f}")
            print(f"   Время предсказания: {single_prediction_time_ms:.2f} мс")
            print(f"   Нормализованный Speed Score: {normalized_speed_score:.3f}")
            print(f"   RBA Score: {rba_score:.3f}")

        results_df = results_df.sort_values('RBA_Score', ascending=False)

        print("\n" + "=" * 100)
        print(" ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ ДЛЯ RBA СИСТЕМЫ")
        print("=" * 100)
        print(results_df[['Model', 'AUC', 'Accuracy', 'Prediction_Time_MS',
                          'Normalized_Speed_Score', 'RBA_Score']].sort_values('RBA_Score', ascending=False).to_string(
            index=False))

        best_model = results_df.iloc[0]

        print(f"\n РЕКОМЕНДАЦИЯ ДЛЯ RBA СИСТЕМЫ:")
        print(f"Лучшая модель: {best_model['Model']}")
        print(f"Показатели:")
        print(f"  • AUC: {best_model['AUC']:.4f} (качество обнаружения атак)")
        print(f"  • Время предсказания: {best_model['Prediction_Time_MS']:.2f} мс")
        print(f"  • Нормализованный Speed Score: {best_model['Normalized_Speed_Score']:.3f}")
        print(f"  • RBA Score: {best_model['RBA_Score']:.3f}")

        print(f"\n КРИТЕРИИ ПРИЕМЛЕМОСТИ ДЛЯ RBA:")
        print(f"Отличное качество: AUC > 0.90")
        print(f"Хорошее качество: AUC > 0.85")
        print(f"Приемлемая скорость: < 50 мс")
        print(f"Отличная скорость: < 10 мс")
        print(f"Минимальная производительность: > 100 запросов/сек")

        acceptable_models = results_df[
            (results_df['AUC'] > 0.7) &
            (results_df['Prediction_Time_MS'] < 30)
            ]

        if len(acceptable_models) > 0:
            print(f"\n МОДЕЛИ, СООТВЕТСТВУЮЩИЕ КРИТЕРИЯМ RBA:")
            print(acceptable_models[['Model', 'AUC', 'Prediction_Time_MS', 'RBA_Score']].to_string(index=False))
        else:
            print(f"\n  Нет моделей, полностью соответствующих критериям RBA")
            print("Рекомендуется сбор большего количества данных или feature engineering")

        print(f"\n РАСПРЕДЕЛЕНИЕ КЛАССОВ В ДАННЫХ:")
        print(f"ML данные: {sum(y_test_ml == 0)} легитимных, {sum(y_test_ml == 1)} атак")
        
        print("\n" + "=" * 80)
        print(" ДВА ПОРОГА ДЛЯ ТРЕХ УРОВНЕЙ РИСКА - ВИЗУАЛИЗАЦИЯ ЛУЧШЕЙ МОДЕЛИ")
        print("=" * 80)

        best_model_name = best_model['Model']

        best_pipeline = make_pipeline(best_model_name, preprocessor)
        best_pipeline.fit(X_train_ml, y_train_ml)
        X_val_optimal = X_test_ml
        y_val_optimal = y_test_ml

        thresholds, y_proba = find_risk_level_thresholds(
            best_pipeline, X_val_optimal, y_val_optimal, best_model_name
        )

        with open(f'data/LevelRisk/{User_ID}.txt', 'w') as f:
            f.write(f"{thresholds['low_medium']}\n")
            f.write(f"{thresholds['medium_high']}\n")
            f.write(f"{thresholds['binary']}\n")

        print(f"\n ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ ДЛЯ {best_model_name}:")
        print("=" * 60)
        print(f" ПОРОГИ ДЛЯ RBA СИСТЕМЫ:")
        print(f"    НИЗКИЙ РИСК: < {thresholds['low_medium']:.4f}")
        print(f"      - Действие: Обычная аутентификация (пароль)")
        print(f"      - Цель: Минимизация неудобств для пользователей")
        print()
        print(f"    СРЕДНИЙ РИСК: {thresholds['low_medium']:.4f} - {thresholds['medium_high']:.4f}")
        print(f"      - Действие: Дополнительная проверка (2FA, email, SMS)")
        print(f"      - Цель: Баланс безопасности и удобства")
        print()
        print(f"    ВЫСОКИЙ РИСК: > {thresholds['medium_high']:.4f}")
        print(f"      - Действие: Блокировка входа, уведомление безопасности")
        print(f"      - Цель: Максимальная защита от атак")

        joblib.dump(best_pipeline, f'data/Models/{User_ID}.pkl')
        response_data = f'{{"Learning Model": "Success", "User_ID": "{User_ID}"}}'

        return response_data, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print("Ошибка:", e)
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9293)
