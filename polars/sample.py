# (1) 工程データの加工関数
def process_data_transformation(file_path):
    df = pd.read_csv(file_path)

    # "EQUIPMENT_CODE_DATA_ITEM_NAME" 形式のカラム作成
    df["項目"] = df["EQUIPMENT_CODE"] + "_" + df["DATA_ITEM_NAME"]

    # VIN_CODE ごとに横持ち変換
    df_wide = df.pivot_table(index="VIN_CODE", columns="項目", values="MEASURED_TEXT", aggfunc="mean").reset_index()

    return df_wide

# (2) 検査データの加工関数（VIN_CODE のないものは除外）
def inspection_data_transformation(file_path):
    df_inspection = pd.read_csv(file_path)

    # データが存在しない場合、空のデータフレームを返す
    if df_inspection.empty:
        return pd.DataFrame()

    # 不具合項目を「責任工程_不具合名称」の形式で作成
    df_inspection["項目"] = df_inspection["責任工程"] + "_" + df_inspection["不具合名称"]

    # 各「責任工程_不具合名称」のカウント
    df_wide = df_inspection.pivot_table(index="VIN_NO", columns="項目", aggfunc="size", fill_value=0).reset_index()

    # 各不具合名称の合計（全工程合計）
    total_defect_counts = df_inspection.pivot_table(index="VIN_NO", columns="不具合名称", aggfunc="size", fill_value=0).reset_index()

    # 各工程ごとの合計値を追加
    process_totals = df_inspection.groupby(["VIN_NO", "責任工程"]).size().unstack(fill_value=0).reset_index()
    process_totals = process_totals.rename(columns=lambda x: f"{x}_合計" if x in ["上塗り", "シーリング", "ED"] else x)

    # 全不具合の合計
    total_defects = df_inspection.groupby("VIN_NO").size().reset_index(name="全不具合_合計")

    # すべての情報を統合
    df_wide = df_wide.merge(total_defect_counts, on="VIN_NO", how="outer").fillna(0)
    df_wide = df_wide.merge(process_totals, on="VIN_NO", how="outer").fillna(0)
    df_wide = df_wide.merge(total_defects, on="VIN_NO", how="outer").fillna(0)

    return df_wide

def merge_process_inspection(process_df, inspection_df):
    # VIN_CODE をキーに LEFT JOIN
    df_merged = process_df.merge(inspection_df, left_on="VIN_CODE", right_on="VIN_NO", how="left")

    # VIN_NO カラムは不要なので削除
    df_merged = df_merged.drop(columns=["VIN_NO"])

    # 検査データのカラム（VIN_CODE を除く）を 0 埋め
    inspection_cols = [col for col in df_merged.columns if col not in ["VIN_CODE"] + list(process_df.columns)]
    df_merged[inspection_cols] = df_merged[inspection_cols].fillna(0)

    return df_merged

def analyze_df_merged(df_merged):
    df_merged = df_merged.copy()
    # --- (1) 工程側の変数を分類 ---
    process_cols = [col for col in df_merged.columns if col.startswith("EQUIP_")]
    defect_cols = [col for col in df_merged.columns if not col.startswith("EQUIP_") and col != "VIN_CODE"]

    variable_classification = {}
    for col in process_cols:
        unique_vals = df_merged[col].dropna().unique()
        num_unique = len(unique_vals)

        if num_unique == 1:
            var_type = "ユニーク (議論対象外)"
        elif num_unique == 2:
            var_type = "2値"
        elif 3 <= num_unique <= 10:
            var_type = "離散"
        else:
            var_type = "連続"

        variable_classification[col] = {"種類": var_type, "ユニーク値": num_unique}

    df_var_class = pd.DataFrame.from_dict(variable_classification, orient="index")
    print("\n--- 変数分類結果 ---")
    df_var_class = df_var_class.reset_index().rename(columns={"index": "変数名", "種類": "分類"})
    print(df_var_class)

    # --- (2) 変数ごとの傾向分析 ---
    results = {}

    # 2値・離散変数の分析
    for col, info in variable_classification.items():
        if info["種類"] in ["2値", "離散"]:

            # 各値ごとのVIN_CODE数
            df_count = df_merged.groupby(col, observed=False, dropna=False)["VIN_CODE"].count().rename("車両台数")

            # 各検査項目の統計（不具合発生数の平均・不具合発生率）
            df_defect_stats = df_merged.groupby(col, observed=False, dropna=False)[defect_cols].agg(["mean", lambda x: (x > 0).mean()])
            df_defect_stats.columns = [f"{c[0]}_不具合発生数平均" if c[1] == "mean" else f"{c[0]}_不具合率" for c in df_defect_stats.columns]

            # 結合
            df_result = df_count.to_frame().join(df_defect_stats).reset_index()
            results[col] = df_result

    # 連続変数の分析（5つのbinに分ける）
    for col, info in variable_classification.items():
        if info["種類"] == "連続":

            df_merged[f"{col}_bin"] = pd.qcut(df_merged[col], q=5, duplicates="drop")

            # 各binごとのVIN_CODE数
            df_count = df_merged.groupby(f"{col}_bin", observed=False, dropna=False)["VIN_CODE"].count().rename("車両台数")

            # 各検査項目の統計（不具合発生数の平均・不具合発生率）
            df_defect_stats = df_merged.groupby(f"{col}_bin", observed=False, dropna=False)[defect_cols].agg(["mean", lambda x: (x > 0).mean()])
            df_defect_stats.columns = [f"{c[0]}_不具合発生数平均" if c[1] == "mean" else f"{c[0]}_不具合率" for c in df_defect_stats.columns]

            # 結合（連続変数の `bin` もカラムとして残す）
            df_result = df_count.to_frame().join(df_defect_stats).reset_index().rename(columns={f"{col}_bin": col})
            results[col] = df_result

    return df_var_class, results

def z_test_for_binary_variable(binary_var, analysis_results):
    results = []

    # 2値変数の集計データを取得
    df_binary = analysis_results[binary_var]

    for defect_col in df_binary.columns:
        if defect_col == "車両台数" or defect_col == binary_var:
            continue

        # 0と1のグループごとの値を取得（すでに平均値なのでそのまま使用）
        p_0 = df_binary.loc[0, defect_col]
        p_1 = df_binary.loc[1, defect_col]

        n_0 = df_binary.loc[0, "車両台数"]
        n_1 = df_binary.loc[1, "車両台数"]

        if n_0 == 0 or n_1 == 0:
            results.append({"工程変数": binary_var, "検査項目": defect_col, "p値": np.nan})
            continue  # どちらかのサンプルが0ならスキップ

        # 2群の比率検定（Z検定）
        p_combined = (p_0 * n_0 + p_1 * n_1) / (n_0 + n_1)
        se = np.sqrt(p_combined * (1 - p_combined) * (1 / n_0 + 1 / n_1))

        if se == 0:
            results.append({"工程変数": binary_var, "検査項目": defect_col, "p値": np.nan})
            continue  # 標準誤差が0ならスキップ

        z_score = (p_1 - p_0) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # 両側検定

        results.append({"工程変数": binary_var, "検査項目": defect_col, "p値": p_value})

    return results


def analyze_significant_pairs(df_var_class, analysis_results):
    all_results = []

    # 2値変数のリストを取得
    binary_vars = df_var_class[df_var_class["分類"] == "2値"]["変数名"].tolist()

    for binary_var in binary_vars:
        all_results.extend(z_test_for_binary_variable(binary_var, analysis_results))

    # 結果をデータフレームに変換
    df_results = pd.DataFrame(all_results)

    return df_results

def analyze_discrete_variables(df_merged, df_var_class):
    # 離散変数のリストを取得
    discrete_vars = df_var_class[df_var_class["分類"] == "離散"]["変数名"].tolist()
    
    # 検査結果のカラム（不具合発生数のカラム）を取得（工程変数を除外）
    defect_columns = [col for col in df_merged.columns if col not in df_var_class["変数名"].tolist() and col != "VIN_CODE"]
    
    # 結果を格納するリスト
    results = []
    
    # 各離散変数についてクラスカル・ウォリス検定を実施
    for var in discrete_vars:
        for defect_col in defect_columns:
            # 欠損値を除外
            df_subset = df_merged[[var, defect_col]].dropna()
            
            # 各カテゴリごとのグループを作成
            groups = [group[defect_col].values for _, group in df_subset.groupby(var)]
            
            if len(groups) > 1:  # 2つ以上のグループが必要
                stat, p_value = kruskal(*groups)
                results.append({
                    "工程変数": var,
                    "検査項目": defect_col,
                    "p値": p_value
                })
    
    # 結果をデータフレームに変換
    df_results = pd.DataFrame(results)
    return df_results


def analyze_continuous_variables(df_merged, df_var_class):
    # 連続変数のリストを取得
    continuous_vars = df_var_class[df_var_class["分類"] == "連続"]["変数名"].tolist()

    
    # 検査結果のカラム（不具合発生数のカラム）を取得（工程変数を除外）
    defect_columns = [col for col in df_merged.columns if col not in df_var_class["変数名"].tolist() and col != "VIN_CODE"]
    
    # 結果を格納するリスト
    results = []
    
    # 連続変数ごとにピアソン相関を計算
    for var in continuous_vars:
        for defect_col in defect_columns:
            # 欠損値を除外して相関計算
            df_subset = df_merged[[var, defect_col]].dropna()
            if df_subset.shape[0] > 1:  # データ点が1つ以下だと相関計算できない
                corr, p_value = pearsonr(df_subset[var], df_subset[defect_col])
                results.append({
                    "工程変数": var,
                    "検査項目": defect_col,
                    "相関係数": corr,
                    "p値": p_value,
                    "有意": p_value < 0.05
                })
    
    # 結果をデータフレームに変換
    df_results = pd.DataFrame(results)
    return df_results

def save_results_to_excel(df_var_class, analysis_results, df_continuous_variables, twovalue_pvalues, file_name):
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        # 変数の分類情報を保存
        df_var_class.to_excel(writer, sheet_name='変数分類', index=True)
        
        # analysis_results の各データフレームを個別のシートに保存
        for key, df in analysis_results.items():
            df.to_excel(writer, sheet_name=key, index=True)
        
        # 連続変数の分析結果を保存
        df_continuous_variables.to_excel(writer, sheet_name='連続変数_結果', index=False)
        
        # 2値変数の p 値結果を保存
        twovalue_pvalues.to_excel(writer, sheet_name='2値変数_p値', index=False)
        
        # シートごとにセルの幅を自動調整
        workbook = writer.book
        for sheet_name, df in {
            '変数分類': df_var_class, 
            '連続変数_結果': df_continuous_variables, 
            '2値変数_p値': twovalue_pvalues,
            **analysis_results
        }.items():
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2  # 余白を持たせる
                worksheet.set_column(i, i, max_len)
