import pandas as pd
import concurrent.futures


def read_data():
    df = pd.read_excel('results.xlsx')
    df['diff'] = df['kirsch'] - df['nonkirsch']
    df['inoutdiff'] = (df['avg_inside'] - df['avg_outside']).abs()
    df['sd'] = df['variance'].pow(1./2)

    modalities = [
        'XRay',
        'Dermoscopy',
        'Fundus',
        'Mammo',
        'Microscopy',
        'OCT',
        'US'
    ]

    train_parts = []
    test_parts = []

    for modality in modalities:
        modality_df = df[df['case'].str.contains(modality)]
        split = int(len(modality_df) * 0.8)
        train_parts.append(modality_df.iloc[:split])
        test_parts.append(modality_df.iloc[split:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def compute_aggregated_level_for_threshold_with_params(df, threshold, metaclassifier, vorzeichen):
    if vorzeichen == 'pos':
        kirsch_filtered_df = df[df[metaclassifier] > threshold]
        nonkirsch_filtered_df = df[df[metaclassifier] <= threshold]
    else:
        kirsch_filtered_df = df[df[metaclassifier] <= threshold]
        nonkirsch_filtered_df = df[df[metaclassifier] > threshold]

    if kirsch_filtered_df.empty or nonkirsch_filtered_df.empty:
        print(kirsch_filtered_df)
        print(nonkirsch_filtered_df)
        return threshold, float('-inf')

    metaperf = (kirsch_filtered_df['kirsch'].mean() * len(kirsch_filtered_df) +
                nonkirsch_filtered_df['nonkirsch'].mean() * len(nonkirsch_filtered_df)) / len(df)

    return threshold, metaperf


def find_best_thresholds():
    modalities = [
        'XRay',
        'Dermoscopy',
        'Fundus',
        'Mammo',
        'Microscopy',
        'OCT',
        'US'
    ]
    metaclassifiers = [
        'entropy',
        'entropy',
        'entropy',
        'entropy',
        'entropy',
        'entropy',
        'sd'
    ]
    vorzeichen = [
        'pos',
        'neg',
        'neg',
        'pos',
        'pos',
        'pos',
        'neg'
    ]
    values = []
    for modality, metaclassifier, v in zip(modalities, metaclassifiers, vorzeichen):
        output = find_best_threshold(modality, metaclassifier, v)
        values.append(output)
    print(values)


def find_best_threshold(modality, metaclassifier, vorzeichen):
    train_df, test_df = read_data()
    train_df = train_df[train_df['case'].str.contains(modality)]
    test_df = test_df[test_df['case'].str.contains(modality)]

    thresholds = sorted(train_df[metaclassifier].unique()[:-1])
    best_threshold = None
    best_metaperf = -float('inf')

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        results = executor.map(compute_aggregated_level_for_threshold_with_params,
                               [train_df] * len(thresholds), thresholds,
                               [metaclassifier] * len(thresholds), [vorzeichen] * len(thresholds))

        for threshold, metaperf in results:
            print(f"Performance für {modality} mit {metaclassifier}: Schwellenwert {threshold}: {metaperf}")
            if metaperf > best_metaperf:
                best_metaperf = metaperf
                best_threshold = threshold

    thres, testperf = compute_aggregated_level_for_threshold_with_params(test_df, best_threshold, metaclassifier, vorzeichen)
    print(f"Testing set für {modality} mit Schwellenwert {best_threshold}: Performance {testperf}")
    return [modality, metaclassifier, best_threshold, best_metaperf, testperf]


def fix_dataframe():
    df = pd.read_excel('data/final_results_miccai_v2.xlsx')
    df = df.drop_duplicates(subset=['file'])
    df.to_excel('data/final_results_miccai_v2.xlsx', index=False)


def main():
    find_best_thresholds()


if __name__ == '__main__':
    main()
