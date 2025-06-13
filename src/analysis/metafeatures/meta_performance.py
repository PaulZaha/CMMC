import pandas as pd
import concurrent.futures


def read_data():
    df = pd.read_excel('data.xlsx')
    df['diff'] = df['kirsch'] - df['raw']
    df['inoutdiff'] = (df['avg_inside'] - df['avg_outside']).abs()
    df['sd'] = df['variance'].pow(1./2)
    return df


def compute_aggregated_level_for_threshold_with_params(df, threshold, metaclassifier, vorzeichen):

    if vorzeichen == 'pos':
        kirsch_filtered_df = df[df[metaclassifier] > threshold]
        raw_filtered_df = df[df[metaclassifier] <= threshold]
    else:
        kirsch_filtered_df = df[df[metaclassifier] <= threshold]
        raw_filtered_df = df[df[metaclassifier] > threshold]
    
    metaperf = (kirsch_filtered_df['kirsch'].mean() * len(kirsch_filtered_df) + 
                raw_filtered_df['raw'].mean() * len(raw_filtered_df)) / len(df)
    
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
        'neg',
        'pos',
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
    df = read_data()
    
    df = df[df['case'].str.contains(modality)]

    thresholds = sorted(df[metaclassifier].unique())
    
    best_threshold = None
    best_metaperf = -float('inf')
    

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:

        results = executor.map(compute_aggregated_level_for_threshold_with_params, 
                               [df]*len(thresholds), thresholds, [metaclassifier]*len(thresholds), [vorzeichen]*len(thresholds))
        
        for threshold, metaperf in results:
            print(f"Performance für {modality} mit {metaclassifier}: Schwellenwert {threshold}: {metaperf}")
            
            if metaperf > best_metaperf:
                best_metaperf = metaperf
                best_threshold = threshold
    
    return [modality, metaclassifier, best_threshold, best_metaperf]

def fix_dataframe():
    df = read_data()
    df = df.drop_duplicates(subset=['file'])
    print(df)
    df.to_excel('data.xlsx',index=False)

def main():
    find_best_thresholds()

if __name__ == '__main__':
    main()
