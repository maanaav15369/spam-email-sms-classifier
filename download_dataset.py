from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

DATA_DIR = Path(__file__).resolve().parent / 'data'
RAW_FILE = DATA_DIR / 'SMSSpamCollection'
CSV_FILE = DATA_DIR / 'sms_spam.csv'
ZIP_URL = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'


def download_and_prepare():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        print(f'Found dataset: {RAW_FILE}')
    else:
        print('Downloading official UCI SMS Spam Collection...')
        try:
            with urlopen(ZIP_URL, timeout=30) as response:
                content = response.read()
        except Exception as exc:
            raise RuntimeError(
                'Could not download the UCI dataset automatically.\n'
                'Open the official page:\n'
                'https://archive.ics.uci.edu/dataset/228/sms-spam-collection\n'
                'Download the dataset and place the file named SMSSpamCollection\n'
                f'inside: {DATA_DIR}\n\nOriginal error: {exc}'
            ) from exc

        with ZipFile(BytesIO(content)) as zf:
            matches = [name for name in zf.namelist() if Path(name).name == 'SMSSpamCollection']
            if not matches:
                raise RuntimeError('The downloaded UCI archive did not contain SMSSpamCollection.')
            with zf.open(matches[0]) as src, RAW_FILE.open('wb') as dst:
                dst.write(src.read())
        print(f'Saved raw UCI dataset to: {RAW_FILE}')

    rows = []
    with RAW_FILE.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2 and parts[0] in {'ham', 'spam'}:
                rows.append((parts[0], parts[1]))

    if not rows:
        raise RuntimeError('No valid ham/spam rows were found in SMSSpamCollection.')

    import pandas as pd
    df = pd.DataFrame(rows, columns=['label', 'message'])
    df.to_csv(CSV_FILE, index=False)
    print(f'Created CSV: {CSV_FILE}')
    print(f'Rows: {len(df)} | Ham: {(df.label == "ham").sum()} | Spam: {(df.label == "spam").sum()}')
    return df


if __name__ == '__main__':
    download_and_prepare()
