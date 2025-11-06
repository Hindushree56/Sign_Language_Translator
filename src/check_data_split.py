import os

for split in ['train', 'val', 'test']:
    path = f'data_split/{split}'
    if not os.path.exists(path):
        print(f'{path} ❌ not found')
    else:
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        print(f'{split} ✅ found {len(folders)} class folders')
        if len(folders) > 0:
            first = folders[0]
            count = len(os.listdir(os.path.join(path, first)))
            print(f'👉 Example: folder "{first}" has {count} images')
    print('-'*60)
