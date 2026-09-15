import os
import urllib.request
import sys

def download_file(url, output_path):
    print(f"Downloading {url} -> {output_path}...")
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Custom progress reporter
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 100.0 / totalsize
            s = f"\rProgress: {percent:.1f}% ({readsofar / (1024*1024):.1f} MB of {totalsize / (1024*1024):.1f} MB)"
            sys.stdout.write(s)
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rProgress: {readsofar / (1024*1024):.1f} MB")
            sys.stdout.flush()

    try:
        # Using a User-Agent header to avoid potential blocks on direct API calls
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading file via Python: {e}")
        # Try fallback using curl
        print("Attempting download using curl fallback...")
        cmd = f'curl -L -o "{output_path}" "{url}"'
        res = os.system(cmd)
        if res == 0:
            print("Download complete via curl!")
        else:
            raise RuntimeError(f"Failed to download {url} using Python and curl.")

def main():
    datasets = {
        "METR-LA": {
            "url": "https://raw.githubusercontent.com/leilin-research/GCGRNN/master/data/METR-LA_traffic_speed/metr-la.h5",
            "path": "data/raw/METR-LA/metr-la.h5"
        },
        "PEMS-BAY": {
            "url": "https://zenodo.org/records/4263971/files/pems-bay.h5?download=1",
            "path": "data/raw/PEMS-BAY/pems-bay.h5"
        }
    }
    
    for name, info in datasets.items():
        if os.path.exists(info["path"]):
            print(f"Dataset '{name}' already exists at '{info['path']}'. Skipping.")
        else:
            download_file(info["url"], info["path"])

if __name__ == '__main__':
    main()
