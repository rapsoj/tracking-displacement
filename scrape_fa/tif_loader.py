import io
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import click

from .util.env_loader import require_env_file

def download_tif_files_from_public_folder(search_string: str, download_dir='.'):
    """
    Download all .tif files containing search_string from a public Google Drive folder using Google API.
    Args:
        folder_id (str): The ID of the public Google Drive folder.
        search_string (str): String to search for in file names.
        download_dir (str): Directory to save downloaded files.
    """
    folder_id = os.getenv("GDRIVE_ID")
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        try:
            api_key = input("Enter your Google API key: ").strip()
        except EOFError:
            print("No API key provided. Exiting.")
            return
        if not api_key:
            print("No API key provided. Exiting.")
            return
    service = build('drive', 'v3', developerKey=api_key)
    # Recursively search for .tif files in the folder and all subfolders
    def get_all_tif_files(parent_id):
        tif_files = []
        # List .tif files in this folder
        query = f"'{parent_id}' in parents and mimeType='image/tiff' and trashed=false"
        page_token = None
        while True:
            results = service.files().list(q=query, fields="nextPageToken, files(id, name)", pageToken=page_token).execute()
            files = results.get('files', [])
            tif_files.extend(files)
            page_token = results.get('nextPageToken', None)
            if not page_token:
                break
        # List subfolders
        folder_query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        page_token = None
        while True:
            results = service.files().list(q=folder_query, fields="nextPageToken, files(id, name)", pageToken=page_token).execute()
            folders = results.get('files', [])
            for folder in folders:
                tif_files.extend(get_all_tif_files(folder['id']))
            page_token = results.get('nextPageToken', None)
            if not page_token:
                break
        return tif_files

    all_tif_files = get_all_tif_files(folder_id)
    matched_files = [f for f in all_tif_files if search_string in f['name']]
    if not matched_files:
        print(f"No .tif files containing '{search_string}' found in folder or subfolders.")
        return
    os.makedirs(download_dir, exist_ok=True)
    for file in matched_files:
        file_id = file['id']
        file_name = file['name']
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(os.path.join(download_dir, file_name), 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        print(f"Downloading {file_name}...", end="\r")
        while not done:
            status, done = downloader.next_chunk()
        print(f"Downloaded {file_name}")


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('search_string')
@click.option('--download-dir', default='.', show_default=True,
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              help='Directory to save files')
@require_env_file(["GOOGLE_API_KEY", "GDRIVE_ID"])
def tif_loader(search_string: str, download_dir: str) -> None:
    """Download .tif files from a public Google Drive folder.

    FOLDER_ID is the Google Drive folder ID. SEARCH_STRING filters file names.
    """
    download_tif_files_from_public_folder(search_string, download_dir)


if __name__ == "__main__":
    tif_loader()

