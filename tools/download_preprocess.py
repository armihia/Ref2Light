from pathlib import Path
import argparse
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
URLS={
    'face_landmarker.task':'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'pose_landmarker_lite.task':'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}


def main():
    parser=argparse.ArgumentParser(description='Download official models for ordinary-view localization and face capture, retaining valid existing files.')
    parser.add_argument('--out',type=Path,default=ROOT/'assets')
    args=parser.parse_args()
    directory=args.out.resolve();directory.mkdir(parents=True,exist_ok=True)
    for name,url in URLS.items():
        path=directory/name
        if path.is_file():
            if not zipfile.is_zipfile(path):raise ValueError('The existing file is not a valid task bundle: '+str(path))
            print('Keeping existing file: ',path);continue
        temporary=path.with_suffix('.download')
        with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as stream:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                stream.write(chunk)
        if temporary.stat().st_size<100000:raise ValueError('The downloaded file is too small; check the network connection: '+str(temporary))
        if not zipfile.is_zipfile(temporary):raise ValueError('The response is not a task model bundle; check the network connection: '+str(temporary))
        temporary.replace(path)
        print('Downloaded: ',path)


if __name__=='__main__':main()
