"""Resumable small-model downloads; completed, verified files stay offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    'yolox_nano.onnx': {
        'url':'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx',
        'sha256':'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d',
        'minimum_bytes':3_000_000,
    },
    'face_detection_yunet_2023mar.onnx': {
        'url':'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        'sha256':'8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
        'minimum_bytes':200_000,
    },
    'osnet_x0_25_msmt17.onnx': {
        'url':'https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx',
        'sha256':'e78604f4ccda49b8f41cd0f8f7303800ce75d2361895ebb0729513c1bf53d277',
        'minimum_bytes':900_000,
    },
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(65536),b''): h.update(block)
    return h.hexdigest()


def valid(path,spec):
    if not path.is_file() or path.stat().st_size<spec['minimum_bytes']:
        return False
    return not spec['sha256'] or digest(path)==spec['sha256']


def download(path,spec,retries=8):
    if valid(path,spec):
        print(f'{path.name}: already verified',flush=True)
        return
    part=path.with_name(path.name+'.part')
    for attempt in range(retries):
        offset=part.stat().st_size if part.exists() else 0
        headers={'User-Agent':'FaceBodyTracking/1.0'}
        if offset: headers['Range']=f'bytes={offset}-'
        try:
            request=urllib.request.Request(spec['url'],headers=headers)
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status==206:
                    match=re.match(r'bytes (\d+)-\d+/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=offset:
                        raise OSError('Server returned an unexpected download range')
                    total=int(match[2])
                else:
                    offset=0
                    total=int(response.headers.get('Content-Length',0))
                print(f'{path.name}: starting at {offset:,} bytes',flush=True)
                with part.open('ab' if offset else 'wb') as output:
                    written=offset
                    last_report=time.monotonic()
                    while True:
                        block=response.read(32768)
                        if not block: break
                        output.write(block)
                        output.flush()
                        written+=len(block)
                        if time.monotonic()-last_report>5:
                            print(f'  {written:,} / {total or "unknown"} bytes',flush=True)
                            last_report=time.monotonic()
                if total and written!=total:
                    raise OSError(f'Incomplete response: {written}/{total}')
            if not valid(part,spec):
                # A full response with the wrong digest must never be installed.
                raise ValueError(f'Checksum or length failed for {path.name}; remove {part} and retry')
            part.replace(path)
            print(f'{path.name}: verified',flush=True)
            return
        except urllib.error.HTTPError as error:
            if error.code==416 and valid(part,spec):
                part.replace(path)
                return
            if attempt==retries-1: raise
            print(f'{path.name}: HTTP {error.code}; retrying',flush=True)
        except (OSError,urllib.error.URLError) as error:
            if attempt==retries-1: raise
            print(f'{path.name}: interrupted ({error}); partial file preserved',flush=True)
        time.sleep(min(2+attempt,10))
    raise RuntimeError(f'Could not download {path.name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=ROOT/'models')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    args.models.mkdir(parents=True,exist_ok=True)
    for name,spec in MODELS.items():
        path=args.models/name
        if args.check_only:
            if not valid(path,spec): parser.error(f'Model missing or invalid: {path}')
        else: download(path,spec)
    arcface=args.models/'embedder_arcface.onnx'
    if not arcface.is_file():
        parser.error('Copy models/embedder_arcface.onnx from the previous FaceLocking project')
    manifest={name:{'sha256':digest(args.models/name),'bytes':(args.models/name).stat().st_size,
                    'url':spec['url']} for name,spec in MODELS.items()}
    manifest[arcface.name]={'sha256':digest(arcface),'bytes':arcface.stat().st_size,'source':'Copied original project ArcFace model'}
    (args.models/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('All four models are present and verified. No downloads needed at runtime.')


if __name__=='__main__': main()
