"""한글 경로에서도 되는 사진 읽기·쓰기.

왜 필요한가: Windows의 OpenCV cv2.imread / cv2.imwrite는 경로에 한글이 있으면 조용히 실패한다
(읽기는 None, 쓰기는 False). 프로젝트 폴더가 '컴퓨터비전_미니프로젝트' 아래로 옮겨지면서 생긴 문제.
파일을 바이트로 직접 읽고(np.fromfile) 메모리에서 디코드/인코드해 경로 문제를 피한다.
(영상 cv2.VideoCapture / VideoWriter는 한글 경로에서도 동작함을 확인했다.)
"""
import pathlib
import numpy as np, cv2


def imread(path, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    except (FileNotFoundError, OSError):
        return None


def imwrite(path, img, params=()):
    ok, buf = cv2.imencode(pathlib.Path(path).suffix or ".jpg", img, list(params))
    if ok:
        buf.tofile(str(path))
    return ok
