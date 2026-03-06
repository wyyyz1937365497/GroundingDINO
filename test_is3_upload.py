"""
固定配置的 iS3 上传测试脚本（基于 Python 版 MetadataAPI）

运行：
python test_is3_upload.py
"""

from __future__ import annotations

import json
from pathlib import Path

from is3_metadata_api import MetadataAPI


CONFIG = {
    "file": "temp_rtsp_frame.jpg",
    "base_url": "https://server.is3.net.cn",
    "file_base_url": "https://file.is3.net.cn",
    "folder_id": "",
    "project_id": "",
    "ak": "",
    "sk": "",
    "timeout": 20,
}


def main() -> int:
    file_path = Path(CONFIG["file"])
    if not file_path.exists():
        print(f"[错误] 文件不存在: {file_path}")
        return 2

    api = MetadataAPI(
        base_url=CONFIG["base_url"],
        prj_id=CONFIG["project_id"],
        headers={
            "X-Access-Key": CONFIG["ak"],
            "X-Secret-Key": CONFIG["sk"],
        },
        folder_id=CONFIG["folder_id"],
        file_base_url=CONFIG["file_base_url"],
        timeout=CONFIG["timeout"],
    )

    print("=" * 72)
    print("iS3 上传测试（MetadataAPI）")
    print(f"file       : {file_path}")
    print(f"folder_id  : {CONFIG['folder_id']}")
    print(f"project_id : {CONFIG['project_id']}")
    print(f"base_url   : {CONFIG['base_url']}")
    print("=" * 72)

    try:
        list_resp = api.get_file_list(CONFIG["folder_id"], page_num=1, page_size=1)
        print("[目录校验] getFileList 返回:")
        print(json.dumps(list_resp, ensure_ascii=False, indent=2)[:800])
    except Exception as exc:
        print(f"[目录校验失败] {exc}")

    try:
        result = api.upload_file(str(file_path))
        print("[上传返回]")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if isinstance(result, dict) and result.get("success", False):
            data = result.get("data") or {}
            rel_url = data.get("url")
            full_url = f"{CONFIG['file_base_url']}{rel_url}" if rel_url else None
            print(f"✅ 上传成功, file_id={data.get('id')}")
            print(f"✅ 访问地址: {full_url}")
            return 0

        print("❌ 上传失败：接口返回 success=false")
        return 1
    except Exception as exc:
        print(f"❌ 上传异常: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
