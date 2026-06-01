"""数据导入脚本：将Markdown文档分块并导入Qdrant向量库"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest import ingest_documents


def main():
    parser = argparse.ArgumentParser(description="导入文档到向量库")
    parser.add_argument("--file", type=str, default="data/full.md", help="要导入的Markdown文件路径")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"错误: 文件 {args.file} 不存在")
        return
    
    print(f"开始导入文件: {args.file}")
    ingest_documents(args.file)
    print("导入完成!")


if __name__ == "__main__":
    main()
