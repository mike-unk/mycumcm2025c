如果要使用 uv 安装虚拟环境：

```bash
uv init
uv add -r .agents/skills/cumcm-step-review/requirements.txt
uv sync
```

那么运行时就要：

```bash
.venv/bin/python xxx.py
```

题目和数据都放在根目录。
