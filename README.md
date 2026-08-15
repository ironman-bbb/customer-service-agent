# 企业售后智能客服 Agent

这是一条单一、可追踪的真实 AI 主线：

```text
用户
  → LangChain create_agent
  → Chat Model（可配置 Key / URL / Model）
  → Tools
      ├── orders.json 订单查询和退款审核
      └── Embedding Model → Milvus 语义检索
  → Pydantic 结构化回答
  → PostgreSQL 保存用户/助手消息
```

订单暂时保存在 JSON，方便学习和修改；PostgreSQL 只用于 LangGraph 持久化记忆。后期可在保持 Repository 接口不变的情况下，由你将 JSON 替换为数据库。

## 项目结构

```text
customer-service-agent/
├── .env                         # 本地密钥和连接信息，不提交 Git
├── data/orders.json              # 当前阶段的模拟订单
├── knowledge/                   # 原始 Markdown 业务知识
├── scripts/
│   └── ingest_knowledge.py       # 文档切块、Embedding、写入 Milvus
├── src/customer_service_agent/
│   ├── app.py                    # 唯一启动入口
│   ├── agent.py                  # 真实 LangChain Agent
│   ├── config.py                 # 所有外部配置
│   ├── repositories.py           # JSON 订单数据访问
│   ├── embeddings.py             # Embedding 模型适配
│   ├── ingestion.py              # 文档加载与切块
│   ├── retrieval.py              # Milvus 入库与向量检索
│   ├── tools.py                  # Agent Tools 与退款规则
│   ├── schemas.py                # Pydantic 输入输出
│   └── prompts.py                # Agent 系统规则
└── tests/                        # 不消耗 API 的单元测试
```

## 1. 填写 `.env`

打开项目根目录的 `.env`，填入你本机的实际配置。不要把 Key 或数据库密码发到聊天、截图或 Git。

```dotenv
CHAT_MODEL_NAME=
CHAT_API_KEY=
CHAT_BASE_URL=

EMBEDDING_MODEL_NAME=
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_DIMENSIONS=

DB_URL=postgresql://用户名:密码@localhost:5432?sslmode=disable
POSTGRES_DATABASE=customer_service_agent

MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_DATABASE=default
MILVUS_COLLECTION=customer_service_knowledge
```

`EMBEDDING_DIMENSIONS` 必须与 Embedding API 实际返回的向量长度一致。更换维度后，必须使用 `--recreate` 重建 Milvus collection。

## 2. 创建环境并安装

```powershell
cd "C:\Users\55302\Desktop\Gkk上位机学习\ai学习\customer-service-agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## 3. 准备 PostgreSQL 记忆库

只需在 PostgreSQL 中创建空数据库 `customer_service_agent`。启动 Agent 时，LangGraph `PostgresSaver.setup()` 会自动创建 checkpoint 表，持久化消息、Tool 调用和 Agent 状态。PostgreSQL 中不保存订单。

## 4. 将知识库写入 Milvus

首次执行：

```powershell
python scripts/ingest_knowledge.py
```

如果更换了 Embedding 模型的向量维度：

```powershell
python scripts/ingest_knowledge.py --recreate
```

`--recreate` 会删除 `.env` 中 `MILVUS_COLLECTION` 指定的 collection，然后按新维度创建。

## 5. 启动真实 Agent

```powershell
python -m customer_service_agent.app
```

输入同一个会话 ID 时，程序会从 PostgreSQL 恢复历史；程序重启后仍然有效。

建议测试：

```text
专业版多少钱？
查询 ORDER-1001
它可以退款吗？
退出并重启，使用同一会话 ID
刚才的订单是什么？
```

## 代码阅读顺序

1. `app.py`：看完整调用链。
2. `agent.py`：看 `ChatOpenAI` 和 `create_agent`。
3. `tools.py`：看 Agent 可选择的三个 Tools。
4. `repositories.py`：看 JSON 订单查询；`app.py` 中的 `PostgresSaver` 负责持久化 Agent 记忆。
5. `embeddings.py` + `retrieval.py`：看 Embedding 和 Milvus 检索。
6. `ingestion.py`：看文档如何进入向量库。

## 安全与生产边界

- SQL 使用 Psycopg 参数化查询，不拼接用户输入。
- API Key 、PostgreSQL 密码和 Milvus Token 只存放于 `.env`。
- Agent 只审核退款资格，没有真实退款权限。
- Milvus 返回的文档是不可信数据，不能覆盖 System Prompt。
- 生产上线仍需增加身份认证、租户隔离、权限校验、重试限流、Tracing、评估数据集和部署流水线。

## 测试

单元测试不连接模型、PostgreSQL 或 Milvus，因此不产生 API 费用：

```powershell
python -m unittest discover -s tests -v
```

真实的端到端验证需先完成 `.env`、PostgreSQL 初始化和 Milvus 向量入库。
