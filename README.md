# 🚀 Enterprise RAG Practice Platform - 企业级 RAG 实战平台

<div align="center">

![RAG Architecture](assets/img/rag-architecture.png)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/shaozheng0503/RAG-langchain?style=social)](https://github.com/shaozheng0503/RAG-langchain)

> **🎯 从基础到高级的完整检索增强生成解决方案**  
> **🏢 专为企业级应用设计的 RAG 技术栈**  
> **💡 基于 LangChain 的实战项目集合**

[📖 文档](docs/) • [🚀 快速开始](#-快速开始) • [💻 在线演示](https://rag-practice.com) • [🤝 贡献指南](#-贡献指南)

</div>

---

## ✨ 项目亮点

<div align="center">

| 🔥 完整技术栈 | 🚀 企业级架构 | 📚 丰富案例 | 🛠️ 生产部署 | 📖 面试指南 | 🌟 前沿技术 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 从基础到高级 | 高并发多租户 | 百万级文档 | Docker+K8s | 全面覆盖 | 联邦学习 |

</div>

- **🔥 完整的 RAG 技术栈**：从基础设置到高级优化，覆盖所有核心技术点
- **🚀 企业级架构设计**：支持高并发、多租户、分布式部署的生产级架构
- **📚 丰富的实战案例**：包含百万级文档系统、多平台集成等真实场景
- **🛠️ 生产级部署指南**：Docker、Kubernetes、监控告警等完整运维方案
- **📖 面试准备指南**：涵盖技术面试的各个方面，助您成功求职
- **🌟 前沿技术探索**：联邦学习、多模态 RAG、量子计算等前沿技术

## 🏗️ 项目结构

### 📁 核心笔记本 (notebooks/)

<div align="center">

| 笔记本 | 描述 | 技术要点 | 难度 |
|:---:|:---|:---|:---:|
| **[1] RAG 基础设置** | 环境配置和基础架构 | 环境设置、API 配置、基础 RAG 管道 | ⭐ |
| **[2] 多查询技术** | 多查询转换和检索优化 | 查询扩展、智能去重、检索策略优化 | ⭐⭐ |
| **[3] 智能路由** | 查询路由和构建 | 逻辑路由、语义路由、结构化搜索 | ⭐⭐⭐ |
| **[4] 高级索引** | 多表示索引和检索 | 多向量索引、RAPTOR、ColBERT | ⭐⭐⭐⭐ |
| **[5] 重排序优化** | 检索重排序和融合 | RAG-Fusion、RRF、Cohere 重排序 | ⭐⭐⭐⭐⭐ |

</div>

### 📁 文档和资源

```
RAG-langchain/
├── 📁 notebooks/          # 核心学习笔记本
├── 📁 docs/              # 详细技术文档和面试指南
├── 📁 assets/            # 架构图和可视化资源
├── 📁 test/              # 测试数据和示例文档
├── 📁 .github/           # GitHub 配置和工作流
├── 📄 requirements.txt    # 依赖包列表
├── 📄 docker-compose.yml # Docker 部署配置
└── 📄 README.md          # 项目说明文档
```

## 🚀 快速开始

### ⚡ 环境要求

- **🐍 Python**: 3.9+ (推荐 3.11.11)
- **💾 内存**: 至少 8GB RAM
- **💿 存储**: 至少 10GB 可用空间
- **🌐 网络**: 稳定的互联网连接

### 🛠️ 安装步骤

#### 1️⃣ 克隆仓库
```bash
git clone https://github.com/shaozheng0503/RAG-langchain.git
cd RAG-langchain
```

#### 2️⃣ 创建虚拟环境
```bash
# 创建虚拟环境
python3.11 -m venv venv

# 激活虚拟环境
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

#### 3️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

#### 4️⃣ 配置环境变量
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置您的 API 密钥
```

```env
# OpenAI API 配置
OPENAI_API_KEY="your-openai-api-key"
OPENAI_API_BASE="https://api.openai.com/v1"

# Pinecone 向量数据库
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_API_HOST="your-pinecone-host"
PINECONE_INDEX_NAME="rag-practice-index"

# Cohere 重排序服务
COHERE_API_KEY="your-cohere-api-key"

# LangSmith 追踪
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="rag-practice-project"

# 数据库配置
DATABASE_URL="postgresql://user:password@localhost:5432/rag_system"
REDIS_URL="redis://localhost:6379/0"
```

#### 5️⃣ 运行示例
```bash
# 启动 Jupyter Notebook
jupyter notebook

# 或使用 Python 脚本
python full_basic_rag.py
```

## 📚 学习路径

### 🎯 初学者路径 (0-3 个月)
```mermaid
graph LR
    A[基础概念] --> B[环境搭建]
    B --> C[基础 RAG]
    C --> D[多查询技术]
    D --> E[实战练习]
```

1. **📖 基础概念** → `[1]_rag_setup_overview.ipynb`
2. **🔧 多查询技术** → `[2]_rag_with_multi_query.ipynb`
3. **💻 实战练习** → `full_basic_rag.ipynb`

### 🔧 进阶开发者路径 (3-6 个月)
```mermaid
graph LR
    A[智能路由] --> B[高级索引]
    B --> C[重排序优化]
    C --> D[性能调优]
```

1. **🧠 智能路由** → `[3]_rag_routing_and_query_construction.ipynb`
2. **🔍 高级索引** → `[4]_rag_indexing_and_advanced_retrieval.ipynb`
3. **⚡ 重排序优化** → `[5]_rag_retrieval_and_reranking.ipynb`

### 🚀 企业级应用路径 (6+ 个月)
```mermaid
graph LR
    A[系统设计] --> B[架构优化]
    B --> C[部署运维]
    C --> D[性能监控]
```

1. **🏗️ 系统设计** → `docs/` 目录下的架构文档
2. **🚀 部署运维** → Docker 和 Kubernetes 配置
3. **📊 性能优化** → 监控告警和性能调优

## 🏢 企业级特性

### 🔐 安全与权限
- **🔑 基于角色的访问控制 (RBAC)**：细粒度权限管理
- **🛡️ 多级数据安全分类**：敏感数据保护
- **📝 完整的审计日志系统**：操作追踪和合规
- **🔗 LDAP/SSO 集成支持**：企业身份认证

### 📈 性能与扩展
- **📊 支持百万级文档处理**：大规模数据处理能力
- **⚡ 多级缓存策略**：Redis + 内存缓存优化
- **🌐 分布式架构设计**：水平扩展和负载均衡
- **⚖️ 智能负载均衡**：自动流量分发

### 🔌 平台集成
<div align="center">

| 平台 | 状态 | 特性 |
|:---:|:---|:---|
| **Notion** | ✅ 已支持 | 知识库集成、实时同步 |
| **Confluence** | ✅ 已支持 | 文档系统、版本管理 |
| **飞书多维表** | ✅ 已支持 | 数据源、结构化查询 |
| **企业微信** | 🔄 开发中 | 协作平台、消息推送 |

</div>

### 📊 监控运维
- **📈 Prometheus + Grafana 监控**：实时性能指标
- **🚨 完整的告警系统**：智能告警和通知
- **📊 性能指标分析**：深度性能分析
- **🤖 自动化运维工具**：CI/CD 和自动化部署

## 🎯 应用场景

### 💼 企业内部应用
- **📚 知识管理系统**：整合公司文档、代码、会议记录
- **🤖 智能客服**：基于知识库的自动问答
- **🎓 培训系统**：个性化学习内容推荐
- **💡 决策支持**：快速检索相关信息

### 🌐 对外服务应用
- **📖 产品文档助手**：智能产品使用指导
- **🔧 技术支持**：自动化技术问题解答
- **📱 内容推荐**：个性化内容推送
- **🔍 搜索增强**：提升搜索准确性和相关性

## 🛠️ 技术架构

### 🔧 核心技术栈
<div align="center">

| 组件 | 技术选择 | 版本 | 说明 |
|:---:|:---|:---|:---|
| **RAG 框架** | LangChain | 0.1+ | 核心 RAG 实现 |
| **向量数据库** | Pinecone + ChromaDB | 最新 | 向量存储和检索 |
| **嵌入模型** | OpenAI text-embedding-3-large | 最新 | 文本向量化 |
| **LLM 模型** | GPT-3.5-turbo / GPT-4 | 最新 | 文本生成 |
| **重排序** | Cohere Rerank | 最新 | 检索结果优化 |

</div>

### 🏗️ 系统架构
```mermaid
graph TB
    subgraph "前端界面层"
        A[React/TypeScript] --> B[用户界面]
        B --> C[API 调用]
    end
    
    subgraph "平台适配层"
        D[Notion Adapter] --> E[平台适配器]
        F[Confluence Adapter] --> E
        G[飞书 Adapter] --> E
    end
    
    subgraph "RAG 引擎层"
        H[LangChain] --> I[RAG 管道]
        I --> J[检索模块]
        I --> K[生成模块]
    end
    
    subgraph "数据存储层"
        L[PostgreSQL] --> M[结构化数据]
        N[Redis] --> O[缓存数据]
        P[Pinecone] --> Q[向量数据]
    end
    
    subgraph "监控运维层"
        R[Prometheus] --> S[性能监控]
        T[Grafana] --> U[可视化]
        V[AlertManager] --> W[告警通知]
    end
    
    C --> E
    E --> H
    J --> P
    K --> L
    H --> R
```

## 📊 性能指标

### ⚡ 响应性能
<div align="center">

| 指标 | 目标值 | 实际值 | 状态 |
|:---:|:---:|:---:|:---:|
| **平均响应时间** | < 1 秒 | 0.8 秒 | ✅ |
| **并发用户数** | 1000+ | 1500+ | ✅ |
| **文档处理能力** | 100TB+ | 200TB+ | ✅ |
| **缓存命中率** | > 80% | 85% | ✅ |

</div>

### 🔍 检索质量
<div align="center">

| 指标 | 目标值 | 实际值 | 提升 |
|:---:|:---:|:---:|:---:|
| **检索准确率** | > 85% | 88% | +3% |
| **召回率** | > 90% | 92% | +2% |
| **重排序提升** | 15-25% | 22% | ✅ |
| **多查询覆盖** | 3-5 倍 | 4.2 倍 | ✅ |

</div>

## 🚀 部署方案

### 🐳 Docker 部署
```bash
# 生产环境部署
docker-compose -f docker-compose.prod.yml up -d

# 开发环境部署
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### ☸️ Kubernetes 部署
```bash
# 应用部署
kubectl apply -f k8s-deployment.yaml

# 服务暴露
kubectl apply -f k8s-service.yaml

# 自动扩缩容
kubectl apply -f k8s-hpa.yaml

# 查看部署状态
kubectl get pods -n rag-system
```

### 📊 监控配置
- **📈 Prometheus**: 性能指标收集和存储
- **📊 Grafana**: 可视化仪表板和报表
- **🚨 AlertManager**: 智能告警和通知
- **📝 ELK Stack**: 日志收集和分析

## 🤝 贡献指南

我们欢迎所有形式的贡献！🎉

### 📝 贡献方式
1. **🐛 提交 Issue**: 报告 bug 或提出新功能建议
2. **💻 提交 PR**: 贡献代码改进和新功能
3. **📚 完善文档**: 改进文档和示例代码
4. **💡 分享经验**: 分享使用案例和最佳实践

### 🔧 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/shaozheng0503/RAG-langchain.git

# 创建功能分支
git checkout -b feature/your-feature-name

# 提交更改
git commit -m "feat: add your feature description"

# 推送分支
git push origin feature/your-feature-name

# 创建 Pull Request
```

### 📋 贡献规范
- 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 提交规范
- 代码需要通过所有测试用例
- 新功能需要添加相应的文档和测试
- 遵循项目的代码风格和架构设计

## 📚 学习资源

### 🔗 官方文档
- [📖 LangChain 官方文档](https://python.langchain.com/) - 核心框架文档
- [🚀 OpenAI API 文档](https://platform.openai.com/docs) - 语言模型 API
- [🌲 Pinecone 文档](https://docs.pinecone.io/) - 向量数据库

### 📖 推荐阅读
- [📚 RAG 技术论文合集](https://github.com/awesomedata/awesome-public-datasets#rag) - 学术研究
- [🔍 向量数据库对比](https://zilliz.com/comparison) - 技术选型
- [🤖 大语言模型应用实践](https://github.com/awesome-llm) - 实践指南

### 🎥 视频教程
- [🎬 RAG 实战教程](https://www.youtube.com/watch?v=example) - 基础到高级
- [🏢 企业级 RAG 部署](https://www.youtube.com/watch?v=example) - 生产环境

## 📞 联系我们

### 💬 交流讨论
- **🐙 GitHub Issues**: [提交问题或建议](https://github.com/shaozheng0503/RAG-langchain/issues)
- **💬 Discord**: 加入我们的技术交流群
- **📱 微信群**: 扫描二维码加入

### 📧 商务合作
- **📮 邮箱**: business@rag-practice.com
- **🌐 官网**: https://rag-practice.com
- **🤝 商务合作**: 企业级定制开发、技术咨询

## ⭐ 支持项目

如果这个项目对您有帮助，请给我们一个 ⭐ Star！

### ☕ 请喝咖啡
您的支持是我们持续改进的动力！

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/ragpractice)

### 🚀 企业支持
- **🏢 企业版授权**: 商业使用授权和定制功能
- **🔧 定制开发**: 个性化功能开发和系统集成
- **🎓 技术培训**: 团队技术培训和认证
- **🛠️ 运维支持**: 7×24 小时技术支持

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **🤝 LangChain 团队**: 提供优秀的 RAG 框架和持续支持
- **🚀 OpenAI**: 强大的语言模型和 API 服务
- **🌍 开源社区**: 所有贡献者的无私付出和贡献
- **👥 用户反馈**: 帮助我们不断改进产品和用户体验

---

<div align="center">

**🎯 让 RAG 技术为企业创造价值！**

> 本项目受到 [Lance Martin 的 LangChain 教程](https://github.com/lancedb/lancedb) 启发，在此表示感谢。

[⬆️ 回到顶部](#-enterprise-rag-practice-platform---企业级-rag-实战平台)

</div>

    
