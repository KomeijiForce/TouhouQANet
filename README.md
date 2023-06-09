# 东方问答网络/東方QAネットワーク/TouhouQANet
![Instance](https://github.com/KomeijiForce/TouhouQANet/blob/main/instance.png)

[知识图谱/知識グラフ/Knowledge Graph（TouhouQANet.jsonl）](TouhouQANet.jsonl)

这是一个基于(Chat)GPT-4和东方Project的知识库Thbwiki构建的知识图谱项目，包含了带有问题增强信息的边（元组），可以用于知识检索和为大模型打补丁。

このプロジェクトは、東方Projectに基づいた(Chat)GPT-4とThbwikiの知識ベースを使用して構築された知識グラフのプロジェクトです。この知識グラフには、問題を増強する情報を持つエッジ（タプル）が含まれており、知識の検索や大規模モデルへのパッチ提供に使用することができます。

This is a knowledge graph project built on the basis of Thbwiki by (Chat)GPT-4, a knowledge base based on the Touhou Project. It contains edges (tuples) with enhanced information for questions, which can be used for knowledge retrieval and patching for large models.


此外本项目还附带了一个拥有125个问题的东方问答测试集，可以用于检验大模型在东方Project上的知识水平。

さらに、このプロジェクトには125問を含む東方QAテストセットが付属しており、大規模モデルの東方Projectにおける知識水準を検証するために使用できます。

In addition, this project also includes a Touhou QA test set with 125 questions, which can be used to assess the knowledge level of large models on the Touhou Project.


[阅读报告/レポート読み/Read the Paper](https://github.com/KomeijiForce/TouhouQANet/blob/main/TouhouQANet.pdf)

# 补丁性能/パッチング表現/Patching Performance

![Main Performance](https://github.com/KomeijiForce/TouhouQANet/blob/main/Patched_LLMs.png)

[抽取的实现/検索器の実現/Implementation of Retriever](https://github.com/KomeijiForce/TouhouQANet/blob/main/retrivers.py)

# 统计学特征/統計情報/Statistics

![Statistics](https://github.com/KomeijiForce/TouhouQANet/blob/main/TouhouQANet_Stats.png)

# 标注/注釈/Annotation

![Statistics](https://github.com/KomeijiForce/TouhouQANet/blob/main/pipeline.png)
