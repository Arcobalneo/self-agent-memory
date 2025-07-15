from langchain_community.graphs.networkx_graph import (
    NetworkxEntityGraph,
    KnowledgeTriple,
)

# 创建一个新的图形实例
graph = NetworkxEntityGraph()

# 添加节点
graph.add_node("张三")
graph.add_node("李四")
graph.add_node("王五")

# 添加边（使用三元组：主体-关系-客体）
# 需要使用KnowledgeTriple类
graph.add_triple(KnowledgeTriple(subject="张三", predicate="是朋友", object_="李四"))
graph.add_triple(KnowledgeTriple(subject="李四", predicate="认识", object_="王五"))
graph.add_triple(KnowledgeTriple(subject="张三", predicate="想认识", object_="王五"))

# 查询图中的节点数
node_count = graph.get_number_of_nodes()
print(f"图中共有 {node_count} 个节点")

# 获取所有三元组（边）
triples = graph.get_triples()
print("图中的所有关系：")
for triple in triples:
    print(f"  {triple[0]} - {triple[2]} -> {triple[1]}")

# 检查特定的边是否存在
has_relation = graph.has_edge("张三", "李四")
print(f"张三和李四之间是否有直接关系: {has_relation}")

# 获取特定节点的邻居
neighbors = graph.get_neighbors("李四")
print(f"李四的相邻节点: {neighbors}")

# 保存图形到文件
graph.write_to_gml("my_knowledge_graph.gml")
print("图形已保存到 my_knowledge_graph.gml")
