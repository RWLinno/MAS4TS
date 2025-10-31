#!/usr/bin/env python3
"""
OnCallAgent 实验数据集准备脚本
基于真实OnCall文档生成问答测试集
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from model_configs import get_model_config, list_supported_models

class OnCallDatasetBuilder:
    """OnCall场景数据集构建器"""
    
    def __init__(self, data_dir: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        
        # 获取模型配置
        self.model_config = get_model_config(model_name)
        
        # 初始化Hugging Face模型
        print(f"正在加载模型: {model_name} ({self.model_config.name})")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # 使用pipeline方式，更简单易用
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            print(f"✓ 模型加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用预定义问答对...")
            self.generator = None
        
        # 加载现有文档
        self.documents = self._load_documents()
        
    def _load_documents(self) -> Dict[str, str]:
        """加载所有OnCall相关文档"""
        docs = {}
        
        # 加载autoops文档
        autoops_dir = self.data_dir / "documents" / "engineering" / "autoops"
        if autoops_dir.exists():
            for md_file in autoops_dir.glob("*.md"):
                with open(md_file, 'r', encoding='utf-8') as f:
                    docs[md_file.name] = f.read()
        
        # 加载其他工程文档
        for subdir in ["databases", "infrastructure", "monitoring"]:
            subdir_path = self.data_dir / "documents" / "engineering" / subdir
            if subdir_path.exists():
                for md_file in subdir_path.rglob("*.md"):
                    docs[f"{subdir}/{md_file.name}"] = md_file.read_text(encoding='utf-8')
        
        print(f"✓ 加载了 {len(docs)} 个文档")
        return docs
    
    def generate_qa_pairs(self, doc_content: str, doc_name: str, num_pairs: int = 5) -> List[Dict[str, Any]]:
        """使用开源LLM基于文档内容生成问答对"""
        if not self.generator:
            # 如果模型加载失败，返回预定义的问答对
            return self._get_predefined_qa_pairs(doc_name, num_pairs)
        
        # 构建用户内容
        user_content = f"""基于以下OnCall技术文档，生成{num_pairs}个真实的DevOps问答对：

文档内容：
{doc_content[:1500]}...

要求：
1. 问题应该是工程师在OnCall时会遇到的真实问题
2. 涵盖不同难度级别：简单查询、故障诊断、复杂分析
3. 包含不同类型：纯文本、需要图像分析、需要文档检索
4. 答案应该准确、可操作

请按照以下JSON格式输出：
{{
  "qa_pairs": [
    {{
      "question": "问题内容",
      "answer": "详细答案",
      "difficulty": "easy/medium/hard",
      "type": "text/image/retrieval/multimodal",
      "keywords": ["关键词1", "关键词2"],
      "source_doc": "{doc_name}"
    }}
  ]
}}"""
        
        # 使用模型特定的提示词模板
        prompt = self.model_config.prompt_template(user_content)
        
        try:
            # 使用模型特定的生成参数
            generation_params = self.model_config.generation_params.copy()
            generation_params["pad_token_id"] = self.generator.tokenizer.eos_token_id
            
            # 使用Hugging Face pipeline生成
            outputs = self.generator(prompt, **generation_params)
            
            # 提取生成的文本
            generated_text = outputs[0]['generated_text']
            
            # 提取assistant部分的回答
            assistant_start = generated_text.find("<|im_start|>assistant") + len("<|im_start|>assistant")
            assistant_response = generated_text[assistant_start:].strip()
            
            # 尝试解析JSON
            try:
                # 查找JSON部分
                json_start = assistant_response.find("{")
                json_end = assistant_response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = assistant_response[json_start:json_end]
                    result = json.loads(json_str)
                    qa_pairs = result.get("qa_pairs", [])
                    
                    # 验证生成的问答对格式
                    validated_pairs = []
                    for pair in qa_pairs:
                        if all(key in pair for key in ["question", "answer", "difficulty", "type"]):
                            # 确保必要字段存在
                            pair["keywords"] = pair.get("keywords", [])
                            pair["source_doc"] = doc_name
                            validated_pairs.append(pair)
                    
                    if validated_pairs:
                        return validated_pairs[:num_pairs]  # 限制数量
                        
            except json.JSONDecodeError:
                print(f"JSON解析失败，使用预定义问答对: {doc_name}")
                
        except Exception as e:
            print(f"使用开源模型生成问答对失败: {e}")
        
        # 如果生成失败，返回预定义问答对
        return self._get_predefined_qa_pairs(doc_name, num_pairs)
    
    def _get_predefined_qa_pairs(self, doc_name: str, num_pairs: int) -> List[Dict[str, Any]]:
        """预定义的问答对（当没有API key时使用）"""
        predefined_pairs = [
            {
                "question": "Redis连接超时问题如何排查？",
                "answer": "1. 检查网络连接和防火墙设置\n2. 验证Redis服务状态\n3. 检查连接池配置\n4. 查看Redis日志和监控指标",
                "difficulty": "medium",
                "type": "text",
                "keywords": ["redis", "timeout", "connection"],
                "source_doc": doc_name
            },
            {
                "question": "如何分析系统性能监控截图中的异常指标？",
                "answer": "需要结合截图分析：1. CPU使用率趋势\n2. 内存使用情况\n3. 磁盘I/O状态\n4. 网络流量变化",
                "difficulty": "hard",
                "type": "image",
                "keywords": ["monitoring", "performance", "analysis"],
                "source_doc": doc_name
            },
            {
                "question": "Kafka消费延迟过高的常见原因有哪些？",
                "answer": "1. 消费者处理能力不足\n2. 分区数配置不当\n3. 网络带宽限制\n4. Broker资源不足",
                "difficulty": "medium",
                "type": "text",
                "keywords": ["kafka", "latency", "consumer"],
                "source_doc": doc_name
            },
            {
                "question": "请查找@mysql_troubleshooting.md中关于死锁的解决方案",
                "answer": "需要检索MySQL故障排除文档，重点关注死锁检测、分析和预防措施",
                "difficulty": "easy",
                "type": "retrieval",
                "keywords": ["mysql", "deadlock", "troubleshooting"],
                "source_doc": doc_name
            },
            {
                "question": "如何结合日志和监控图表分析微服务调用链问题？",
                "answer": "1. 分析调用链追踪日志\n2. 对比各服务的响应时间监控\n3. 检查错误率和成功率指标\n4. 识别瓶颈服务",
                "difficulty": "hard",
                "type": "multimodal",
                "keywords": ["microservice", "tracing", "monitoring"],
                "source_doc": doc_name
            }
        ]
        
        # 随机选择指定数量的问答对
        return random.sample(predefined_pairs, min(num_pairs, len(predefined_pairs)))
    
    def build_dataset(self, total_size: int = 500) -> Dict[str, List[Dict[str, Any]]]:
        """构建完整的测试数据集"""
        all_qa_pairs = []
        
        # 根据文档数量平均分配问答对
        pairs_per_doc = max(1, total_size // len(self.documents))
        
        print("正在生成问答对...")
        for doc_name, doc_content in tqdm(self.documents.items()):
            qa_pairs = self.generate_qa_pairs(doc_content, doc_name, pairs_per_doc)
            all_qa_pairs.extend(qa_pairs)
        
        # 确保数据集大小
        if len(all_qa_pairs) > total_size:
            all_qa_pairs = random.sample(all_qa_pairs, total_size)
        elif len(all_qa_pairs) < total_size:
            # 如果不够，重复一些问答对
            while len(all_qa_pairs) < total_size:
                all_qa_pairs.extend(random.sample(all_qa_pairs[:total_size//2], 
                                                min(total_size - len(all_qa_pairs), total_size//2)))
        
        # 打乱顺序
        random.shuffle(all_qa_pairs)
        
        # 分配数据类型
        return self._categorize_dataset(all_qa_pairs)
    
    def _categorize_dataset(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """将数据集按类型分类"""
        categorized = {
            "text_queries": [],
            "image_queries": [],
            "retrieval_queries": [],
            "multimodal_queries": []
        }
        
        for qa_pair in qa_pairs:
            qa_type = qa_pair.get("type", "text")
            if qa_type == "text":
                categorized["text_queries"].append(qa_pair)
            elif qa_type == "image":
                categorized["image_queries"].append(qa_pair)
            elif qa_type == "retrieval":
                categorized["retrieval_queries"].append(qa_pair)
            elif qa_type == "multimodal":
                categorized["multimodal_queries"].append(qa_pair)
            else:
                categorized["text_queries"].append(qa_pair)
        
        return categorized
    
    def save_dataset(self, dataset: Dict[str, List[Dict[str, Any]]], output_dir: str):
        """保存数据集到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存完整数据集
        all_data = []
        for category, qa_pairs in dataset.items():
            all_data.extend(qa_pairs)
        
        # 分割训练集和测试集（8:2）
        random.shuffle(all_data)
        split_idx = int(len(all_data) * 0.8)
        train_set = all_data[:split_idx]
        test_set = all_data[split_idx:]
        
        # 保存文件
        with open(output_path / "train_set.json", 'w', encoding='utf-8') as f:
            json.dump(train_set, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "test_set.json", 'w', encoding='utf-8') as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "full_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # 保存数据集信息
        dataset_info = {
            "total_samples": len(all_data),
            "train_samples": len(train_set),
            "test_samples": len(test_set),
            "categories": {cat: len(pairs) for cat, pairs in dataset.items()},
            "difficulty_distribution": self._get_difficulty_distribution(all_data),
            "source_documents": list(self.documents.keys())
        }
        
        with open(output_path / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 数据集已保存到 {output_path}")
        print(f"  训练集: {len(train_set)} 样本")
        print(f"  测试集: {len(test_set)} 样本")
        print(f"  类别分布: {dataset_info['categories']}")
    
    def _get_difficulty_distribution(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取难度分布统计"""
        distribution = {"easy": 0, "medium": 0, "hard": 0}
        for qa_pair in qa_pairs:
            difficulty = qa_pair.get("difficulty", "medium")
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution

def main():
    parser = argparse.ArgumentParser(description="准备OnCallAgent实验数据集")
    parser.add_argument("--data_dir", required=True, help="数据目录路径")
    parser.add_argument("--output_dir", required=True, help="输出目录路径")
    parser.add_argument("--size", type=int, default=500, help="数据集大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct", 
                       help="用于生成问答对的Hugging Face模型名称")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 构建数据集
    builder = OnCallDatasetBuilder(args.data_dir, args.model_name)
    dataset = builder.build_dataset(args.size)
    builder.save_dataset(dataset, args.output_dir)
    
    print("✓ 数据集准备完成")

if __name__ == "__main__":
    main()
