import os
import json
import re
from openai import OpenAI
from pathlib import Path
# 从相对路径导入API密钥
from ali_utils import API_KEY, MAX_CONVERSATION_TURNS, VIDEO_SCRIPT_PROMPT

# Initialize the OpenAI client with Aliyun Dashscope
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 专门用于prompt优化的系统提示词 - 更新为通用化版本
PROMPT_OPTIMIZATION_SYSTEM = """您是一个专业的AI图像生成prompt优化专家。您的任务是根据原始prompt和场景描述，生成高质量、写实的图像生成prompt。

## 核心优化原则

### 1. 写实风格强制要求
- **必须强调写实风格**：每个prompt都必须包含photorealistic, realistic, lifelike等关键词
- **避免卡通化**：明确排除cartoon, anime, illustration, stylized等风格

### 2. 格式要求三步骤：1、人物、实体、场景描述，比如amy：具体的情况描述；狼群：具体的描述；父母：具体的描述等等；森林：具体的描述
        2、人物、实体、场景的关系描述；物理位置，他们的动作等
        3、其他参数要求：高质量、写实等等
### 3. 去除与图片描述无关的描述，比如画面外的东西，剧情的描述、视频的描述等，只保留与构图沟通风格相关的描述


## 输出格式要求
请直接返回优化后的英文prompt
"""

class ChatService:
    def __init__(self, project_id=None):
        self.conversation_history = []
        self.project_id = project_id
        # Initialize with the video script system prompt
        self.system_message = {"role": "system", "content": VIDEO_SCRIPT_PROMPT}
        
        # 如果有项目ID，加载已有对话历史
        if project_id:
            from project_manage.project_service import ProjectService
            project = ProjectService.get_project(project_id)
            if project and project.conversation:
                self.conversation_history = project.conversation
    
    def optimize_image_prompt(self, original_prompt: str, shot_description: str = "") -> str:
        """使用大模型API优化图像生成prompt"""
        try:
            # 构建优化请求
            optimization_request = f"""
请优化以下图像生成prompt，确保生成写实风格的高质量图像：

原始prompt:
{original_prompt}
"""
            
            messages = [
                {"role": "system", "content": PROMPT_OPTIMIZATION_SYSTEM},
                {"role": "user", "content": optimization_request}
            ]
            
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                temperature=0.3,  # 较低温度保证一致性
                max_tokens=600
            )
            
            optimized_prompt = completion.choices[0].message.content.strip()
            
            # 移除可能的引号或markdown标记
            optimized_prompt = re.sub(r'^["`\']+|["`\']+$', '', optimized_prompt)
            optimized_prompt = re.sub(r'^```\w*\n?|```$', '', optimized_prompt, flags=re.MULTILINE)
            
            return optimized_prompt
            
        except Exception as e:
            print(f"⚠️ Prompt优化失败，使用原始prompt: {e}")
            return original_prompt
    
