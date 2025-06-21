import os
import re
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ShotInfo:
    """镜头信息数据类"""
    shot_number: str
    shot_title: str
    shot_type: str
    frame_size: str
    camera_movement: str
    duration: str
    emotion_tags: List[str]
    content: str
    notes: str

class ShotPromptGenerator:
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 一次性生成图片和视频提示词的系统提示
        self.prompt_system = """你是专业的AI图像和视频生成提示词专家。请根据镜头信息同时生成图片提示词和视频提示词。

        第一步：请根据镜头信息，决定图片的设计，图片对应的是视频的首帧，图片要包含人物、场景，人物与场景的关系，要具体的，具体的描述，不要抽象的描述，不要有感受类的描述，不要有心理的描述，第一要有人物的状态，人物之间的关系，场景的描述，人物与场景的关系
        第二步：请根据图片的设计，设计视频的提示词，视频提示词要增加动作运镜等，每个人物物体都要设计动作，并加入运镜方法（视频的目标等与生成视频无关的东西不要出现），动作要简单、单一，千万不要复杂的动作

        要求：
        1. 图片提示词：描述视频的首帧静态画面，控制在150字以内
        2. 视频提示词：基于首帧描述接下来的动作和变化，控制在150字以内
        3. 必须写实风格，避免卡通化
        4. 具体描述人物、场景、光线、构图
        5. 使用中英文输出

        输出格式（必须严格按照以下格式，不能有任何变化）：
        === 图片生成中文提示词 ===
        [图片提示词内容]

        === 图片生成英文提示词 ===
        [图片提示词内容]

        === 视频生成中文提示词 ===
        [视频提示词内容]

        === 视频生成英文提示词 ===
        [视频提示词内容]
        """

    def parse_script(self, script_path: str) -> List[ShotInfo]:
        """解析分镜头脚本"""
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        shots = []
        # 使用正则表达式分割镜头，适配新格式 SHOT X - 标题
        shot_pattern = r'\*\*SHOT (\d+) - ([^*]+?)\*\*'
        shot_matches = re.finditer(shot_pattern, content)
        
        shot_blocks = []
        for match in shot_matches:
            start_pos = match.start()
            shot_blocks.append((match.group(1), match.group(2), start_pos))
        
        # 提取每个镜头的完整内容
        for i, (shot_num, shot_title, start_pos) in enumerate(shot_blocks):
            end_pos = shot_blocks[i + 1][2] if i + 1 < len(shot_blocks) else len(content)
            shot_content = content[start_pos:end_pos]
            
            shot_info = self._parse_shot_content(shot_num, shot_title, shot_content)
            if shot_info:
                shots.append(shot_info)
        
        return shots

    def _parse_shot_content(self, shot_number: str, shot_title: str, content: str) -> Optional[ShotInfo]:
        """解析单个镜头内容"""
        try:
            # 提取景别、时长等信息
            bracket_pattern = r'\[([^\]]+)\]'
            brackets = re.findall(bracket_pattern, content)
            
            frame_size = brackets[0] if len(brackets) > 0 else ""
            duration = brackets[1] if len(brackets) > 1 else ""
            
            # 情感标签
            emotion_tags = []
            tag_pattern = r'#(\w+)'
            emotion_tags = re.findall(tag_pattern, content)
            
            # 提取画面内容
            content_match = re.search(r'\*\*画面内容：\*\*\s*([^*]+)', content)
            shot_content = content_match.group(1).strip() if content_match else ""
            
            # 提取情绪重点（作为备注）
            emotion_match = re.search(r'\*\*情绪重点：\*\*\s*([^*]+)', content)
            notes = emotion_match.group(1).strip() if emotion_match else ""
            
            # 提取AI描述重点
            ai_desc_match = re.search(r'\*\*AI描述重点：\*\*\s*([^*]+)', content)
            if ai_desc_match:
                ai_desc = ai_desc_match.group(1).strip()
                shot_content += "\n\nAI描述重点：" + ai_desc
            
            return ShotInfo(
                shot_number=shot_number,
                shot_title=shot_title.strip(),
                shot_type="",  # 这个格式中没有明确的镜头类型
                frame_size=frame_size,
                camera_movement="",  # 运动信息包含在景别中
                duration=duration,
                emotion_tags=emotion_tags,
                content=shot_content,
                notes=notes
            )
        except Exception as e:
            print(f"解析镜头 {shot_number} 时出错: {e}")
            return None

    def generate_prompts(self, shot_info: ShotInfo) -> Tuple[Dict[str, str], str]:
        """一次性生成图片和视频提示词（中英文）"""
        user_message = f"""
镜头信息：
- 镜头编号：{shot_info.shot_number}
- 镜头标题：{shot_info.shot_title}
- 镜头类型：{shot_info.shot_type}
- 景别：{shot_info.frame_size}
- 镜头运动：{shot_info.camera_movement}
- 时长：{shot_info.duration}
- 情感标签：{', '.join(shot_info.emotion_tags)}
- 画面内容：{shot_info.content}
- 备注：{shot_info.notes}

请同时生成这个镜头的图片提示词和视频提示词：
"""
        
        print(f"🔍 调试信息 - 镜头 {shot_info.shot_number}")
        print(f"📝 用户输入消息：\n{user_message}")
        print(f"🤖 系统提示词：\n{self.prompt_system[:200]}...")
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": self.prompt_system},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            print(f"📤 AI返回内容：\n{content}")
            print(f"📏 返回内容长度：{len(content)}")
            
            # 解析返回的内容，分离中英文提示词
            result = {
                "image_cn": "",
                "image_en": "",
                "video_cn": "",
                "video_en": ""
            }
            
            # 检查是否包含预期的标记
            print(f"🔍 检查标记存在性：")
            print(f"  - 图片中文标记: {'=== 图片生成中文提示词 ===' in content}")
            print(f"  - 图片英文标记: {'=== 图片生成英文提示词 ===' in content}")
            print(f"  - 视频中文标记: {'=== 视频生成中文提示词 ===' in content}")
            print(f"  - 视频英文标记: {'=== 视频生成英文提示词 ===' in content}")
            
            # 使用分割方式解析，更直观
            import re
            
            # 先按照 === 分割内容
            sections = re.split(r'=== ([^=]+) ===', content)
            
            print(f"🔍 分割后的sections数量: {len(sections)}")
            for i, section in enumerate(sections):
                if section.strip():
                    print(f"  Section {i}: {section.strip()[:50]}...")
            
            # 遍历sections，找到对应的内容
            for i in range(1, len(sections), 2):  # 奇数位置是标题，偶数位置是内容
                if i + 1 < len(sections):
                    title = sections[i].strip()
                    content_part = sections[i + 1].strip()
                    
                    print(f"🏷️ 标题: '{title}' -> 内容长度: {len(content_part)}")
                    
                    if '图片生成中文提示词' in title:
                        result['image_cn'] = content_part
                    elif '图片生成英文' in title:
                        result['image_en'] = content_part
                    elif '视频生成中文提示词' in title:
                        result['video_cn'] = content_part
                    elif '视频生成英文' in title:
                        result['video_en'] = content_part
            
            print(f"📋 解析结果：")
            print(f"  - 图片中文: {len(result['image_cn'])} 字符")
            print(f"  - 图片英文: {len(result['image_en'])} 字符")
            print(f"  - 视频中文: {len(result['video_cn'])} 字符")
            print(f"  - 视频英文: {len(result['video_en'])} 字符")
            
            if result['image_cn']:
                print(f"  - 图片中文内容预览: {result['image_cn'][:50]}...")
            if result['image_en']:
                print(f"  - 图片英文内容预览: {result['image_en'][:50]}...")
            
            return result, content
            
        except Exception as e:
            print(f"❌ 生成提示词失败: {e}")
            return {"image_cn": "", "image_en": "", "video_cn": "", "video_en": ""}, ""

    def process_all_shots(self, script_path: str, output_dir: str):
        """处理所有镜头"""
        # 解析脚本
        shots = self.parse_script(script_path)
        print(f"解析到 {len(shots)} 个镜头")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理每个镜头
        for i, shot in enumerate(shots, 1):
            print(f"处理镜头 {i}/{len(shots)}: {shot.shot_number}")
            
            # 一次性生成图片和视频提示词（中英文）
            prompts, raw_content = self.generate_prompts(shot)
            print(f"提示词生成完成")
            
            # 保存到文件
            output_file = os.path.join(output_dir, f"SHOT_{shot.shot_number}_提示词.txt")
            self._save_prompts(shot, prompts, output_file, raw_content)
            
            print(f"镜头 {shot.shot_number} 处理完成\n")

    def _save_prompts(self, shot_info: ShotInfo, prompts: Dict[str, str], output_file: str, raw_content: str = ""):
        """保存提示词到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"镜头编号：{shot_info.shot_number}\n")
            f.write(f"镜头类型：{shot_info.shot_type} ({shot_info.shot_title})\n")
            f.write(f"时长：{shot_info.duration}\n")
            f.write(f"景别：{shot_info.frame_size}\n")
            f.write(f"运动：{shot_info.camera_movement}\n\n")
            
            f.write("=== 图片生成中文提示词 ===\n")
            f.write(prompts.get("image_cn", "") + "\n\n")
            
            f.write("=== 图片生成英文提示词 ===\n")
            f.write(prompts.get("image_en", "") + "\n\n")
            
            f.write("=== 视频生成中文提示词 ===\n")
            f.write(prompts.get("video_cn", "") + "\n\n")
            
            f.write("=== 视频生成英文提示词 ===\n")
            f.write(prompts.get("video_en", "") + "\n\n")
            
            # 保存AI返回的原始内容
            if raw_content:
                f.write("=" * 50 + "\n")
                f.write("=== AI返回的原始内容（用于调试） ===\n")
                f.write("=" * 50 + "\n")
                f.write(raw_content + "\n")

def main():
    # 配置参数
    # 从ali_utils导入API密钥
    from ali_utils import API_KEY
    SCRIPT_PATH = "../create_video_ais/youtube/宝宝海洋历险_20250531/04_镜头设计/01_分镜头脚本.md"
    OUTPUT_DIR = "../create_video_ais/youtube/宝宝海洋历险_20250531/04_镜头设计/02_镜头提示词"
    
    # 初始化生成器
    generator = ShotPromptGenerator(API_KEY)
    
    # 处理所有镜头
    generator.process_all_shots(SCRIPT_PATH, OUTPUT_DIR)
    
    print("所有镜头处理完成！")

if __name__ == "__main__":
    main()