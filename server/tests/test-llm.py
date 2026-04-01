"""
测试 LiteLLM Gateway
"""
import os
from pathlib import Path
import httpx
import openai
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

API_BASE = os.getenv("TEST_LLM_BASE", "http://litellm-service.primus-safe.svc.cluster.local:4000/v1")
API_KEY = os.getenv("TEST_LLM_KEY")

client = openai.OpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
    http_client=httpx.Client(verify=False),
)


def test_non_streaming():
    """测试非流式请求"""
    print("=" * 50)
    print("测试非流式请求")
    print("=" * 50)
    
    response = client.chat.completions.create(
        model="gpt-5.2",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Say hello in one sentence."}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}")
    if response.usage:
        print(f"\nUsage:")
        print(f"  prompt_tokens: {response.usage.prompt_tokens}")
        print(f"  completion_tokens: {response.usage.completion_tokens}")
        print(f"  total_tokens: {response.usage.total_tokens}")
    print("✅ 非流式请求成功")


def test_streaming():
    """测试流式请求"""
    print("\n" + "=" * 50)
    print("测试流式请求")
    print("=" * 50)
    
    response = client.chat.completions.create(
        model="gpt-5.2",
        max_tokens=100,
        stream=True,
        stream_options={"include_usage": True},  # 开启 usage 返回
        messages=[
            {"role": "user", "content": "Count from 1 to 5."}
        ]
    )
    
    # 流式处理
    full_content = ""
    usage = None
    
    print("Response: ", end="")
    for chunk in response:
        # 获取内容
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_content += content
        
        # 最后一个 chunk 包含 usage（当 include_usage=True）
        if chunk.usage:
            usage = chunk.usage
    
    print()  # 换行
    
    # 打印 usage
    if usage:
        print(f"\nUsage:")
        print(f"  prompt_tokens: {usage.prompt_tokens}")
        print(f"  completion_tokens: {usage.completion_tokens}")
        print(f"  total_tokens: {usage.total_tokens}")
    
    print("✅ 流式请求成功")


def test_models():
    """测试获取模型列表"""
    print("\n" + "=" * 50)
    print("测试模型列表")
    print("=" * 50)
    
    models = client.models.list()
    print("可用模型:")
    for model in models.data:
        print(f"  - {model.id}")
    print("✅ 模型列表获取成功")


def test_thinking():
    """测试 thinking 模式"""
    print("\n" + "=" * 50)
    print("测试 Thinking 模式")
    print("=" * 50)
    
    response = client.chat.completions.create(
        model="gpt-5.2",
        max_tokens=500,
        extra_body={
            "reasoning_effort": "high"
        },
        messages=[
            {"role": "user", "content": "What is 25 * 37? Think step by step."}
        ]
    )
    
    message = response.choices[0].message
    
    # 检查是否有思考内容（不同模型可能用不同的字段名）
    print("\n--- 响应结构 ---")
    print(f"message 字段: {dir(message)}")
    
    # 尝试获取 reasoning_content（OpenAI o1/o3 使用）
    reasoning = getattr(message, 'reasoning_content', None)
    if reasoning:
        print(f"\n🧠 Thinking 内容:\n{reasoning}")
    
    # 尝试获取其他可能的字段
    for attr in ['reasoning', 'thinking', 'thought', 'internal_thoughts']:
        val = getattr(message, attr, None)
        if val:
            print(f"\n🧠 {attr}:\n{val}")
    
    print(f"\n📝 Response:\n{message.content}")
    
    # 显示 usage（reasoning tokens 会单独显示）
    if response.usage:
        print(f"\nUsage:")
        print(f"  prompt_tokens: {response.usage.prompt_tokens}")
        print(f"  completion_tokens: {response.usage.completion_tokens}")
        print(f"  total_tokens: {response.usage.total_tokens}")
        # 检查是否有 reasoning_tokens
        reasoning_tokens = getattr(response.usage, 'reasoning_tokens', None)
        if reasoning_tokens:
            print(f"  reasoning_tokens: {reasoning_tokens}")
    
    print("✅ Thinking 模式测试完成")

if __name__ == "__main__":
    try:
        # test_models()
        test_non_streaming()
        # test_streaming()
        # test_thinking()
        print("\n" + "=" * 50)
        print("🎉 所有测试通过!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
