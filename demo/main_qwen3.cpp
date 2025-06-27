#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/qwen3.h"
#include <sstream>
#include <regex>

// 新增：过滤掉<think>标签内容的函数
std::string filter_think_tags(const std::string& text) {
    std::string result = text;
    
    // 查找<think>标签的位置
    size_t think_start = result.find("<think>");
    while (think_start != std::string::npos) {
        size_t think_end = result.find("</think>", think_start);
        if (think_end != std::string::npos) {
            // 删除从<think>到</think>的所有内容，包括标签本身
            result.erase(think_start, think_end - think_start + 8); // 8是"</think>"的长度
        } else {
            // 如果没有找到结束标签，只删除开始标签
            result.erase(think_start, 7); // 7是"<think>"的长度
        }
        think_start = result.find("<think>");
    }
    
    // 去除开头的空白字符
    size_t first_non_space = result.find_first_not_of(" \n\r\t");
    if (first_non_space != std::string::npos) {
        result = result.substr(first_non_space);
    }
    
    return result;
}

int32_t generate(const model::Qwen3Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false, std::string* output_text = nullptr) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  std::string raw_output;  // 存储原始输出
  
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      if (next != 151645 && next != 151644) {
        words.push_back(next);
      }
    }
    if (model.is_sentence_ending(next)) {
      break;
    }

    if (is_prompt) {
      next = tokens.at(pos + 1);
    }
    pos += 1;
  }
  
  std::string decoded = model.decode(words);
  
  // 过滤think标签
  std::string filtered = filter_think_tags(decoded);
  
  if (need_output) {
    printf("%s", filtered.data());
    fflush(stdout);
  }
  if (output_text != nullptr) {
    *output_text = filtered;  // 保存过滤后的文本
  }
  return std::min(pos, total_steps);
}

std::string fill_template(const std::string& content) {
  const std::string format =
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
  std::string result = format;
  size_t pos = result.find("%s");
  if (pos != std::string::npos) {
    result.replace(pos, 2, content);
  }
  return result;
}

// 构建多轮对话的完整prompt
std::string build_conversation(const std::vector<std::pair<std::string, std::string>>& history, 
                              const std::string& new_input) {
  std::stringstream ss;
  
  // 添加历史对话
  for (const auto& turn : history) {
    ss << "<|im_start|>user\n" << turn.first << "<|im_end|>\n";
    ss << "<|im_start|>assistant\n" << turn.second << "<|im_end|>\n";
  }
  
  // 添加新的用户输入
  ss << "<|im_start|>user\n" << new_input << "<|im_end|>\n";
  ss << "<|im_start|>assistant\n";
  
  return ss.str();
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];

  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }

  // 存储对话历史: pair<用户输入, 助手回复>
  std::vector<std::pair<std::string, std::string>> conversation_history;
  
  std::cout << "=== 多轮对话模式 (输入 'quit' 退出, 'clear' 清空历史) ===" << std::endl;
  
  while (true) {
    std::string user_input;
    std::cout << "\n用户> ";
    std::getline(std::cin, user_input);
    
    // 检查是否退出
    if (user_input == "quit" || user_input == "exit") {
      std::cout << "再见！" << std::endl;
      break;
    }
    
    // 检查是否清空历史
    if (user_input == "clear") {
      conversation_history.clear();
      std::cout << "对话历史已清空。" << std::endl;
      continue;
    }
    
    // 构建包含历史的完整prompt
    std::string full_prompt = build_conversation(conversation_history, user_input);
    
    // 生成回复
    std::cout << "助手> ";
    fflush(stdout);
    
    auto start = std::chrono::steady_clock::now();
    std::string assistant_response;
    int steps = generate(model, full_prompt, 2560, true, &assistant_response);
    auto end = std::chrono::steady_clock::now();
    
    // 保存到历史（已经过滤了think标签）
    conversation_history.push_back({user_input, assistant_response});
    
    // 限制历史长度（保留最近5轮对话）
    if (conversation_history.size() > 5) {
      conversation_history.erase(conversation_history.begin());
    }
    
    // 输出性能统计
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\n[耗时:%.2fs]", duration);
    fflush(stdout);
  }
  
  return 0;
}