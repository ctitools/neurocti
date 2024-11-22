# Inference - here is how to run it?

---

Since our model is based on mistral-nemo-2407 we use the model specific inference code:
https://github.com/mistralai/mistral-inference

At least in principle these instructions apply to other models of the mistral familiy, and to some extend the larger LLM ecosystem, though LoRA compatibility is a kind of arcane topic and we had trouble applying out LoRA using the more popular vLLM framework.

## First lets download some weights


## Model download

| Name        | Download | md5sum |
|-------------|-------|-------|
| Nemo Base | https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-base-2407.tar | `c5d079ac4b55fc1ae35f51f0a3c0eb83` |
| Nemo Instruct | https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-instruct-2407.tar | `296fbdf911cb88e6f0be74cd04827fe7` |

## Install mistral-inference

```bash
pip install mistral-inference
```
## Gettings started

Here is some example code running inference with mistral-nemo, once without and once with out lora applied. Since we are using a base model, the model will try to complete our prompt "APT28 is..."
...

### Base model without LoRA


```python
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

tokenizer = MistralTokenizer.from_file("./models/mistral-nemo-base/tekken.json")  
model = Transformer.from_folder("./models/mistral-nemo-base") 

prompt = "APT28 is"

completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=1024, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
```

    /usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 2.1.3
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"


     đầu tiên của Liên đoàn Đấu vật Cổ đại châu Âu (EWA) được tổ chức vào ngày 28 tháng 10 năm 2017 tại Sofia, Bulgaria. Đây là sự kiện đấu vật cổ đại đầu tiên được tổ chức tại Bulgaria.
    
    ## Kết quả
    
    | **Sự kiện** | **Vô địch** | **Thách thức** | **Phương thức** | **Thời gian** |
    | EWA World Heavyweight Championship | The Bulgarian | The Russian |  |  |
    | EWA World Tag Team Championship | The Bulgarian | The Russian |  |  |
    | EWA World Cruiserweight Championship | The Bulgarian | The Russian |  |  |
    | EWA World Women's Championship | The Bulgarian | The Russian |  |  |
    | EWA World Junior Heavyweight Championship | The Bulgarian | The Russian |  |  |
    | EWA World Hardcore Championship | The Bulgarian | The Russian |  |  |
    | EWA World Street Fight Championship | The Bulgarian | The Russian |  |  |
    | EWA World Extreme Championship | The Bulgarian | The Russian |  |  |
    | EWA World Iron Man Championship | The Bulgarian | The Russian |  |  |
    | EWA World Intergender Championship | The Bulgarian | The Russian |  |  |
    | EWA World Mixed Tag Team Championship | The Bulgarian | The Russian |  |  |
    | EWA World Tag Team Championship | The Bulgarian | The Russian |  |  |
    | EWA World Television Championship | The Bulgarian | The Russian |  |  |
    | EWA World Women's Tag Team Championship | The Bulgarian | The Russian |  |  |


### Base model with LoRA


```python
from pathlib import Path
lora_path = Path("./models/mistral-nemo-base-lora/lora.safetensors")

tokenizer = MistralTokenizer.from_file("./models/mistral-nemo-base/tekken.json")  
model = Transformer.from_folder("./models/mistral-nemo-base") 
model.load_lora(lora_path)

prompt = "APT28 is"

completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=1024, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
```

     APT28, a Russian cyber espionage group also known as Fancy Bear, Pawn Storm, or Sofacy. The group has been active since at least 2004 and has made headlines frequently in recent years. APT28 is believed to be behind major cyber attacks, including the 2016 U.S. presidential election hacks, the 2017 French presidential campaign hacks, and many others. The group is also known for its involvement in the WannaCry ransomware attacks.
    
    ## APT28 Overview
    
    APT28 is a highly sophisticated and organized cyber espionage group that is known for its stealthy techniques and ability to adapt to new challenges. The group is believed to be a Russian state-sponsored organization, with a focus on gathering intelligence and conducting cyber operations against a range of targets, including governments, military organizations, and private entities.
    
    ### Key Points About APT28
    
    - APT28 is a well-funded and organized cyber espionage group that has been active for many years.
    - The group is known for its sophisticated techniques and ability to adapt to new challenges in the cyber landscape.
    - APT28 is believed to be a Russian state-sponsored organization, with a focus on gathering intelligence and conducting cyber operations against a range of targets.
    - The group has been involved in high-profile incidents, including the 2016 U.S. presidential election hacks and the 2017 French presidential campaign hacks.
    - APT28 is also known for its use of the WannaCry ransomware, which caused widespread disruptions in May 2017.
    
    ## APT28 Techniques
    
    APT28 is known for its sophisticated techniques and ability to adapt to new challenges in the cyber landscape. The group has been observed using a range of techniques, including:
    
    - **Spear phishing**: APT28 has been observed using spear phishing emails to deliver malware, which can be used to steal information or conduct other malicious activities.
    - **Watering hole attacks**: The group has been known to compromise websites frequently used by its targets, allowing them to deliver malware to visitors who match the targets' profile.
    - **Exploiting vulnerabilities**: APT28 has been observed exploiting vulnerabilities in software such as Java and Adobe Flash to compromise its targets.
    - **Malware development**: The group is known for its development of custom malware, including backdoors and other tools that can be used to conduct cyber operations.
    
    ## APT28 Tools
    
    APT28 is known for its development of custom malware, including:
    
    - **Sofacy**: A backdoor commonly used by APT28 that can be used to steal information or conduct other malicious activities.
    - **X-Agent**: A backdoor that is believed to be used exclusively by APT28. It has been observed being delivered through spear phishing emails or watering hole attacks.
    - **X-Tunnel**: A tool that is believed to be used to maintain access to compromised networks. It is often used in conjunction with other malware to provide a backup access channel.
    
    ## APT28 Targets
    
    APT28 has been observed targeting a range of organizations and sectors, including:
    
    - **Governments**: APT28 has been observed targeting governments worldwide, with a particular focus on NATO countries.
    - **Military organizations**: The group has been known to target military organizations, including the U.S. Department of Defense and the UK Ministry of Defense.
    - **Private companies**: APT28 has been observed targeting private companies, particularly those in the defense and aerospace sectors.
    - **Media organizations**: The group has been known to target media organizations, likely in an effort to gather intelligence or manipulate public opinion.
    
    ## APT28 Impact
    
    APT28 is known for its involvement in high-profile cyber attacks, including the 2016 U.S. presidential election hacks and the 2017 French presidential campaign hacks. The group's activities have had a significant impact on organizations and governments worldwide.
    
    - **Political influence**: APT28 is known for its attempts to influence politics and public opinion through cyber operations. This has included hacking political organizations and leaking sensitive information.
    - **Economic impact**: The group's cyber operations have been known to disrupt critical infrastructure, leading to financial losses and operational disruptions.
    - **Global impact**: APT28's activities are not limited to specific regions or sectors, and its operations can have a global impact.
    
    ## APT28 Mitigation Strategies
    
    To protect against APT28's cyber espionage activities, organizations can implement the following strategies:
    
    - **Cybersecurity awareness training**: Educating employees about cybersecurity threats and how to identify and respond to them is crucial in mitigating the risk of APT28 attacks.
    - **Regular security updates**: Keeping software and applications up to date is essential for patching vulnerabilities that A



```python

```
