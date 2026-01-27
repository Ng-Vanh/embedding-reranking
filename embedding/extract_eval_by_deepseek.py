import json
from tqdm import tqdm

API_KEY="sk-88f2a5a4476d49fb81664f87d5e107be"  
BASE_URL="https://api.deepseek.com"
MODEL="deepseek-reasoner"  
from openai import OpenAI
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)
inp = "/home/anhnd/gendata-websrc/stage1/stage1_eval.json"
out = "/home/anhnd/gendata-websrc/stage1/stage1_eval_deepseek.json"

with open(inp, "r", encoding="utf-8") as f:
    data = json.load(f)
new_data = []
for item in tqdm(data, desc="Processing eval items"):
    query = item["query"]
    positive = item["positive"]
    prompt = f"""
###  System / Instruction

> You are a **web automation data expert**.
> Your task is to generate **hard negative UI nodes** for training a model that selects the correct DOM element for a given user action.
>
> The goal is to make the negatives **semantically close** to the positive but **incorrect**, so the model learns fine-grained distinctions.

---

###  Input Format

You will be given:

```json
{{
  "query": "...",
  "positive": "..."
}}
```

Where:

* `query` contains:

  * `[ACTION]`: the current user intent
  * `[HISTORY]`: previous UI interactions
* `positive` is the **correct DOM node** for the action

---

###  Node Format (STRICT)

 **ALL nodes MUST follow exactly this format**:

```
[TAG] <tag_name> [CLASS] <class_name> [TEXT] <visible_text>
```

Rules:

* Do NOT add extra fields
* Do NOT change the format
* Text should look like **real UI text**

---

###  Hard Negative Definition

Hard negatives must:

* Be **plausible UI elements on the same page**
* Be **semantically similar** to the positive
* Be **wrong for the given action**
* Differ subtly by:

  * Sorting criteria (price vs distance vs rating)
  * Related but different filters
  * Same component, wrong option
  * Similar wording

 Do NOT include:

* Random or unrelated elements
* Duplicates
* Elements that exactly match the positive

---

### 🔹 Output Format

```json
{{
  "positive": "...",
  "hard_negatives": [
    "...",
    "...",
    ...
  ]
}}
```

Generate **exactly 50 hard negative nodes**.

---

##  FEW-SHOT EXAMPLE

###  Example Input

```json
{{
  "query": "[ACTION] Find the cheapest wheelchair accessible parking for the Pittsburgh Pirates at St. Louis Cardinals event in Busch Stadium on Apr 13. [HISTORY] [button] Filter -> CLICK; [checkbox] Wheelchair Accessible (10) -> CLICK; [button] Show 10 Results -> CLICK",
  "positive": "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Distance Sort by Price"
}}
```

---

###  Example Output (TRUNCATED)

```json
{{
  "positive": "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Distance Sort by Price",
  "hard_negatives": [
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Price",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Distance",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Rating",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Best Match",
    "[TAG] button [CLASS] FormElement-item [TEXT] Apply Sort",
    "[TAG] button [CLASS] FormElement-item [TEXT] Filter Results",
    "[TAG] checkbox [CLASS] FormElement-item [TEXT] Accessible Parking",
    "[TAG] checkbox [CLASS] FormElement-item [TEXT] Handicap Accessible",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Distance Only",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Lowest Price",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Closest",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Cheapest",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Distance and Time",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Price and Rating",
    "[TAG] select [CLASS] FormElement-item [TEXT] Sort by Popularity"
    // ... total 50
  ]
}}
```
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1.0
        )
        
        content = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object in the content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content
            
            result = json.loads(json_str)
            if "hard_negatives" in result and isinstance(result["hard_negatives"], list):
                item["hard_negatives"] = result["hard_negatives"]
            else:
                print(f"Warning: Unexpected response format for item, using empty list")
                print(f"Response content: {content[:500]}")
                item["hard_negatives"] = []
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON response for item, using empty list")
            print(f"JSON error: {e}")
            print(f"Response content: {content[:500]}")
            item["hard_negatives"] = []
            
    except Exception as e:
        print(f"Error processing item: {e}")
        item["hard_negatives"] = []
    
    new_data.append(item)
    
    # Save progress periodically (every 10 items)
    if len(new_data) % 10 == 0:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

# Final save
with open(out, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"Processing complete. Results saved to {out}")
print(f"Total items processed: {len(new_data)}")
