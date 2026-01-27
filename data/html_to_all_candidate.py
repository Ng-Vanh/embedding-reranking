import json
from bs4 import BeautifulSoup
from lxml import etree, html
import argparse
import os


def get_xpath(element, tree):
    """Get absolute xpath of an element"""
    try:
        path_parts = []
        current = element
        
        while current is not None:
            parent = current.getparent()
            if parent is None:
                path_parts.append(current.tag)
                break
            
            # Count position among siblings with same tag
            siblings = [child for child in parent if hasattr(child, 'tag') and child.tag == current.tag]
            if len(siblings) > 1:
                index = siblings.index(current) + 1
                path_parts.append(f"{current.tag}[{index}]")
            else:
                path_parts.append(current.tag)
            
            current = parent
        
        path_parts.reverse()
        return "/" + "/".join(path_parts)
    except Exception as e:
        return ""


def get_relative_xpath(element):
    """Get relative xpath using attribute-based selectors"""
    try:
        # Priority: id > name > class > other attributes
        id_attr = element.get('id', '')
        if id_attr:
            return f"//*[@id='{id_attr}']"
        
        name_attr = element.get('name', '')
        if name_attr:
            return f"//{element.tag}[@name='{name_attr}']"
        
        class_attr = element.get('class', '')
        if class_attr:
            return f"//{element.tag}[@class='{class_attr}']"
        
        # Check for other unique attributes
        for attr in ['href', 'src', 'type', 'value']:
            attr_value = element.get(attr, '')
            if attr_value:
                return f"//{element.tag}[@{attr}='{attr_value}']"
        
        # Fallback to position-based xpath
        parent = element.getparent()
        if parent is not None:
            siblings = [child for child in parent if hasattr(child, 'tag') and child.tag == element.tag]
            if len(siblings) > 1:
                index = siblings.index(element) + 1
                return f"//{element.tag}[{index}]"
        
        return f"//{element.tag}"
    except:
        return ""




def extract_href_text(href):
    """Extract meaningful text from href URL"""
    if not href:
        return ""
    
    # Skip non-useful hrefs
    skip_patterns = ['#', 'javascript:', 'mailto:', 'tel:']
    for pattern in skip_patterns:
        if href.startswith(pattern):
            return ""
    
    try:
        # Remove protocol and domain
        from urllib.parse import urlparse
        parsed = urlparse(href)
        path = parsed.path.strip('/')
        
        if not path:
            return ""
        
        # Remove common prefixes
        prefixes_to_remove = ['c/', 'category/', 'tag/', 'page/']
        for prefix in prefixes_to_remove:
            if path.startswith(prefix):
                path = path[len(prefix):]
        
        # Convert path to readable text
        # Replace hyphens and slashes with spaces
        text = path.replace('-', ' ').replace('/', ' ')
        
        # Remove file extensions
        text = text.split('.')[0] if '.' in text else text
        
        # Clean up and limit length
        text = ' '.join(text.split())  # Remove extra spaces
        
        # Limit to reasonable length
        words = text.split()
        if len(words) > 8:
            text = ' '.join(words[:8])
        
        return text
    except:
        return ""


def get_element_text(element):
    """Extract only direct text from element (not including children), limited to first 100 chars"""
    try:
        # Get only direct text of this element, not text from children
        text = element.text or ""
        text = text.strip()
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Limit length
        if len(text) > 100:
            text = text[:100] + "..."
        return text
    except:
        return ""


def format_candidate(element):
    """Format candidate as [TAG] tag [ID]id[CLASS] class [TITLE]title [HREF]href"""
    tag = element.tag if hasattr(element, 'tag') else ""
    id_attr = element.get('id', '')
    classes = element.get('class', '')
    title_attr = element.get('title', '')
    href_attr = element.get('href', '')
    place_holder = element.get('placeholder', '')
    aria_label = element.get('aria-label', '')

    text = get_element_text(element)
    
    # Filter classes to remove noise
    filtered_classes = filter_classes(classes)
    
    # Extract meaningful href text
    href_text = extract_href_text(href_attr) if tag == 'a' else ""
    
    # Format: [TAG] tag_name [ID]id_value[CLASS] class_value [TITLE]title_value [HREF]href_text [TEXT]text_content
    parts = []
    parts.append(f"[TAG] {tag}")
    
    if id_attr:
        parts.append(f"[ID] {id_attr}")
    
    if filtered_classes:
        parts.append(f"[CLASS] {filtered_classes}")
    
    if title_attr or aria_label:
        title_value = title_attr if title_attr else aria_label
        parts.append(f"[TITLE] {title_value}")
    
    if href_text:
        parts.append(f"[HREF] {href_text}")
    
    if text:
        parts.append(f"[TEXT] {text}")
    
    if place_holder:
        parts.append(f"[PLACEHOLDER] {place_holder}")
    
    return " ".join(parts)


def is_interactive_element(element):
    """
    Check if an element is potentially interactive
    Interactive elements include: buttons, links, inputs, selects, textareas,
    and elements with onclick, role, or tabindex attributes
    """
    if not hasattr(element, 'tag'):
        return False
    
    # Interactive tags
    interactive_tags = {
        'a', 'button', 'input', 'select', 'textarea', 
        'option', 'label', 'form', 'details', 'summary'
    }
    
    if element.tag in interactive_tags:
        return True
    
    # Check for interactive attributes
    interactive_attrs = ['onclick', 'onchange', 'onsubmit', 'role', 'tabindex']
    for attr in interactive_attrs:
        if element.get(attr):
            return True
    
    # Check for role attribute suggesting interactivity
    role = element.get('role', '')
    interactive_roles = {'button', 'link', 'checkbox', 'radio', 'tab', 'menuitem', 'option'}
    if role in interactive_roles:
        return True
    
    return False


def get_context_text(element, max_length=50):
    """Get text context from parent or siblings"""
    try:
        parent = element.getparent()
        if parent is not None:
            # Get parent's direct text
            context = parent.text or ""
            context = context.strip()
            context = " ".join(context.split())
            if len(context) > max_length:
                context = context[:max_length]
            return context
    except:
        pass
    return ""


def generate_queries(element):
    """
    Generate multiple natural language queries for an element
    Returns list of query strings
    """
    queries = []
    tag = element.tag
    text = get_element_text(element)
    id_attr = element.get('id', '')
    title_attr = element.get('title', '')
    href_attr = element.get('href', '')
    name_attr = element.get('name', '')
    type_attr = element.get('type', '')
    value_attr = element.get('value', '')
    placeholder = element.get('placeholder', '')
    aria_label = element.get('aria-label', '')
    
    # Extract href text for links
    href_text = extract_href_text(href_attr) if tag == 'a' else ""
    
    # Method 1: Action-based queries (based on element type)
    if tag == 'button':
        if text:
            queries.append(f"Click on {text}")
            queries.append(f"Press the {text} button")
            queries.append(f"Click {text} button")
        if title_attr:
            queries.append(f"Click button with title {title_attr}")
    
    elif tag == 'a':
        if text:
            queries.append(f"Click on {text}")
            queries.append(f"Navigate to {text}")
            queries.append(f"Go to {text}")
        if aria_label:
            queries.append(f"Click link {aria_label}")
        if href_text:
            queries.append(f"Go to {href_text}")
            queries.append(f"Navigate to {href_text}")
        if title_attr:
            queries.append(f"Click link {title_attr}")
    
    elif tag == 'input':
        if placeholder:
            queries.append(f"Enter text in {placeholder}")
            queries.append(f"Type in {placeholder} field")

        if type_attr == 'text' or type_attr == 'search':
            if placeholder:
                queries.append(f"Enter text in {placeholder}")
                queries.append(f"Type in {placeholder} field")
            if name_attr:
                queries.append(f"Enter {name_attr}")
                queries.append(f"Fill {name_attr} field")
            if aria_label:
                queries.append(f"Enter text in {aria_label}")
        
        elif type_attr == 'submit':
            if value_attr:
                queries.append(f"Click {value_attr}")
                queries.append(f"Press {value_attr} button")
            else:
                queries.append("Submit form")
        
        elif type_attr == 'checkbox':
            if text or aria_label:
                label = text or aria_label
                queries.append(f"Check {label}")
                queries.append(f"Select {label} checkbox")
        
        elif type_attr == 'radio':
            if text or aria_label:
                label = text or aria_label
                queries.append(f"Select {label}")
                queries.append(f"Choose {label} option")
    
    elif tag == 'select':
        if name_attr:
            queries.append(f"Select from {name_attr} dropdown")
            queries.append(f"Choose {name_attr}")
        if aria_label:
            queries.append(f"Select {aria_label}")
    
    elif tag == 'textarea':
        if name_attr:
            queries.append(f"Enter text in {name_attr}")
            queries.append(f"Type in {name_attr} area")
        if placeholder:
            queries.append(f"Enter {placeholder}")
    
    # Method 2: Attribute-based queries
    if id_attr:
        # Convert id to natural language
        readable_id = id_attr.replace('_', ' ').replace('-', ' ')
        queries.append(f"Click {readable_id}")
        queries.append(f"Select {readable_id}")
    
    # Method 3: Title/aria-label based
    if title_attr and title_attr not in queries:
        queries.append(f"Click {title_attr}")
    
    if aria_label and aria_label not in queries:
        queries.append(f"Click {aria_label}")
    
    # Method 4: Simple text-based
    if text and len(queries) < 2:
        queries.append(f"Click {text}")
        queries.append(f"Select {text}")
    
    # Method 5: Context-based queries
    context = get_context_text(element)
    if context and text:
        queries.append(f"Click {text} in {context}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)
    
    # Limit to top 5 most relevant queries
    return unique_queries[:5]


def extract_interactive_nodes(html_content):
    """Extract all interactive nodes from HTML content"""
    # Parse HTML with lxml
    tree = html.fromstring(html_content)
    
    candidates = []
    
    # Traverse all elements
    for element in tree.iter():
        if is_interactive_element(element):
            try:
                # Get raw HTML of the element (only opening tag with attributes, no children)
                tag = element.tag
                attrs = ' '.join([f'{k}="{v}"' for k, v in element.attrib.items()])
                if attrs:
                    raw_candidate = f"<{tag} {attrs}>"
                else:
                    raw_candidate = f"<{tag}>"
                
                # Generate queries for this node
                queries = generate_queries(element)
                
                # Create candidate object
                candidate_obj = {
                    "candidate": format_candidate(element),
                    "absolute_xpath": get_xpath(element, tree),
                    "relative_xpath": get_relative_xpath(element),
                    "raw_candidate": raw_candidate,
                    "queries": queries
                }
                
                candidates.append(candidate_obj)
            except Exception as e:
                print(f"Error processing element: {e}")
                continue
    
    return candidates


def process_html_file(html_path, output_path=None):
    """
    Process HTML file and extract interactive candidates
    
    Args:
        html_path: Path or URL to HTML file
        output_path: Path to save JSON output (optional)
    
    Returns:
        List of candidate objects
    """
    # Read HTML content
    if html_path.startswith('http://') or html_path.startswith('https://'):
        import requests
        response = requests.get(html_path)
        html_content = response.text
    else:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    
    # Extract candidates
    candidates = extract_interactive_nodes(html_content)
    
    print(f"Found {len(candidates)} interactive candidates")
    
    # Save to JSON if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_path}")
    
    return candidates


def main():
    parser = argparse.ArgumentParser(description='Extract interactive nodes from HTML file')
    parser.add_argument('html_path', type=str, help='Path or URL to HTML file')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Generate default output path if not provided
    output_path = args.output
    if not output_path:
        base_name = os.path.basename(args.html_path).replace('.html', '')
        output_path = f"{base_name}_candidates.json"
    
    candidates = process_html_file(args.html_path, output_path)
    
    # Print sample
    if candidates:
        print("\nSample candidate:")
        print(json.dumps(candidates[0], ensure_ascii=False, indent=2))

def filter_classes(classes):
    """Filter out noise classes that don't carry semantic meaning"""
    if not classes:
        return ""
    
    drop_keywords = [
        "color", "font", "hover", "spacing", "letter-spacing",
        "margin", "padding", "size", "weight", "width", "height",
        "u-", "lrv-u-", "lrv-a-", "xs", "sm", "md", "lg", "xl",
        "display", "flex", "grid", "align", "justify",
        "border", "radius", "shadow", "opacity",
        "transition", "transform", "animation","c-lazy-image__link", 
        "c-title__link","c-link","c-span__link","icon-","-icon","ot-",
        "sc-","css-","e1iflr850","ell52qj1","tpl-","kyt-","byline",
        "__link","__link--","article__","article-","button--","__text__","module__",
        "dcr-","lpt-","lmr-","lpb-","hdp","hao","hdb","-a","iwc","ib","lpl","hax","lpr","fs14",
        "cuzo","ksbe","wp-","_link_","contrast","x:","--magazine","mntl-","-scroll",
        "_6o3a","_1fr089","_1pmvkj","rccwc","_2srr0x1","usntA42","__button","header__",
        "fides-","-green-","-400","-500","-600","-700","-800","-900","-dark","-light",
        "px-","py-","pt-","pb-","pl-","pr-","mx-","my-","mt-","mb-","ml-","mr-","group",
        "text-","items","inset-","[99999]","h-100", "btn","fw-","p-0","nav-item","nav-link",
        "d-","justify-","align-","self-","content-","basis-","-hidden","more-link","entry-title-link",
        "prev-link","cmpbox","ConsumerMarketingUnitThemedWrapper-jkpAEW","hssEkF","svelte",
        "typo-s","ease-in-out","duration-","hover:","focus:","md:","lg:","xl:","2xl:",
        "ursor-pointer","no-underline","dark:","darkbg-","darktext-","darkborder-",
        "bg-","textcenter","text-center","textleft","text-left","textright","text-right",
        "uppercase","lowercase","capitalize","normalcase","gap-","-full","kbkde","cpkA",
        "hnPKxn","fdHTcp","jaFDQv","ldMimX","astro-","p-16","type-link","type-h4",
        "osano","comp","js-","cky-hide"
    ]
    
    class_list = classes.split()
    filtered_classes = []
    
    for cls in class_list:
        # Check if class contains any drop keyword
        should_drop = False
        for keyword in drop_keywords:
            if keyword in cls.lower():
                should_drop = True
                break
        
        if not should_drop:
            filtered_classes.append(cls)
    
    return " ".join(filtered_classes) if filtered_classes else ""

if __name__ == "__main__":
    # Example usage
    raw_html = "/mnt/disk2/anhnv/rr/stage1/data/raw_html_train/weworkremote.html"
    
    # Uncomment to run with hardcoded path
    candidates = process_html_file(raw_html, "/mnt/disk2/anhnv/rr/stage1/data/raw_html_train/weworkremote_candidates.json")
    
    # Or use command line arguments
    # main()