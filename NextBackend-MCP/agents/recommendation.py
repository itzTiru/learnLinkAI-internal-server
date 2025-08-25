from typing import List, Dict, Any

def recommend(ranked: List[Dict[str, Any]], top_k: int = 10, per_platform_limit: int = 10) -> List[Dict[str, Any]]:
    seen = set()
    counts = {}
    out = []
    for item in ranked:
        url = item.get("url")
        plat = item.get("platform","unknown")
        if url in seen:
            continue
        if counts.get(plat, 0) >= per_platform_limit:
            continue
        out.append(item)
        seen.add(url)
        counts[plat] = counts.get(plat, 0) + 1
        if len(out) >= top_k:
            break
    return out
