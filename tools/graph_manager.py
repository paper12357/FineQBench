import os
import json
import difflib
from typing import List, Dict, Optional

class GraphManager:
    def __init__(self, db_path: list):

        nodes_path = db_path[0]
        edges_path = db_path[1]

        self.nodes: List[Dict] = []
        self.edges: List[Dict] = []
        try:
            with open(nodes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.nodes = data.get("nodes", data) if isinstance(data, dict) else data
        except FileNotFoundError:
            raise FileNotFoundError(f"kg_nodes.json not found at: {nodes_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load nodes file {nodes_path}: {e}")

        try:
            with open(edges_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.edges = data.get("edges", data) if isinstance(data, dict) else data
        except FileNotFoundError:
            raise FileNotFoundError(f"kg_edges.json not found at: {edges_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load edges file {edges_path}: {e}")

        self.name_list = [node.get("name", "") for node in self.nodes]

    def _best_match(self, query: str, candidates: List[str], cutoff: float = 0.6) -> Optional[str]:
        if not query or not candidates:
            return None
        lower_map = {}
        lower_list = []
        for c in candidates:
            lc = c.lower()
            if lc not in lower_map:
                lower_map[lc] = c
                lower_list.append(lc)

        ql = query.lower()
        matches = difflib.get_close_matches(ql, lower_list, n=3, cutoff=cutoff)
        if matches:
            ret = []
            for m in matches[:3]:
                ret.append(lower_map[m])
            return ret
        
        for lc in lower_list:
            if ql in lc or lc in ql:
                return [lower_map[lc]]
        return []

    def search_entity(self, query_text: str) -> str:
        match_name = self._best_match(query_text, self.name_list, cutoff=0.6)
        if len(match_name) == 0:
            return f"No matching entity found for query: '{query_text}'."
        
        lines = []

        for name in match_name:
            matched_node = next((n for n in self.nodes if n.get("name") == name), None)
            if not matched_node:
                return f"Matched name '{name}' but node record not found."

            lines.append(f"Name: {matched_node.get('name')}")
            descs = matched_node.get("descriptions") or matched_node.get("description") or []
            if isinstance(descs, str):
                descs = [descs]
            lines.append("Descriptions:")
            if descs:
                for d in descs:
                    lines.append(f"  - {d}")
            else:
                lines.append("  - (no descriptions)")
            if "count" in matched_node:
                lines.append(f"Count: {matched_node.get('count')}")
            lines.append("-" * 40)
        
        return "\n".join(lines)

    def search_relation(self, entity1: str, entity2: str) -> str:
        if not entity1 or not entity2:
            return "Both entity1 and entity2 must be provided."

        match1 = self._best_match(entity1, self.name_list, cutoff=0.6)
        match2 = self._best_match(entity2, self.name_list, cutoff=0.6)

        if len(match1) == 0 and len(match2) == 0:
            return f"No matches found for '{entity1}' or '{entity2}'."
        if len(match1) == 0:
            return f"No match found for entity1: '{entity1}'."
        if len(match2) == 0:
            return f"No match found for entity2: '{entity2}'."

        matched_edges = []
        for e in self.edges:
            src = e.get("source")
            tgt = e.get("target")
            if src in match1 and tgt in match2:
                matched_edges.append(("forward", e))
            elif src in match2 and tgt in match1:
                matched_edges.append(("reverse", e))

        if not matched_edges:
            return f"No relation found between '{match1}' and '{match2}'."

        out_lines = [f"Found {len(matched_edges)} relation(s) between '{match1}' and '{match2}':"]
        for direction, e in matched_edges:
            out_lines.append("-" * 30)
            out_lines.append(f"Direction: {direction}")
            out_lines.append(f"Type: {e.get('type', '(no type)')}")
            out_lines.append(f"Source: {e.get('source')}")
            out_lines.append(f"Target: {e.get('target')}")
            descs = e.get("descriptions") or e.get("description") or []
            if isinstance(descs, str):
                descs = [descs]
            out_lines.append("Descriptions:")
            if descs:
                for d in descs:
                    out_lines.append(f"  - {d}")
            else:
                out_lines.append("  - (no descriptions)")
            if "weight" in e:
                out_lines.append(f"Weight: {e.get('weight')}")
        return "\n".join(out_lines)
