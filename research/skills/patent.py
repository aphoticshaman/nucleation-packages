# packages/research/patent_miner.py
"""
Patent Mining Utility for LatticeForge
Extracts mathematical frameworks from expired patents
"""

import requests
import json
import fitz
import re
import os
import spacy
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time

OUTPUT_DIR = "patent_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

# IPC/CPC Classification Mapping
MATH_TAXONOMY = {
    "kalman_filtering": ["G06F17/18", "G06N7/02", "G01S13/00"],
    "information_theory": ["G06F17/16", "G06N99/00"],
    "topological_data_analysis": ["G06F17/30", "G06N3/04"],
    "spectral_graph": ["G06F17/27", "G06N7/02"],
    "stochastic_sde": ["G06Q50/00", "G06F17/14"],
    "bayesian_nonparametric": ["G06N7/02", "G06F17/16"],
    "compressed_sensing": ["G06F17/18", "H04L27/26"],
    "copula_functions": ["G06Q40/00", "G06F17/16"],
    "wavelet_analysis": ["G06F17/18", "H04N21/00"],
    "reinforcement_learning": ["G06N5/00", "G06N7/02"],
}

PRIORITY_ASSIGNEES = [
    "Raytheon", "Lockheed Martin", "Northrop Grumman",
    "Bloomberg", "Refinitiv", "Two Sigma", "Citadel",
    "Nokia Bell Labs", "Qualcomm", "IBM Research",
    "MIT", "Stanford", "CMU", "Caltech"
]

@dataclass
class PatentExtract:
    patent_id: str
    title: str
    assignee: str
    expiration_date: str
    math_category: str
    equations: List[str]
    pseudocode: List[str]
    claims: List[str]
    relevance_score: float


def extract_equations(text: str) -> List[str]:
    """Extract LaTeX and inline equations from patent text"""
    patterns = [
        r"\$\$.*?\$\$",  # Display math
        r"\$[^$]+\$",    # Inline math
        r"[A-Za-z_]\s*=\s*[A-Za-z0-9_\+\-\*/\^\(\)\s]+",  # Assignments
        r"∑|∫|∂|∇|Σ|Π|λ|μ|σ|θ|α|β|γ",  # Greek/math symbols indicate equations nearby
        r"argmin|argmax|max|min|log|exp|sin|cos|tanh",  # Common functions
    ]
    equations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        equations.extend([m.strip() for m in matches if len(m) > 3])
    return list(set(equations))


def extract_pseudocode(text: str) -> List[str]:
    """Extract algorithmic steps and pseudocode blocks"""
    doc = nlp(text)
    steps = []
    triggers = [
        r"step\s*\d*", r"initialize", r"compute", r"update", 
        r"predict", r"iterate", r"for\s+each", r"while", 
        r"if\s+", r"return", r"output", r"input",
        r"algorithm\s*\d*", r"procedure", r"function"
    ]
    pattern = re.compile("|".join(triggers), re.IGNORECASE)
    
    for sent in doc.sents:
        if pattern.search(sent.text):
            steps.append(sent.text.strip())
    return steps


def extract_claims(text: str) -> List[str]:
    """Extract patent claims (where the math usually lives)"""
    claims = []
    claim_pattern = re.compile(r"(?:Claim\s*\d+[:\.]?\s*)(.+?)(?=Claim\s*\d+|$)", re.DOTALL | re.IGNORECASE)
    matches = claim_pattern.findall(text)
    for match in matches[:10]:  # First 10 claims usually most relevant
        clean = re.sub(r"\s+", " ", match).strip()
        if len(clean) > 50:
            claims.append(clean[:500])  # Truncate long claims
    return claims


def score_relevance(text: str, category: str) -> float:
    """Score patent relevance to signal fusion / anomaly detection"""
    keywords = {
        "signal": 3, "fusion": 4, "anomaly": 5, "detection": 3,
        "prediction": 3, "forecast": 3, "time series": 4,
        "real-time": 3, "streaming": 2, "sensor": 2,
        "financial": 2, "market": 2, "regime": 5,
        "phase transition": 5, "threshold": 3, "adaptive": 3,
        "multivariate": 3, "correlation": 2, "covariance": 2,
        "uncertainty": 3, "confidence": 2, "probability": 2
    }
    text_lower = text.lower()
    score = sum(weight for kw, weight in keywords.items() if kw in text_lower)
    return min(score / 50.0, 1.0)  # Normalize to 0-1


def query_uspto_expired(cpc_codes: List[str], start_date: str, end_date: str) -> List[Dict]:
    """
    Query USPTO for expired patents in date range
    Note: Requires USPTO API key for production use
    """
    base_url = "https://developer.uspto.gov/ibd-api/v1/patent/application"
    results = []
    
    for cpc in cpc_codes:
        params = {
            "searchText": f"cpcCodes:{cpc}",
            "start": 0,
            "rows": 100,
        }
        try:
            # USPTO API requires authentication in production
            # This is a placeholder for the actual implementation
            time.sleep(0.5)  # Rate limiting
            # response = requests.get(base_url, params=params)
            # results.extend(response.json().get("results", []))
        except Exception as e:
            print(f"Error querying USPTO for {cpc}: {e}")
    
    return results


def process_patent_pdf(pdf_path: str, category: str) -> Optional[PatentExtract]:
    """Process a downloaded patent PDF"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        equations = extract_equations(full_text)
        pseudocode = extract_pseudocode(full_text)
        claims = extract_claims(full_text)
        relevance = score_relevance(full_text, category)
        
        # Extract metadata from first page
        first_page = doc[0].get_text()
        patent_id = re.search(r"US\s*[\d,]+", first_page)
        
        return PatentExtract(
            patent_id=patent_id.group() if patent_id else "Unknown",
            title=first_page[:200].split("\n")[0],
            assignee="TBD",  # Would parse from structured data
            expiration_date="TBD",
            math_category=category,
            equations=equations[:20],  # Top 20
            pseudocode=pseudocode[:15],
            claims=claims,
            relevance_score=relevance
        )
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def generate_search_queries() -> Dict[str, str]:
    """Generate optimized search queries for each patent database"""
    queries = {}
    
    date_filter = "expiration:[2022-12-01 TO 2024-12-01]"
    assignee_filter = " OR ".join([f'assignee:"{a}"' for a in PRIORITY_ASSIGNEES])
    
    for category, cpc_codes in MATH_TAXONOMY.items():
        cpc_filter = " OR ".join([f"cpc:{c}" for c in cpc_codes])
        queries[category] = f"({cpc_filter}) AND ({assignee_filter}) AND {date_filter}"
    
    return queries


def export_findings(extracts: List[PatentExtract], output_path: str):
    """Export findings to JSON for engineering consumption"""
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_patents": len(extracts),
        "by_category": {},
        "high_relevance": []
    }
    
    for extract in extracts:
        cat = extract.math_category
        if cat not in output["by_category"]:
            output["by_category"][cat] = []
        output["by_category"][cat].append({
            "patent_id": extract.patent_id,
            "title": extract.title,
            "relevance": extract.relevance_score,
            "equations": extract.equations,
            "pseudocode": extract.pseudocode,
            "key_claims": extract.claims[:3]
        })
        
        if extract.relevance_score > 0.7:
            output["high_relevance"].append(extract.patent_id)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Exported {len(extracts)} patents to {output_path}")


if __name__ == "__main__":
    # Generate search queries for manual use in patent databases
    queries = generate_search_queries()
    print("=== USPTO/EPO/WIPO Search Queries ===\n")
    for category, query in queries.items():
        print(f"[{category}]")
        print(f"  {query}\n")
