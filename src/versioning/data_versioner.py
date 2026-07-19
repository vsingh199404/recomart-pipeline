import os
import json
import hashlib
import pandas as pd
from typing import Dict, Any
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {filepath}: {e}")
        return ""

def get_file_stats(filepath: str) -> Dict[str, Any]:
    """Get rows and columns if file is CSV or Parquet, plus file size."""
    stats = {"file_size": 0, "row_count": 0, "column_count": 0}
    if not os.path.exists(filepath):
        return stats
    
    stats["file_size"] = os.path.getsize(filepath)
    
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            stats["row_count"], stats["column_count"] = df.shape
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
            stats["row_count"], stats["column_count"] = df.shape
    except Exception as e:
        logger.warning(f"Could not read stats for {filepath}: {e}")
        
    return stats

def version_data(config: Dict[str, Any], feature_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Compute hashes and record versions and lineage for data files.
    feature_paths is expected to contain keys like 'raw_data', 'validated_data', 'prepared_data', 'features'.
    """
    logger.info("Starting data versioning and lineage tracking...")
    
    # Get versioning directory from config or fallback
    if 'paths' in config and 'versioning' in config['paths']:
        version_dir = config['paths']['versioning']
    else:
        version_dir = 'data/versioning'
        
    os.makedirs(version_dir, exist_ok=True)
    
    versions_file = os.path.join(version_dir, 'versions.json')
    lineage_file = os.path.join(version_dir, 'lineage.json')
    
    timestamp = datetime.now().isoformat()
    
    # Load existing versions
    all_versions = []
    if os.path.exists(versions_file):
        try:
            with open(versions_file, 'r') as f:
                all_versions = json.load(f)
        except json.JSONDecodeError:
            logger.warning("versions.json was corrupt or empty, starting fresh.")
            
    # Determine the next version number
    version_number = 1
    if all_versions:
        version_number = max([v.get('version_number', 0) for v in all_versions]) + 1
        
    current_versions = {}
    
    # Iterate over provided paths
    for key, path in feature_paths.items():
        if not os.path.exists(path):
            logger.warning(f"File {path} for key {key} does not exist. Skipping versioning.")
            continue
            
        file_hash = compute_sha256(path)
        stats = get_file_stats(path)
        
        meta = {
            "step": key,
            "filename": os.path.basename(path),
            "filepath": path,
            "hash": file_hash,
            "file_size": stats["file_size"],
            "row_count": stats["row_count"],
            "column_count": stats["column_count"],
            "source": "local",
            "transformation_applied": key,
            "version_number": version_number,
            "timestamp": timestamp
        }
        all_versions.append(meta)
        current_versions[key] = meta
        
    # Save versions
    with open(versions_file, 'w') as f:
        json.dump(all_versions, f, indent=4)
        
    # Create lineage
    lineage = {
        "timestamp": timestamp,
        "version_number": version_number,
        "nodes": [],
        "edges": []
    }
    
    # Order of expected steps
    steps_order = ['raw_data', 'validated_data', 'prepared_data', 'features']
    prev_key = None
    
    # Ensure any extra keys not in steps_order are included
    keys_to_process = steps_order + [k for k in current_versions.keys() if k not in steps_order]
    
    for step in keys_to_process:
        if step in current_versions:
            lineage["nodes"].append({
                "id": step,
                "label": step,
                "hash": current_versions[step]["hash"],
                "filename": current_versions[step]["filename"]
            })
            if prev_key:
                lineage["edges"].append({
                    "source": prev_key,
                    "target": step,
                    "transformation": f"{prev_key}_to_{step}"
                })
            prev_key = step
            
    # Save lineage
    existing_lineage = []
    if os.path.exists(lineage_file):
        try:
            with open(lineage_file, 'r') as f:
                existing_lineage = json.load(f)
        except json.JSONDecodeError:
            pass
            
    existing_lineage.append(lineage)
    with open(lineage_file, 'w') as f:
        json.dump(existing_lineage, f, indent=4)
        
    # Print summary table
    print(f"\n--- Data Versioning Summary (Version {version_number}) ---")
    print(f"{'Step':<15} | {'Filename':<25} | {'Rows':<8} | {'Cols':<5} | {'Size (B)':<10} | {'Hash (short)'}")
    print("-" * 88)
    for key, meta in current_versions.items():
        short_hash = meta['hash'][:8] + "..." if meta['hash'] else "N/A"
        print(f"{key:<15} | {meta['filename']:<25} | {meta['row_count']:<8} | {meta['column_count']:<5} | {meta['file_size']:<10} | {short_hash}")
    print("-" * 88)
    print(f"Versions saved to: {versions_file}")
    print(f"Lineage saved to: {lineage_file}\n")
    
    logger.info("Data versioning completed successfully.")
    
    return {
        "version_number": version_number,
        "timestamp": timestamp,
        "files_versioned": current_versions,
        "lineage": lineage
    }
