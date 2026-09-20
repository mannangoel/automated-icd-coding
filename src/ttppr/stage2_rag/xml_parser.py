# src/ttppr/stage2_rag/xml_parser.py
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_cms_icd10_xml(
    xml_path: str | Path, output_json_path: str | Path
) -> None:
    """Parses CMS ICD-10-CM Tabular List XML file into a structured JSON file

    containing codes, descriptions, Excludes1, and Excludes2 instruction
    rules.
    """
    xml_path = Path(xml_path)
    output_json_path = Path(output_json_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"CMS XML file not found at expected path: {xml_path.resolve()}"
        )

    print(f"Loading and parsing XML file: {xml_path} ...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    parsed_records = []

    # Iterate over every diagnosis element (<diag>) in the XML hierarchy
    for diag in root.iter("diag"):
        name_node = diag.find("name")
        desc_node = diag.find("desc")

        if name_node is None or desc_node is None:
            continue

        code = name_node.text.strip() if name_node.text else ""
        description = desc_node.text.strip() if desc_node.text else ""

        # Extract Excludes1 notes (codes that can NEVER be used together)
        excludes1 = []
        excludes1_node = diag.find("excludes1")
        if excludes1_node is not None:
            for note in excludes1_node.findall(".//note"):
                if note.text:
                    excludes1.append(note.text.strip())

        # Extract Excludes2 notes (conditions not included here, but can be coded separately)
        excludes2 = []
        excludes2_node = diag.find("excludes2")
        if excludes2_node is not None:
            for note in excludes2_node.findall(".//note"):
                if note.text:
                    excludes2.append(note.text.strip())

        parsed_records.append({
            "code": code,
            "description": description,
            "excludes1": excludes1,
            "excludes2": excludes2,
        })

    # Ensure output destination directory exists
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to formatted JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_records, f, indent=2)

    print(
        f"[✔] Successfully parsed {len(parsed_records)} ICD-10 codes/rules to '{output_json_path}'."
    )