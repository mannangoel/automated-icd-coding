# scripts/1_parse_xml.py
from src.ttppr.stage2_rag.xml_parser import parse_cms_icd10_xml

if __name__ == "__main__":
    raw_xml_path = "data/raw/cms/icd10cm_tabular_2027.xml"
    output_json_path = "data/processed/icd10_rules_2027.json"

    print("Starting CMS ICD-10 XML Parsing Job...")
    parse_cms_icd10_xml(raw_xml_path, output_json_path)