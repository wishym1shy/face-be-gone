"""Turn failures in a pytest JUnit XML report into GitHub Actions error
annotations, so they show on the PR without opening the raw logs.

    python tests/annotate.py report.xml
"""

import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
for case in root.iter("testcase"):
    for problem in list(case.findall("failure")) + list(case.findall("error")):
        name = f"{case.get('classname')}::{case.get('name')}"
        text = (problem.text or problem.get("message") or "").strip()
        text = text[-3000:].replace("%", "%25").replace("\r", "").replace("\n", "%0A")
        print(f"::error title={name}::{text}")
