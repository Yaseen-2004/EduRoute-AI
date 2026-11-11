from src.data_collection.wikitext import wikitext_to_text
from src.rag.text_cleaning import clean_article

SAMPLE = """{{Short description|Thermodynamic cycle}}
{{Infobox thermodynamic cycle
| name = Rankine {{nowrap|cycle}}
}}
[[File:Rankine cycle layout.png|thumb|Layout of the [[Rankine cycle]] with a [[boiler]].]]
The '''Rankine cycle''' is an idealized [[thermodynamic cycle]] of a [[heat engine]]<ref>{{cite book|title=Thermo|year=2001}}</ref> that converts heat into work.<ref name="a"/>
Its efficiency is {{math|''η'' {{=}} ''W''/''Q''}} and <math>\\eta = 1 - T_c/T_h</math> at best.

== Description ==
* The [[pump]] raises the pressure.
* The [[Boiler|boiler]] heats the water.
{| class="wikitable"
|-
| row || data
|}
<!-- hidden note -->

== References ==
{{reflist}}
* Smith, J. (2001). ''Thermo''. ISBN 123.
[[Category:Thermodynamic cycles]]
"""


def test_wikitext_conversion_keeps_prose_headings_and_formulas():
    text = wikitext_to_text(SAMPLE)
    assert "The Rankine cycle is an idealized thermodynamic cycle of a heat engine that converts heat into work." in text
    assert "== Description ==" in text
    assert "The pump raises the pressure." in text and "The boiler heats the water." in text
    assert "T_c/T_h" in text                      # <math> kept
    assert "W/Q" in text.replace(" ", "")        # {{math}} kept
    for junk in ("Infobox", "cite book", "File:", "Category:", "wikitable", "hidden note", "<ref", "'''"):
        assert junk not in text, junk


def test_converted_text_flows_through_section_cleaner():
    sections, dropped = clean_article(wikitext_to_text(SAMPLE))
    body = " ".join(s["text"] for s in sections)
    assert "ISBN" not in body and "References" in dropped["sections"]
    assert [s["heading_path"] for s in sections] == [[], ["Description"]]
