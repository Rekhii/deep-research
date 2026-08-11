from src.ingest import clean


def test_strips_bold_markers():
    assert clean("this is **bold** text") == "this is bold text"


def test_strips_italic_markers_but_keeps_internal_underscores():
    # snake_case identifiers and similar should survive
    assert clean("_emphasis_ and snake_case_name") == "emphasis and snake_case_name"


def test_collapses_blank_line_runs():
    assert clean("first\n\n\n\n\nsecond") == "first\n\nsecond"


def test_strips_glued_footnote_digit_before_proper_noun():
    # pymupdf4llm flattens superscripts onto the baseline, producing "6Barsalou"
    assert clean("as noted by 6Barsalou") == "as noted by Barsalou"


def test_strips_glued_footnote_digit_before_acronym():
    assert clean("see 3GWT for details") == "see GWT for details"


def test_repairs_lida_with_trailing_marker():
    # superscript markers glued to LIDA produced LIDAC, LIDAI and similar
    assert clean("the LIDAC model") == "the LIDA model"


def test_leaves_real_words_starting_with_lida_alone():
    # the rule must not fire on longer legitimate strings
    assert clean("LIDA is described") == "LIDA is described"


def test_promotes_glued_section_heading_to_atx():
    # pymupdf4llm missed one heading and emitted it inline after a sentence
    text = "modulate the action [117]. VIII. LIDA AND THE UNDERLYING NEURAL PROCESSES As emphasized"
    result = clean(text)
    assert "\n\n## VIII. LIDA AND THE UNDERLYING NEURAL PROCESSES" in result


def test_does_not_promote_ordinary_cross_reference():
    # a numeral followed by only one capitalised word is not a heading
    text = "described in IV. Above we noted"
    assert "##" not in clean(text)


def test_removes_ieee_copyright_notice():
    text = "real content here. Copyright (c) 2014 IEEE. Personal use is permitted. more content"
    result = clean(text)
    assert "Copyright" not in result
    assert "real content here." in result
    assert "more content" in result