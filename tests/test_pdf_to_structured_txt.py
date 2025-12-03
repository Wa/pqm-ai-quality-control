import pdf_to_structured_txt as parser


def test_detect_vertical_frames_splits_on_large_gap():
    coords = [10, 20, 30, 200, 210, 220]
    frames = parser.detect_vertical_frames(coords, axis_length=300, gap_ratio=0.2)
    assert len(frames) == 2
    assert frames[0][0] == 0.0
    assert frames[0][1] <= frames[1][0]


def test_detect_vertical_frames_single_when_no_gap():
    coords = [10, 40, 70]
    frames = parser.detect_vertical_frames(coords, axis_length=120, gap_ratio=0.5)
    assert frames == [(0.0, 120)]


def test_detect_vertical_frames_histogram_fallback():
    coords = [20, 30, 40, 220, 230, 240]
    frames = parser.detect_vertical_frames(coords, axis_length=400, gap_ratio=0.6)
    assert len(frames) == 2
    assert frames[0][1] < frames[1][0]


def test_normalize_measurements_merges_tolerance_patterns():
    line = "235 + 1 / - 0 mm"
    normalized = parser.normalize_measurements(line)
    assert normalized == "235 +1/-0 mm"


def test_post_process_lines_merges_split_tolerance():
    lines = ["2×∅29.2+0.3", "-0.1"]
    processed = parser.post_process_lines(lines)
    assert processed[0] == "2×∅29.2 +0.3/-0.1"


def test_post_process_lines_skips_separator_tokens():
    lines = ["235+1", "/", "-0"]
    processed = parser.post_process_lines(lines)
    assert processed[0] == "235 +1/-0"


def test_post_process_lines_closes_parentheses():
    lines = ["PET(0.02 mm)+亚克力(0.01 mm"]
    processed = parser.post_process_lines(lines)
    assert processed[0].endswith(")")


def test_serialize_words_preserves_order():
    # (x0, y0, x1, y1, text, block, line, word)
    words = [
        (0, 0, 1, 1, "Hello", 0, 0, 0),
        (1, 0, 2, 1, "World", 0, 0, 1),
        (0, 1, 1, 2, "Next", 0, 1, 0),
    ]

    lines = parser.serialize_words(words)
    assert "Hello World" in lines[0]
    assert "Next" in lines[1]

