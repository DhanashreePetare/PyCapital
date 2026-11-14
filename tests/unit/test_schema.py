from pycapital.schema import SCHEMAS, empty_frame

def test_empty_frames_have_standard_columns():
    for name, cols in SCHEMAS.items():
        df = empty_frame(name)
        assert df.empty
        assert list(df.columns) == list(cols)